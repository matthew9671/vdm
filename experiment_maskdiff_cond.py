import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple

from vdm.experiment import Experiment
import vdm.transformer as transformer
from vdm.nets import vqgan_tokenizer

from jax.nn import softmax, log_softmax

from functools import partial
import ml_collections
import vdm.train_state
import vdm.utils as utils
import vdm.dataset as dataset
from absl import logging

from clu import parameter_overview
from clu import checkpoint

import tensorflow.compat.v1 as tf
import flax
import flax.jax_utils as flax_utils

from vdm.conditional_sampling import backward_process_no_corrector, backward_process_pc_single, \
  backward_process_pc_multiple, backward_process_pc_k_gillespies, \
  backward_process_pc_k_gillespies_euler

import fidjax
import pandas as pd
import itertools

from PIL import Image
import os

class AbsorbingRateCosine():
  def __init__(self, config):
    self.state_size = S = config.state_size
    self.scalar_rate = 1
    self.eps = config.rate_eps
    
    mask = S-1

    rate = np.zeros((S, S))
    rate[:-1, -1] = self.scalar_rate
    rate -= np.diag(jnp.sum(rate, axis=1))
    self.eigvals, self.eigvecs = jnp.linalg.eigh(rate)
    self.eigvals, self.eigvecs = np.linalg.eig(rate)
    self.inv_eigvecs = np.linalg.inv(self.eigvecs)
        
    self.base_rate = jnp.array(rate, dtype=np.float32)
#       self.rate_matrix = self.base_rate
    self.eigvals = jnp.array(self.eigvals, dtype=np.float32)
    self.eigvecs = jnp.array(self.eigvecs, dtype=np.float32)
    self.inv_eigvecs = jnp.array(self.inv_eigvecs, dtype=np.float32)
        
  def target_logits(self):
    S = self.state_size
    logits = - jnp.ones((S,)) * 10000
    return logits.at[-1].set(0)
          
  def _integral_rate_scalar(self, t):
    # This is -log of (1-m(t)) where m(t) is the desired mask percentage at time t.
    return - jnp.log(1 - jnp.cos(jnp.pi / 2 * (1 - t * (1 - self.eps))))

  def _rate_scalar(self, t):
    b = 1 - self.eps
    theta = jnp.pi / 2 * (1 - t * b)
    return b * jnp.pi / 2 * jnp.sin(theta) / (1 - jnp.cos(theta))
      
  def rate(self, t):
    return self._rate_scalar(t) * self.base_rate

  def transition(self, t, t0 = 0):
    S = self.state_size
    integral_rate_scalar = self._integral_rate_scalar(t+t0) - self._integral_rate_scalar(t0)
    adj_eigvals = integral_rate_scalar * self.eigvals
    trans = jnp.einsum("ij,jk,kl->il", self.eigvecs, jnp.diag(jnp.exp(adj_eigvals)), self.inv_eigvecs, 
                       precision=jax.lax.Precision.HIGHEST)
    trans = jnp.clip(trans, 0., 1.)
    return trans

# Replicated from experiment.py
def copy_dict(dict1, dict2):
  if not isinstance(dict1, dict):
    assert not isinstance(dict2, dict)
    return dict2
  for key in dict1.keys():
    if key in dict2:
      dict1[key] = copy_dict(dict1[key], dict2[key])

  return dict1


def restore_partial(state, state_restore_dict):
  state_dict = flax.serialization.to_state_dict(state)
  state_dict = copy_dict(state_dict, state_restore_dict)
  state = flax.serialization.from_state_dict(state, state_dict)

  return state

class Experiment_MaskDiff_Conditional(Experiment):
  """Train and evaluate a masked discrete diffusion model."""

  # We override the base initialization
  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

    # Set seed before initializing model.
    seed = config.training.seed
    self.rng = utils.with_verbosity("ERROR", lambda: jax.random.PRNGKey(seed))

    # initialize tokenizer
    logging.warning('=== Initializing tokenizer ===')
    tokenizer_path = "/home/yixiuz/maskgit_checkpoints/tokenizer_imagenet256_checkpoint"
    self.load_imagenet_decoder(tokenizer_path)

    # initialize dataset
    logging.warning('=== Initializing dataset ===')
    self.rng, data_rng = jax.random.split(self.rng)
    self.train_iter, self.eval_iter = dataset.create_dataset(config, data_rng)

    # initialize model
    logging.warning('=== Initializing model ===')
    self.rng, model_rng = jax.random.split(self.rng)
    self.model, params = self.get_model_and_params(model_rng)
    parameter_overview.log_parameter_overview(params)

    # initialize forward process
    # Assuming masking forward process
    self.forward_process = AbsorbingRateCosine(config.noise)

    # initialize train state
    logging.info('=== Initializing train state ===')
    self.state = vdm.train_state.TrainState.create(
        apply_fn=self.model.apply,
        variables=params,
        optax_optimizer=self.get_optimizer)
    self.lr_schedule = self.get_lr_schedule()

    # Restore from checkpoint
    ckpt_restore_dir = self.config.get('ckpt_restore_dir', 'None')
    if ckpt_restore_dir != 'None':
      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.state = restore_partial(self.state, state_restore_dict)
      del state_restore_dict, ckpt_restore, checkpoint_to_restore

    # initialize train/eval step
    logging.info('=== Initializing train/eval step ===')
    self.rng, train_rng = jax.random.split(self.rng)
    self.p_train_step = partial(self.train_step, train_rng)
    self.p_train_step = partial(jax.lax.scan, self.p_train_step)
    self.p_train_step = jax.pmap(self.p_train_step, "batch")

    self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
    self.p_eval_step = partial(self.eval_step, eval_rng)
    self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
    # This assumes that we are iterating over something with a batch axis.
    self.p_sample = partial(
        self.sample_fn,
        dummy_inputs=None
    )
    self.p_sample = utils.dist(
        self.p_sample, accumulate='concat', axis_name='batch')

    logging.info('=== Done with Experiment.__init__ ===')

  def get_model_and_params(self, rng: PRNGKey):
    config = self.config
    if config.use_hollow_transformer:
      logging.info("=== Using the hollow transformer ===")
      model = transformer.HollowTransformer(**config.model)
    else:
      logging.info("=== Using the default transformer ===")
      model = transformer.Transformer(**config.model)

    inputs = jnp.zeros((2, config.data.seq_length), dtype=int)
    params = model.init(rng, inputs, 0)

    logging.info(f'Parameter count: {utils.count_parameters(params)}')
    return model, params

  def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    batch_size = inputs["data"].shape[0]
    loss, metrics = jax.vmap(partial(self.loss_single, params, is_train=is_train))(inputs, jr.split(rng, batch_size))
    metrics = jax.tree_map(lambda x: x.mean(axis=0), metrics)

    loss = jnp.mean(loss)
    return loss, metrics

  def loss_single(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    """
    key: a jax PRNGKey.
    data: (H, W, C) int array, each value should be in [0, S)
    """
    config = self.config
    data, label = inputs["data"], inputs["label"]

    x0 = data.flatten()
    S = config.data.codebook_size + 1
    mask = -1
    D = x0.shape[0]

    forward_process = self.forward_process #config["forward_process"]
    min_t = config.training.min_t # config["min_t"]
    max_t = config.training.max_t # config["max_t"]
    eps = config.training.eps # config["eps"]
    nll_weight = config.training.nll_weight # config["nll_weight"]

    key_t, key_T, key_y = jr.split(rng, 3)

    rngs = {}

    if is_train:
      key_y, key_dropout = jr.split(key_y)
      rngs["dropout"] = key_dropout

    # --------------------------------------------------------------
    # 1. Sample a random t
    # --------------------------------------------------------------

    t = jr.uniform(key_t, minval=min_t, maxval=max_t)

    # q^d_{t|0}(*2|*1): (S, S) float array of finite-time transition probabilities
    # for a single dimension
    qt0 = forward_process.transition(t)
    # R^d_t(*1,*2): (S, S) float array of instantenous transition rates
    # for a single dimension
    Rt = forward_process.rate(t)

    # --------------------------------------------------------------
    # 2. Sample y from q(x_t | x_0)
    # --------------------------------------------------------------

    # q^{*1}_{t|0}(*2|x0): (D, S) float array of probabilities
    qt0_eval_x0 = qt0[x0, :]
    log_qt0_eval_x0 = jnp.log(qt0_eval_x0 + eps)
    # (D,) int array
    y = jr.categorical(key_y, logits=log_qt0_eval_x0)
    # Turn all occurences of S into the mask token (-1)
    # This doesn't affect indexing because python
    y = jnp.where((y == (S-1)), mask, y)

    # --------------------------------------------------------------
    # 3. Evaluate the likelihood ratio predicted by the model
    # --------------------------------------------------------------

    # Shift the label and turn it into an array
    label_arr = jnp.array([label + S], dtype=jnp.int32)
    y_with_label = jnp.concatenate([label_arr, y, label_arr])
    x0_logits = self.state.apply_fn(
        {"params": params}, y_with_label[None], t,  # Assume that our model takes in a batch dimension (t is not used at the moment)
        rngs=rngs,
        deterministic=not is_train,
    )

    # Assuming our model auto-adds a batch dimension
    # Also removing the label that gets added to the first and last position
    # Finally note that only the first S dimensions is used in the output.
    x0_logits = x0_logits[0, 1:-1, :S]
    
    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)

    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    qt0_eval_y = qt0[:,y].T + eps

    # s_t^{\theta}^{*1}(y, *2): (D, S) float array of marginal likelihood ratios predicted by the model
    # Also known as the "concrete score" in (Lou et al. 2023)
    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # -------------------------------------------------------------
    # 4. Evaluate the likelihood ratios at t conditioned on data
    # -------------------------------------------------------------

    # # q_{t|0}^{*1}(y^d, x_0^d): (D,) float array of sample probabilities in the forward process
    # qt0_eval_y_x0 = qt0_eval_x0[jnp.arange(D), y] + eps
    # # The likelihood ratio
    # qt0_x_over_y = qt0_eval_x0 / qt0_eval_y_x0[:, None]

    # -------------------------------------------------------------
    # 5. Tying it all together
    # -------------------------------------------------------------

    # R^{*1}_t(*2,y^d): (D, S) float array of transition rates to y
    # for each dimension
    Rt_eval_y = Rt[:, y].T

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)

    # (D, S) float array, with each entry corresponding to a choice of (d, x^d)
    # the cases where x^d = y^d are removed via masking
    Rt_eval_x = Rt[y]
    # st_eval_y represents "score from y"
    # We only care when y is not mask
    score_to_y = jnp.where((y == mask), 1, st_eval_y[jnp.arange(D), y])
    score_entropy = jnp.sum(Rt_eval_y * y_mask * st_eval_y) \
        - jnp.sum(Rt_eval_x[:, mask] * jnp.log(score_to_y + eps)) # changed mean to sum
    
    # Compute the cross entropy between prediction and data
    x0_one_hot = jax.nn.one_hot(x0, S)
    logits = log_softmax(x0_logits, axis=-1)
    x0_nll = - jnp.sum(x0_one_hot * logits) # changed mean to sum

    loss = score_entropy + nll_weight * x0_nll
    
    # Sample from q_T to estimate the elbo
    # (S,) float array of the logits of the stationary distribution
    # pi_logits = forward_process.target_logits()
    # xT = jr.categorical(key_T, logits=pi_logits, shape=(D,))
    # log_pi_eval_xT = jnp.sum(pi_logits[xT])
    # This elbo is probably not meaningful at all
    # elbo = jnp.sum(- score_entropy + Rt_eval_y * y_mask) + log_pi_eval_xT

    scalar_dict = {
        "loss": loss,
        # "elbo": elbo / D,
        "nll": x0_nll
    }

    img_dict = {}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return loss, metrics

  def sample(self, logdir, checkpoint_dir):
    """Perform one evaluation."""
    logging.info('=== Experiment.sample() ===')

    sample_logdir = os.path.join(logdir, 'samples')
    tf.io.gfile.makedirs(sample_logdir)

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    # Distribute training.
    params = flax_utils.replicate(params)

    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle?dl=1'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)
    
    self._sample_and_compute_fid(fid, params, total_samples=self.config.sampler.max_samples,
      samples_per_label=10, save_imgs=True, sample_logdir=sample_logdir)

  def _sample_and_compute_fid(self, fid, params, total_samples=10_000, 
    samples_per_label=10, save_imgs=False, sample_logdir=None):

    image_id = 0
    rng = jax.random.PRNGKey(self.config.sampler.seed)
    all_acts = []
    all_images = []

    while image_id < total_samples:
      rng, curr_rng = jax.random.split(rng)
      # sample a batch of images
      tokens, samples = self.p_sample(params=params, rng=jax.random.split(curr_rng, 8), 
                                      samples_per_label=jnp.ones((8,)) * samples_per_label,
                                      completed_samples=jnp.ones((8,)) * image_id)      
      samples = np.clip(samples, 0, 1)      
      uint8_images = (samples * 255).astype(np.uint8)

      all_images.append(uint8_images)

      if save_imgs and jax.process_index() == 0 and image_id % (128 * 10) == 0:
        # Save some sample images
        img = utils.generate_image_grids(uint8_images[:100])
        path_to_save = sample_logdir + f'/{image_id}.png'
        img = Image.fromarray(np.array(img))
        img.save(path_to_save)

      image_id += samples.shape[0]
      logging.info(f"Number of samples: {image_id}/{max_samples}")

      all_acts.append(fid.compute_acts(uint8_images))

    if jax.process_index() == 0:
      logging.info("Finished saving samples and activations. Computing FID...")
      stats = fid.compute_stats(all_acts)
      # We have to move these to the cpu since matrix sqrt is not supported by tpus yet
      stats_cpu = jax.device_put(stats, device=jax.devices("cpu")[0])
      ref_cpu = jax.device_put(fid.ref, device=jax.devices("cpu")[0])
      score = fid.compute_score(stats_cpu, ref_cpu)
      logging.info(f"FID score: {score}")

      if save_imgs:
        file_name = self.config.sampler.output_file_name or 'out'
        jnp.save(sample_logdir + f'/{file_name}_score={score:.2f}', jnp.concatenate(all_images, axis=0))

      logging.info(f"======= Complete =======")

      return score
    else:
      return None

  def sample_sweep(self, logdir, checkpoint_dir):

    logging.info('=== Experiment.sample_sweep() ===')

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    # Distribute training.
    params = flax_utils.replicate(params)

    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle?dl=1'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)

    file_name = "results_test.csv"
    csv_file = os.path.join(logdir, file_name)

    if os.path.exists(csv_file):
      df = pd.read_csv(csv_file)
    else:
      df = pd.DataFrame(columns=['method', 'num_cstep', 'entry_time', 
        'cstep_size', 'num_pstep', 'corrector', 'fid'])

    methods = ["euler"]
    num_csteps = [1, 2]
    entry_times = [.9, .5, .3]
    cstep_sizes = [2., 1., 5.] # divide by 100 for mpf stepsizes
    num_psteps = [16, 32, 64, 128]
    correctors = ["forward_backward", "mpf", "barker"]

    no_corrector_experiments = itertools.product(
      methods[:1], num_csteps[:1], entry_times[:1], 
      cstep_sizes[:1], num_psteps, [None])
    params_combination = itertools.product(methods, num_csteps, entry_times, 
      cstep_sizes, num_psteps, correctors)

    params_combination = itertools.chain(no_corrector_experiments, 
      params_combination)

    cfg = self.config.sampler

    for method, num_cstep, entry_time, cstep_size, num_pstep, corrector in params_combination:
      # Adjust mpf stepsize
      if corrector == "mpf":
        cstep_size /= 100

      cfg.num_steps = num_pstep
      cfg.update_type = method
      cfg.corrector = corrector
      cfg.corrector_step_size = cstep_size
      cfg.corrector_entry_time = entry_time
      cfg.num_corrector_steps = num_cstep

      fid_score = self._sample_and_compute_fid(fid, params, 
        total_samples=1000,
        samples_per_label=10, save_imgs=False)
      result = {
        'method': method, 
        'num_cstep': num_cstep, 
        'entry_time': entry_time, 
        'cstep_size': cstep_size, 
        'num_pstep': num_pstep, 
        'corrector': corrector, 
        'fid': fid_score
      }

      df = df.append(result, ignore_index=True)
      df.to_csv(csv_file, index=False)

  def sample_fn(self, *, dummy_inputs, rng, params, samples_per_label=11, completed_samples=0):
    # We don't really need to use the dummy inputs.

    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    label = (completed_samples + jax.lax.axis_index('batch')) // samples_per_label
    label = jnp.clip(label, max=999) # There are only 1000 labels in total

    config = self.config

    if config.sampler.corrector:
      if config.sampler.update_type == "gillespies":
        backward_process = backward_process_pc_k_gillespies
      elif config.sampler.update_type == "gillespies_euler":
        backward_process = backward_process_pc_k_gillespies_euler
      else: # tau-leaping or euler
        if config.sampler.num_corrector_steps == 1:
          backward_process = backward_process_pc_single
        else:
          backward_process = backward_process_pc_multiple
      logging.info(f"Using sampling strategy: {config.sampler.update_type}")
      logging.info(f"Using corrector: {config.sampler.corrector}")
    else:
      backward_process = backward_process_no_corrector

    S = config.data.codebook_size + 1
    D = config.data.seq_length
    min_t = config.training.min_t
    max_t = config.training.max_t
    num_steps = config.sampler.num_steps
    
    # Initialize the all-mask state
    xT = jnp.ones((D,), dtype=int) * (S - 1)
    label_arr = jnp.array([label + S], dtype=jnp.int32)
    xT_with_label = jnp.concatenate([label_arr, xT, label_arr])
    
    ts = jnp.linspace(max_t, min_t, num_steps)
    tokens, _ = backward_process(self.state.apply_fn, params, ts, config, xT_with_label, rng, 
      self.forward_process)

    output_tokens = jnp.reshape(tokens, [-1, 16, 16])
    gen_images = self.tokenizer_model.apply(
              self.tokenizer_variables,
              output_tokens,
              method=self.tokenizer_model.decode_from_indices,
              mutable=False)

    return output_tokens, gen_images

  def load_imagenet_decoder(self, checkpoint_path):
    # Assume that we've already downloaded the pretrained vqvae
    self.tokenizer_model = vqgan_tokenizer.VQVAE(config=self.config, dtype=jnp.float32, train=False)
    with tf.io.gfile.GFile(checkpoint_path, "rb") as f:
      self.tokenizer_variables = flax.serialization.from_bytes(None, f.read())
    

      