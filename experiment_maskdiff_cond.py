import numpy as np
import scipy

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
  backward_process_pc_k_gillespies_euler, test_corrector_convergence, \
  backward_process_gibbs, backward_process_maskgit, maskgit, \
  backward_process_remdm

import fidjax
import pandas as pd
import itertools

from PIL import Image
import os

from matplotlib import pyplot as plt

from jax.experimental import checkify

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

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
          
  def mask_percentage(self, t):
    return 1 - self.alpha(t)

  def alpha(self, t):
    """
    Survival probability of a token at time t
    """
    b = 1 - self.eps
    theta = jnp.pi / 2 * (1 - t * b)
    return 1 - jnp.cos(theta)

  def dalpha(self, t):
    """
    Time derivative of alpha_t (the survival probability of a token at time t)
    """
    b = 1 - self.eps
    theta = jnp.pi / 2 * (1 - t * b)
    return - b * jnp.pi / 2 * jnp.sin(theta)

  def _integral_rate_scalar(self, t):
    # This is -log of (1-m(t)) where m(t) is the desired mask percentage at time t.
    return - jnp.log(self.alpha(t))

  def _rate_scalar(self, t):
    return - self.dalpha(t) / self.alpha(t)
      
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
    # self.p_train_step = checkify.checkify(self.p_train_step, errors=checkify.float_checks)
    self.p_train_step = jax.pmap(self.p_train_step, "batch")

    self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
    self.p_eval_step = partial(self.eval_step, eval_rng)
    self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
    # This assumes that we are iterating over something with a batch axis.
    self.p_sample = partial(self.sample_fn, dummy_inputs=None)
    self.p_sample = utils.dist(self.p_sample, accumulate='concat', axis_name='batch')

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

    # Antithetic sampling evenly spaces out t across the batch
    # Reference: https://github.com/google-deepmind/md4/blob/main/md4/models/diffusion/md4.py#L243
    if self.config.training.antithetic_time_sampling:
      rng, rng_t = jr.split(rng)
      t0 = jr.uniform(rng_t)
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / batch_size), 1.0)
      loss, metrics = jax.vmap(partial(self.loss_single, params, is_train=is_train))(inputs, jr.split(rng, batch_size), t)
    else:
      loss, metrics = jax.vmap(partial(self.loss_single, params, is_train=is_train, t=None))(inputs, jr.split(rng, batch_size))

    metrics = jax.tree_map(lambda x: x.mean(axis=0), metrics)
    loss = jnp.mean(loss)
    return loss, metrics

  def loss_single(self, params, inputs, rng, t, is_train) -> Tuple[float, Any]:
    """
    Computes the diffusion loss which is a combination of the MD4 loss (uses the masked dimensions) 
    and our loss (uses the non-masked dimensions).

    rng: a jax PRNGKey.
    data: (H, W, C) int array, each value should be in [0, S)
    """
    config = self.config
    data, label = inputs["data"], inputs["label"]

    x0 = data.flatten()
    S = config.data.codebook_size + 1
    mask = S-1
    D = x0.shape[0]

    forward_process = self.forward_process 
    min_t = config.training.min_t 
    max_t = config.training.max_t 
    eps = config.training.eps

    key_t, key_T, key_y, key_xs_tilde, key_p = jr.split(rng, 5)

    rngs = {"permute": key_p}

    if is_train:
      key_y, key_dropout = jr.split(key_y)
      rngs["dropout"] = key_dropout

    if t is None:
      t = jr.uniform(key_t, minval=min_t, maxval=max_t)

    # q^d_{t|0}(*2|*1): (S, S) float array of finite-time transition probabilities
    # for a single dimension
    qt0 = forward_process.transition(t)

    # q^{*1}_{t|0}(*2|x0): (D, S) float array of probabilities
    qt0_eval_x0 = qt0[x0, :]
    # (D,) int array
    y = jr.categorical(key_y, logits=jnp.log(qt0_eval_x0))
    # Turn all occurences of S-1 into the mask token (-1)
    # This doesn't affect indexing because python
    y = jnp.where((y == (S-1)), mask, y)

    # Shift the label and turn it into an array
    label_arr = jnp.array([label + S], dtype=jnp.int32)
    y_with_label = jnp.concatenate([label_arr, y, label_arr])

    # Feed it to the model to get the denoising logits
    x0_logits = self.state.apply_fn(
        {"params": params}, y_with_label[None], t,  # Assume that our model takes in a batch dimension (t is not used at the moment)
        rngs=rngs, deterministic=not is_train)

    # Assuming our model auto-adds a batch dimension
    # Also removing the label that gets added to the first and last position
    # Only the first S dimensions is used in the output.
    x0_logits = x0_logits[0, 1:-1, :S]
    # Set the mask prob to 0
    x0_logits = x0_logits.at[:, mask].set(-jnp.inf)

    # Normalize x0 logits
    log_p0t_eval_y = x0_logits - jax.scipy.special.logsumexp(x0_logits, axis=-1, 
        keepdims=True)

    # Compute the weights for each dimension
    # Note that when t = 1, alpha is close to but not equal to 0
    # when t = 0, 1 - alpha = 0, but min_t > 0 so we never encounter this case
    alpha = forward_process.alpha(t)
    dalpha = forward_process.dalpha(t)

    mask_dim_weight = dalpha / (1 - alpha)
    non_mask_dim_weight = dalpha / alpha

    mask_loss = jnp.sum(jnp.where((y == mask), mask_dim_weight, 0) * log_p0t_eval_y[jnp.arange(D), x0])
    non_mask_loss = jnp.sum(jnp.where((y == mask), 0, non_mask_dim_weight) * log_p0t_eval_y[jnp.arange(D), x0])

    mixed_loss = 0.5 * (mask_loss + non_mask_loss)

    if config.loss == "mixed":
      loss = mixed_loss
    elif config.loss == "mask":
      loss = mask_loss
    elif config.loss == "non_mask":
      loss = non_mask_loss
    elif config.loss == "sic":
      # Implement the additional terms for SIC
      wt = 0.5 if config.training.corrector_loss_weight == "constant" else t

      # Sample xs_tilde by running 1 ancestral step
      s = t - 1.0 / config.training.sic_steps
      s = jnp.clip(s, 0.0, 1.0) # Edge case s < 0.0
      alpha_t = forward_process.alpha(t)
      alpha_s = forward_process.alpha(s)

      denoising_probs = softmax(x0_logits, axis=-1)
      unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
      probs = unmask_prob * denoising_probs
      probs = probs.at[:,mask].set(1 - unmask_prob)

      to_unmask = tfd.Categorical(probs=probs).sample(seed=key_xs_tilde)
      xs_tilde = jnp.where(y == mask, to_unmask, y)
      # Don't take gradient w.r.t. xs_tilde
      xs_tilde = jax.lax.stop_gradient(xs_tilde)

      # Pass the sampled xs_tilde through the model
      xs_tilde_with_label = jnp.concatenate([label_arr, xs_tilde, label_arr])
      logits_xs_tilde = self.state.apply_fn(
          {"params": params}, xs_tilde_with_label[None], s, rngs=rngs, deterministic=not is_train)

      logits_xs_tilde = logits_xs_tilde[0, 1:-1, :S]
      logits_xs_tilde = logits_xs_tilde.at[:, mask].set(-jnp.inf)
      log_p_xs_tilde = logits_xs_tilde - jax.scipy.special.logsumexp(logits_xs_tilde, axis=-1, keepdims=True)

      # Compute the cross entropy on the updated dimensions
      neg_cross_ent = log_p_xs_tilde[jnp.arange(D), x0]
      is_mask_xs = (xs_tilde == mask).astype('float32')
      is_mask_y = (y == mask).astype('float32')
      corrector_neg_cross_ent = jnp.sum((1 - is_mask_xs) * is_mask_y * neg_cross_ent, axis=-1)

      corrector_loss = -config.training.sic_steps * corrector_neg_cross_ent
      loss = (wt * mask_loss + (1 - wt) * corrector_loss)
    else:
      raise ValueError(f"Unrecognized loss type: {config.loss}")

    if config.loss == "sic":
      scalar_dict = {
          "loss": loss,
          "mask_loss": mask_loss,
          "corrector_loss": corrector_loss,
      }
    else:
      scalar_dict = {
          "loss": loss,
          "mask_loss": mask_loss,
          "non_mask_loss": non_mask_loss,
          "mixed_loss": mixed_loss,
      }

    img_dict = {}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return loss, metrics

  def sample(self, logdir, checkpoint_dir):
    """Perform one evaluation."""
    logging.info('=== Experiment.sample() ===')

    sample_logdir = os.path.join(logdir, 'samples')
    tf.io.gfile.makedirs(sample_logdir)

    if self.config.sampler.use_corrector_model:
      # Load and initialize the corrector model
      # Hard coding the config file for now
      from vdm.configs.maskdiff_cond_hollow_train import get_config
      corr_config = get_config()
      logging.info("=== Using a separately training corrector model ===")
      corr_model = transformer.HollowTransformer(**corr_config.model)

      inputs = jnp.zeros((2, corr_config.data.seq_length), dtype=int)
      # Random key doesn't matter here
      corr_params = corr_model.init(jr.PRNGKey(0), inputs, 0)

      self.corrector_state = vdm.train_state.TrainState.create(
        apply_fn=corr_model.apply,
        variables=corr_params,
        optax_optimizer=self.get_optimizer)

      # Restore checkpoint
      ckpt_restore_dir = corr_config.get('ckpt_restore_dir', 'None')
      assert ckpt_restore_dir != 'None'

      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.corrector_state = restore_partial(self.corrector_state, state_restore_dict)

      c_params = self.corrector_state.params
      c_params = flax_utils.replicate(c_params)
    else:
      c_params = None

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    # Distribute training.
    params = flax_utils.replicate(params)

    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)
    
    num_labels = 1000

    self._sample_and_compute_fid(fid, params, 
      total_samples=self.config.sampler.max_samples,
      samples_per_label=self.config.sampler.max_samples // num_labels, 
      save_imgs=self.config.sampler.save_imgs_and_acts, 
      sample_logdir=sample_logdir, 
      verbose=True,
      c_params=c_params)

  def _sample_and_compute_fid(self, fid, params, total_samples=50_000, 
    samples_per_label=50, save_imgs=False, sample_logdir=None, verbose=True,
    debug=False, c_params=None):

    config = self.config 
    S = config.data.codebook_size + 1
    D = config.data.seq_length

    min_t = config.training.min_t
    max_t = config.training.max_t
    num_steps = config.sampler.num_steps
    
    ts = jnp.linspace(max_t, min_t, num_steps)
    ts = ts[1:]
    xs = jnp.arange(num_steps - 1)
    expected_tokens = D * (1 - self.forward_process.mask_percentage(ts))
    plt.plot(xs, expected_tokens, color="orange")

    image_id = 0
    rng = jax.random.PRNGKey(self.config.sampler.seed)
    all_acts = []
    all_images = []
    mask_curves = []
    batch_images = []

    while image_id < total_samples:
      rng, curr_rng = jax.random.split(rng)
      # sample a batch of images
      tokens_hist, samples = self.p_sample(params=params, rng=jax.random.split(curr_rng, 8), 
                                      samples_per_label=jnp.ones((8,)) * samples_per_label,
                                      completed_samples=jnp.ones((8,)) * image_id, c_params=c_params)
      mask_curve = jnp.sum(tokens_hist != (S-1), axis=-1)
      mask_curve = jnp.reshape(mask_curve, (128, -1))
      mask_curves.append(mask_curve)
      
      samples = np.clip(samples, 0, 1)      
      uint8_images = (samples * 255).astype(np.uint8)

      # all_images.append(uint8_images)
      batch_images.append(uint8_images)

      if (save_imgs and jax.process_index() == 0 
        and image_id % (128 * 10) == 0 
        and not debug):
        # # Save some sample images
        # img = utils.generate_image_grids(uint8_images[:100])
        # path_to_save = sample_logdir + f'/{image_id}.png'
        # img = Image.fromarray(np.array(img))
        # img.save(path_to_save)

        # This somehow doesn't break the TPU
        file_name = utils.get_file_name(self.config)
        if not os.path.exists(sample_logdir + f'/{file_name}_images'):
          os.makedirs(sample_logdir+f'/{file_name}_images')
        np.save(sample_logdir + f'/{file_name}_images/{image_id}', 
          np.concatenate(batch_images), allow_pickle=True)
        batch_images = []

      image_id += samples.shape[0]
      all_acts.append(fid.compute_acts(uint8_images))

      if verbose:
        logging.info(f"Number of samples: {image_id}/{total_samples}")

    # if jax.process_index() == 0 and not debug:
    if not debug:

      file_name = utils.get_file_name(self.config)

      logging.info("Finished saving samples and activations. Computing FID...")
      stats = fid.compute_stats(all_acts)
      # We have to move these to the cpu since matrix sqrt is not supported by tpus yet
      stats_cpu = jax.device_put(stats, device=jax.devices("cpu")[0])
      ref_cpu = jax.device_put(fid.ref, device=jax.devices("cpu")[0])
    
      # score = fid.compute_score(stats_cpu, ref_cpu)
      mu1, sigma1 = stats_cpu
      mu2, sigma2 = ref_cpu
      diff = mu1 - mu2
      offset = jnp.eye(sigma1.shape[0]) * 1e-6
      covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
      covmean = np.real(covmean)
      score = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

      logging.info(f"FID score: {score}")

      if save_imgs:
        jnp.save(sample_logdir + f'/{file_name}_acts', jnp.concatenate(all_acts, axis=0))

        # Instead of saving all images at once, we save them in chunks
        # Create a folder first
        if not os.path.exists(sample_logdir + f'/{file_name}_images'):
          os.makedirs(sample_logdir+f'/{file_name}_images')

        # # Save the images in larger batches
        # save_batch_size = 10
        # for i in range(0, len(all_images), save_batch_size):
        #   # np.savez_compressed(sample_logdir + f'/{file_name}_images/{i}', 
        #   #   jnp.concatenate(all_images[i:i+save_batch_size], axis=0))
        #   np.save(sample_logdir + f'/{file_name}_images/{i}', 
        #     jnp.concatenate(all_images[i:i+save_batch_size], axis=0))

        # for i, images in enumerate(all_images):
        #   np.savez_compressed(sample_logdir + f'/{file_name}_images/{i}', images)
        # all_images = np.concatenate(all_images)
        # logging.info(f"Shape of all_images: {all_images.shape}")
        # jnp.save(sample_logdir + f'/{file_name}_score={score:.2f}', all_images)
        # Use savez_compressed to save disk space
        # np.savez_compressed(sample_logdir + f'/{file_name}_images', all_images)
        logging.info("Saved activations and images")

      logging.info(f"======= Complete =======")

      return score
    else:
      return None

  def sample_sweep(self, logdir, checkpoint_dir):

    logging.info('=== Experiment.sample_sweep() ===')

    if self.config.sampler.use_corrector_model:
      # Load and initialize the corrector model
      # Hard coding the config file for now
      from vdm.configs.maskdiff_cond_hollow_train import get_config
      corr_config = get_config()
      logging.info("=== Using a separately training corrector model ===")
      corr_model = transformer.HollowTransformer(**corr_config.model)

      inputs = jnp.zeros((2, corr_config.data.seq_length), dtype=int)
      # Random key doesn't matter here
      corr_params = corr_model.init(jr.PRNGKey(0), inputs, 0)

      self.corrector_state = vdm.train_state.TrainState.create(
        apply_fn=corr_model.apply,
        variables=corr_params,
        optax_optimizer=self.get_optimizer)

      # Restore checkpoint
      ckpt_restore_dir = corr_config.get('ckpt_restore_dir', 'None')
      assert ckpt_restore_dir != 'None'

      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.corrector_state = restore_partial(self.corrector_state, state_restore_dict)

      c_params = self.corrector_state.params
      c_params = flax_utils.replicate(c_params)
    else:
      c_params = None

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    # Distribute training.
    params = flax_utils.replicate(params)

    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)

    # file_name = "results_k_sweep_05_15_50k.csv"
    # file_name = "results_separate_corrector_05_14_50k.csv"
    file_name = "results_additional_05_14_50k.csv"
    csv_file = os.path.join(logdir, file_name)

    # if jax.process_index() == 0:
    if True:
      if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
      else:
        df = pd.DataFrame(columns=['method', 'num_cstep', 'entry_time', 
          'cstep_size', 'num_pstep', 'corrector', 'fid', 
          'k', 'gibbs_temperature', 'maskgit_temperature'])

    # # Sweeping using existing best configurations
    # # Load the best experimental configurations
    # best_configs_file_path = "/home/yixiuz/vdm/best_results_gibbs_fixed_2_17.csv"
    # best_configs = pd.read_csv(best_configs_file_path)

    # best_configs.fillna("none", inplace=True)

    # # Convert the dataframe into a list of tuples matching the required parameter order
    # params_combination = list(
    #     best_configs[
    #         ["method", "corrector", "num_cstep", "entry_time", "cstep_size", "num_pstep", "k", "gibbs_temperature", "maskgit_temperature"]
    #     ].itertuples(index=False, name=None)
    # )

    # no_corrector_experiments = itertools.product(
    #   ["gibbs"], # Uses md4 sampling
    #   ["gibbs"], # So that we would call backward_process_gibbs
    #   [0], # no corrector
    #   [-1], 
    #   [-1], 
    #   [8, 16, 32, 64, 128, 256], 
    #   [-1], [-1], [-1])

    # Gibbs
    methods = ["gibbs"]
    correctors = ["gibbs"]
    num_csteps = [1,]
    entry_times = [1.]
    cstep_sizes = [-1]
    num_psteps = [8,] 
    ks = [2, 1]
    top_k_temperatures = [1.]
    maskgit_temperatures = [-1]

    psteps_8_gibbs_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)
  
    methods = ["gibbs"]
    correctors = ["gibbs"]
    num_csteps = [1,]
    entry_times = [1.]
    cstep_sizes = [-1]
    num_psteps = [16,]
    ks = [64, 2, 1]
    top_k_temperatures = [2.]
    maskgit_temperatures = [-1]

    psteps_16_gibbs_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)

    # # Maskgit experiments
    # methods = ["maskgit"]
    # correctors = [None]
    # num_csteps = [0,]
    # entry_times = [-1]
    # cstep_sizes = [-1]
    # num_psteps = [16, 32, 64, 128, 256]
    # ks = [-1]
    # top_k_temperatures = [-1]
    # maskgit_temperatures = [.5, 1., 2., 4., 8., 10., 12.]

    # maskgit_experiments = itertools.product(
    #   methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
    #   ks, top_k_temperatures, maskgit_temperatures)

    # Forward-backward experiments
    methods = ["euler"]
    correctors = ["forward_backward"]
    num_csteps = [1,]
    entry_times = [.9]
    cstep_sizes = [.5, 1., 2., 4.] 
    num_psteps = [8, 16, 32, 64, 128] # Save 64 and 128 for later
    ks = [-1]
    top_k_temperatures = [-1]
    maskgit_temperatures = [-1]

    forward_backward_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)

    # Forward-backward experiments
    methods = ["remdm"]
    correctors = ["forward_backward"]
    num_csteps = [1,]
    entry_times = [.9]
    cstep_sizes = [.01, .02, .05] # This is actually the sigma scale 
    num_psteps = [16] # Save 64 and 128 for later
    ks = [-1]
    top_k_temperatures = [-1]
    maskgit_temperatures = [-1]

    remdm_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)

    # params_combination = itertools.chain(
    #   no_corrector_experiments,
    #   maskgit_experiments, 
    #   forward_backward_experiments,
    #   gibbs_experiments)

    # The other half of the gibbs experiments
    methods = ["gibbs"]
    correctors = ["gibbs"]
    num_csteps = [1,]
    entry_times = [1.,]
    cstep_sizes = [-1] 
    num_psteps = [32]
    ks = [32, 64]
    top_k_temperatures = [5.]
    maskgit_temperatures = [-1]

    psteps_32_gibbs_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)

    methods = ["gibbs"]
    correctors = ["gibbs"]
    num_csteps = [1,]
    entry_times = [1.,]
    cstep_sizes = [-1] 
    num_psteps = [64]
    ks = [32, 64]
    top_k_temperatures = [10.]
    maskgit_temperatures = [-1]

    psteps_64_gibbs_experiments = itertools.product(
      methods, correctors, num_csteps, entry_times, cstep_sizes, num_psteps, 
      ks, top_k_temperatures, maskgit_temperatures)

    # params_combination = maskgit_experiments

    params_combination = itertools.chain(
      psteps_8_gibbs_experiments,
      # psteps_16_gibbs_experiments,
      # psteps_32_gibbs_experiments,
      # psteps_64_gibbs_experiments
      # remdm_experiments,
      # no_corrector_experiments,
      # maskgit_experiments,
      # forward_backward_experiments
      )

    cfg = self.config.sampler

    for method, corrector, num_cstep, entry_time, cstep_size, num_pstep, k, top_k_temperature, maskgit_temperature in params_combination:

      # Skip if the experiment has already been done
      if df[(df.method == method) & (df.num_cstep == num_cstep) & (df.entry_time == entry_time) &
        (df.cstep_size == cstep_size) & (df.num_pstep == num_pstep) & (df.corrector == corrector) &
        (df.k == k) & (df.gibbs_temperature == top_k_temperature) & (df.maskgit_temperature == maskgit_temperature)].shape[0] > 0:
        logging.info(f"Skipping experiment with method={method}, num_cstep={num_cstep}, entry_time={entry_time}, cstep_size={cstep_size}, num_pstep={num_pstep}, corrector={corrector}, k={k}, gibbs_temperature={top_k_temperature}, maskgit_temperature={maskgit_temperature}")
        continue

      cfg.update_type = method
      cfg.corrector = corrector
      cfg.num_corrector_steps = num_cstep
      cfg.corrector_entry_time = entry_time
      cfg.corrector_step_size = cstep_size
      cfg.num_steps = num_pstep
      cfg.k = k
      cfg.top_k_temperature = top_k_temperature
      cfg.maskgit_temperature = maskgit_temperature

      # TODO: this is a hack to include the new remdm method
      if method == "remdm":
        cfg.sigma_scale = cstep_size

      save_imgs = False #num_pstep <= 16

      # Redefine the sample function now that we have changed configs
      self.p_sample = partial(self.sample_fn, dummy_inputs=None)
      self.p_sample = utils.dist(self.p_sample, accumulate='concat', axis_name='batch')

      if True:
      # try:
        fid_score = self._sample_and_compute_fid(fid, params, 
          # Search with lower number of samples first
          total_samples=50_000,
          samples_per_label=50, 
          save_imgs=save_imgs,
          sample_logdir="/home/yixiuz/logs/samples",
          c_params=c_params)
      # except:
      #   logging.info('====== Experiment failed due to an unknown reason, moving on... ======')
      #   fid_score = None

      result = {
        "method": [method],
        "num_cstep": [num_cstep],
        "entry_time": [entry_time],
        "cstep_size": [cstep_size],
        "num_pstep": [num_pstep],
        "corrector": [corrector],
        "fid": [fid_score],
        "k": [k],
        "gibbs_temperature": [top_k_temperature],
        "maskgit_temperature": [maskgit_temperature]
      }

      if True:
      # if jax.process_index() == 0:
        df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
        df.to_csv(csv_file, index=False)

  def sample_fn(self, *, dummy_inputs, rng, params, 
    samples_per_label=11, completed_samples=0, c_params=None):
    # We don't really need to use the dummy inputs.
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    label = (completed_samples + jax.lax.axis_index('batch')) // samples_per_label
    label = jnp.clip(label, max=999) # There are only 1000 labels in total

    config = self.config

    # TODO: we need to fix this mess
    if config.sampler.corrector and config.sampler.corrector != "none":
      if config.sampler.update_type == "maskgit":
        backward_process = backward_process_maskgit
      elif config.sampler.update_type == "gillespies":
        backward_process = backward_process_pc_k_gillespies
      elif config.sampler.update_type == "gillespies_euler":
        backward_process = backward_process_pc_k_gillespies_euler
      elif config.sampler.update_type == "test_convergence":
        backward_process = test_corrector_convergence
      elif config.sampler.update_type == "gibbs":
        if self.config.sampler.use_corrector_model:
          backward_process = partial(backward_process_gibbs, c_apply_fn=self.corrector_state.apply_fn, c_params=c_params)
        else:
          backward_process = backward_process_gibbs
      elif config.sampler.update_type == "remdm":
        backward_process = backward_process_remdm
      else: # tau-leaping or euler or md4
        backward_process = backward_process_pc_multiple
      logging.info(f"Using sampling strategy: {config.sampler.update_type}")
      logging.info(f"Using corrector: {config.sampler.corrector}")
    else:
      if config.sampler.update_type == "maskgit":
        backward_process = maskgit
      else:
        backward_process = backward_process_no_corrector

    S = config.data.codebook_size + 1
    D = config.data.seq_length
    # Drop min_t to 0
    min_t = 0 
    max_t = 1
    num_steps = config.sampler.num_steps
    
    # Initialize the all-mask state
    xT = jnp.ones((D,), dtype=int) * (S - 1)
    label_arr = jnp.array([label + S], dtype=jnp.int32)
    xT_with_label = jnp.concatenate([label_arr, xT, label_arr])
    
    # We want the length of the sequence to be num_steps + 1
    # Since we stop immediately after hitting min_t
    ts = jnp.linspace(max_t, min_t, num_steps + 1)
    tokens, hist = backward_process(self.state.apply_fn, params, ts, config, xT_with_label, rng, self.forward_process)

    output_tokens = jnp.reshape(tokens, [-1, 16, 16])
    gen_images = self.tokenizer_model.apply(
              self.tokenizer_variables,
              output_tokens,
              method=self.tokenizer_model.decode_from_indices,
              mutable=False)

    return hist[...,1:-1], gen_images

  def load_imagenet_decoder(self, checkpoint_path):
    # Assume that we've already downloaded the pretrained vqvae
    self.tokenizer_model = vqgan_tokenizer.VQVAE(config=self.config, dtype=jnp.float32, train=False)
    with tf.io.gfile.GFile(checkpoint_path, "rb") as f:
      self.tokenizer_variables = flax.serialization.from_bytes(None, f.read())
    

      