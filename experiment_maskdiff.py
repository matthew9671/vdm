import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple

from vdm.experiment import Experiment
import vdm.transformer as transformer


class AbsorbingRate():
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
    return -jnp.log1p(-(1 - self.eps) * t)
  
  def _rate_scalar(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)
      
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

class Experiment_MaskDiff(Experiment):
  """Train and evaluate a masked discrete diffusion model."""

  # We override the base initialization
  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

    # Set seed before initializing model.
    seed = config.training.seed
    self.rng = utils.with_verbosity("ERROR", lambda: jax.random.PRNGKey(seed))

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
    self.forward_process = AbsorbingRate(config.noise)

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
    self.p_train_step = functools.partial(self.train_step, train_rng)
    self.p_train_step = functools.partial(jax.lax.scan, self.p_train_step)
    self.p_train_step = jax.pmap(self.p_train_step, "batch")

    self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
    self.p_eval_step = functools.partial(self.eval_step, eval_rng)
    self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
    self.p_sample = functools.partial(
        self.sample_fn,
        # dummy_inputs=next(self.eval_iter)["images"][0],
        dummy_inputs=None,
        rng=sample_rng,
    )
    self.p_sample = utils.dist(
        self.p_sample, accumulate='concat', axis_name='batch')

    logging.info('=== Done with Experiment.__init__ ===')


  def get_model_and_params(self, rng: PRNGKey):
    config = self.config
    model = transformer.Transformer(**config.model)

    # Do we need a batch dimension here?
    inputs = jnp.zeros((1, config.data.seq_length), dtype=int)
    params = model.init(rng, inputs, 0)

    # inputs = {"images": jnp.zeros((2, 32, 32, 3), "uint8")}
    # inputs["conditioning"] = jnp.zeros((2,))
    # rng1, rng2 = jax.random.split(rng)
    # params = model.init({"params": rng1, "sample": rng2}, **inputs)
    return model, params

  def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
    )

    rescale_to_bpd = 1./(np.prod(inputs["images"].shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": inputs["images"]}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return bpd, metrics

  def loss_fn(self, params, inputs, key, is_train) -> Tuple[float, Any]:
    """
    key: a jax PRNGKey.
    data: (H, W, C) int array, each value should be in [0, S)
    """
    config = self.config
    data = inputs

    x0 = data.flatten()
    S = config.model.vocab_size # config["state_size"]
    D = x0.shape[0]

    forward_process = self.forward_process #config["forward_process"]
    min_t = config.training.min_t # config["min_t"]
    max_t = config.training.max_t # config["max_t"]
    eps = config.training.eps # config["eps"]
    nll_weight = config.training.nll_weight # config["nll_weight"]

    key_t, key_T, key_y = jr.split(key, 3)

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

    # --------------------------------------------------------------
    # 3. Evaluate the likelihood ratio predicted by the model
    # --------------------------------------------------------------

    # x0_logits = model.apply(params, y, t, rngs={"dropout": key_dropout})
    x0_logits = self.state.apply_fn(
        params, y, t,
        rngs=rngs,
        deterministic=not is_train,
    )

    # Assuming our model auto-adds a batch dimension, we want to remove it here:
    x0_logits = x0_logits[0]
    
    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)
    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    qt0_eval_y = qt0[:,y].T + eps

    # s_t^{\theta}^{*1}(y, *2): (D, S) float array of marginal likelihood ratios predicted by the model
    # Also known as the "concrete score" in (Lou et al. 2023)
#     st_eval_y = (p0t_eval_y / qt0_eval_y) @ qt0 
    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # -------------------------------------------------------------
    # 4. Evaluate the likelihood ratios at t conditioned on data
    # -------------------------------------------------------------

    # q_{t|0}^{*1}(y^d, x_0^d): (D,) float array of sample probabilities in the forward process
    qt0_eval_y_x0 = qt0_eval_x0[jnp.arange(D), y] + eps
    # The likelihood ratio
    qt0_x_over_y = qt0_eval_x0 / qt0_eval_y_x0[:, None]

    # -------------------------------------------------------------
    # 5. Tying it all together
    # -------------------------------------------------------------

    # R^{*1}_t(*2,y^d): (D, S) float array of transition rates to y
    # for each dimension
    # Rt_eval_y = Rt.at[:,y].get().T
    Rt_eval_y = Rt[:,y].T

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)

    # (D, S) float array, with each entry corresponding to a choice of (d, x^d)
    # the cases where x^d = y^d are removed via masking
    score_entropy = Rt_eval_y * y_mask * (st_eval_y - qt0_x_over_y * jnp.log(st_eval_y + eps))
    
    # Compute the cross entropy between prediction and data
    x0_one_hot = jax.nn.one_hot(x0, S)
    # TODO: these are log probabilities, and they can probably be computed better from the actual logits
    logits = jnp.log(p0t_eval_y + eps)
    x0_nll = - jnp.mean(x0_one_hot * logits)

    loss = jnp.mean(score_entropy) + nll_weight * x0_nll
    
    # Sample from q_T to estimate the elbo
    # (S,) float array of the logits of the stationary distribution
    pi_logits = forward_process.target_logits()
    xT = jr.categorical(key_T, logits=pi_logits, shape=(D,))
    log_pi_eval_xT = jnp.sum(pi_logits[xT])
    elbo = jnp.sum(- score_entropy + Rt_eval_y * y_mask) + log_pi_eval_xT

    loss_dict = {
        "loss": loss,
        "elbo": elbo / D,
        # "noisy_sample": y,
        # "score_entropy_array": score_entropy,
        "nll": x0_nll
    }

    return loss, loss_dict

  def sample_fn(self, *, dummy_inputs, rng, params):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    if self.model.config.sm_n_timesteps > 0:
      T = self.model.config.sm_n_timesteps
    else:
      T = 1000

    conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')

    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    z_init = jax.random.normal(sample_rng, dummy_inputs.shape)

    def body_fn(i, z_t):
      return self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T,
          z_t=z_t,
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )

    z_0 = jax.lax.fori_loop(
        lower=0, upper=T, body_fun=body_fn, init_val=z_init)

    samples = self.state.apply_fn(
        variables={'params': params},
        z_0=z_0,
        method=self.model.generate_x,
    )

    return samples