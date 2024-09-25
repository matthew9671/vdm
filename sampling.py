import jax
from jax import numpy as jnp
from jax.nn import softmax
import jax.random as jr
from tqdm import trange
from functools import partial

# TODO: we're using the non-batched version of the sampling code to test what the model has learned

def poisson_jump_reject(key, x, rates):
    D = x.shape[0]
    # S = rates.shape[1]
    # Mask out the self transitions
    rates = rates.at[jnp.arange(D), x].set(0.0)
    # (D, S)
    jump_nums = jr.poisson(key, rates)
    jump_target = jnp.argmax(jump_nums, axis=1)

    # Assuming that the mask is S-1
    out = jnp.where((jnp.sum(jump_nums, axis=1) == 1), jump_target, x)
    return out

def compute_backward(y, t, apply_fn, params, config, forward_process):
    y = y.flatten()
    D = y.shape[0]
    S = config.model.vocab_size
    # forward_process = config.forward_process
    min_t = config.training.min_t
    eps = config.training.eps
    qt0 = forward_process.transition(t)
    # R^d_t(*1,*2): (S, S) float array of instantenous transition rates
    # for a single dimension
    Rt = forward_process.rate(t)
    Rt_eval_y = Rt[:, y].T
    
    # TODO: we used squeeze to get rid of the batch dimension
    x0_logits = apply_fn({"params": params}, y[None], t, 
        deterministic=True).squeeze()
    # # Only take the valid parts of the output
    # x0_logits = x0_logits[...,:S-1]
    
    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)
    # Manually append a 0 column, since it corresponds to the mask token
    # p0t_eval_y = jnp.concatenate((p0t_eval_y, jnp.zeros((D, 1))), axis=1)
    
    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    qt0_eval_y = qt0[:,y].T + eps
    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # Change the score such that the score for the non-masked dimensions are inverted
    # This only works for absorbing (masking diffusion ofcourse)
    mask_token = S - 1
    backward_score_to_curr = st_eval_y[jnp.arange(D), y] + eps
    forward_score_from_curr = jnp.concatenate([jnp.zeros((D, S-1)), 1 / backward_score_to_curr[:, None]], axis=1)
    score = jnp.where((y != mask_token)[:,None], forward_score_from_curr, st_eval_y)

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)
    
    results = {
        "score": score,
        "rates": (st_eval_y * Rt_eval_y) * y_mask,
        "x0_logits": x0_logits,
        "Rt_eval_y": Rt_eval_y,
        "Rt_eval_x": Rt[y]
    }
    return results
    
def backward_process_tau_leaping(apply_fn, params, ts, config, xT, key, forward_process):
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size

    t = ts[0]
    x = xT
    
    poisson_jump = poisson_jump_reject

    def _step(carry, idx):
        x, key = carry
        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        backward_rates = res["rates"]
        x = poisson_jump(key, x, backward_rates * dt)
        key = jr.split(key)[0]
        return (x, key), x

    (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
    res = compute_backward(x, t, apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist

def mpf_corrector(res):
    coeff = res["Rt_eval_x"] + res["Rt_eval_y"]
    score = res["score"]
    return coeff * jnp.sqrt(score)

def barker_corrector(res):
    coeff = res["Rt_eval_x"] + res["Rt_eval_y"]
    score = res["score"]
    return coeff * score / (1 + score)

def forward_backward_corrector(res):
    return res["rates"] + res["Rt_eval_x"]

# def backward_process_pc_tau_leaping(apply_fn, params, ts, config, xT, key, forward_process):
#     """
#     We assume that 1 corrector step is always used after each predictor step 
#     """
#     # Assuming 1D data
#     D = config.data.seq_length
#     S = config.model.vocab_size

#     t = ts[0]
#     x = xT
    
#     poisson_jump = poisson_jump_reject

#     corrector = config.sampler.corrector

#     if corrector == "barker":
#         corrector_rate = barker_corrector
#     elif corrector == "mpf":
#         corrector_rate = mpf_corrector
#     elif corrector == "forward_backward":
#         corrector_rate = forward_backward_corrector
#     else:
#         raise Exception(f"Unknown corrector: {corrector}")

#     corrector_step_size = config.sampler.corrector_step_size

#     def _step(carry, idx):
#         x, key = carry
#         key, p_key, c_key = jr.split(key, 3)

#         t = ts[idx]
#         dt = t - ts[idx+1]
#         res = compute_backward(x, t, apply_fn, params, config, forward_process)
#         backward_rates = res["rates"]
#         x = poisson_jump(p_key, x, backward_rates * dt)
#         # Corrector
#         x = poisson_jump(c_key, x, corrector_rate(res) * dt * corrector_step_size)

#         return (x, key), x

#     (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
#     res = compute_backward(x, t, apply_fn, params, config, forward_process)
#     x0_logits = res["x0_logits"]

#     x0_pred = jnp.argmax(x0_logits, axis=1)

#     return x0_pred, x_hist

def backward_process_pc_tau_leaping(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size

    t = ts[0]
    x = xT
    
    poisson_jump = poisson_jump_reject

    corrector = config.sampler.corrector
    start = int(len(ts) * (1 - config.sampler.corrector_entry_time))
    
    if corrector == "barker":
        corrector_rate = barker_corrector
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "forward_backward":
        corrector_rate = forward_backward_corrector
    else:
        raise Exception(f"Unknown corrector: {corrector}")

    corrector_step_size = config.sampler.corrector_step_size

    def _p_step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x = poisson_jump(p_key, x, rp * dt)

        out = {
            "x": x,
            "rp": rp
        }
        
        return (x, key), out
    
    def _pc_step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x = poisson_jump(p_key, x, rp * dt)
        # Corrector
        
        rc = corrector_rate(res)
        x = poisson_jump(c_key, x, rc * dt * corrector_step_size)

        out = {
            "x": x,
            "rp": rp,
            "rc": rc
        }
        
        return (x, key), out

    t_ids = jnp.arange(len(ts)-1)
    (x, key), out_1 = jax.lax.scan(_p_step, (xT, key), t_ids[:start])
    (x, _), out_2 = jax.lax.scan(_pc_step, (x, key), t_ids[start:])

    x_hist = {
        "x": jnp.concatenate([out_1["x"], out_2["x"]]),
        "rp": jnp.concatenate([out_1["rp"], out_2["rp"]]),
        "rc": out_2["rc"]
    }
    
    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist