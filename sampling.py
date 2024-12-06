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

    out = jnp.where((jnp.sum(jump_nums, axis=1) == 1), jump_target, x)
    return out

def euler_update(key, x, rates):
    D = x.shape[0]
    eps = 1e-8
    # Mask out the self transitions
    rates = rates.at[jnp.arange(D), x].set(0.0)
    sum_rates = jnp.sum(rates, axis=1)
    # transition_logit = jnp.log(-jnp.expm1(-rates)) # Prob = 1 - exp(-rate)
    transition_logit = jnp.log(-jnp.expm1(-sum_rates))[:,None] + jnp.log(rates) - jnp.log(sum_rates + eps)[:,None]
    transition_logit = transition_logit.at[jnp.arange(D), x].set(-sum_rates)
    
    out = jr.categorical(key, transition_logit).astype(jnp.int32)
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
    # This only works for absorbing (masking) diffusion ofcourse
    # TODO: this is incorrect!
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
        "Rt_eval_x": Rt[y],
        "rate_scalar": forward_process._rate_scalar(t)
    }
    return results
    
def backward_process_tau_leaping(apply_fn, params, ts, config, xT, key, forward_process):
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size

    t = ts[0]
    x = xT
    
    if config.sampler.update_type == "euler":
        update_func = euler_update
    elif config.sampler.update_type == "tau_leaping": 
        update_func = poisson_jump_reject
    else:
        raise Exception(f"Unknown update type: {config.sampler.update_type}")

    def _step(carry, idx):
        x, key = carry
        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        backward_rates = res["rates"]
        x = update_func(key, x, backward_rates * dt)
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

def mpf_corrector_log(res):
    coeff = res["Rt_eval_x"] + res["Rt_eval_y"]
    return coeff * jnp.exp(res["log_score"] * 0.5)

def mpf_corrector_full(res):
    coeff = res["rate_scalar"]
    score = res["score"]
    return coeff * jnp.sqrt(score)

def barker_corrector(res):
    coeff = res["Rt_eval_x"] + res["Rt_eval_y"]
    score = res["score"]
    return coeff * score / (1 + score)

def barker_corrector_full(res):
    coeff = res["rate_scalar"]
    score = res["score"]
    return coeff * score / (1 + score)

def forward_backward_corrector(res):
    return res["rates"] + res["Rt_eval_x"]

def backward_process_pc_tau_leaping(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size

    t = ts[0]
    x = xT
    
    if config.sampler.update_type == "euler":
        update_func = euler_update
    elif config.sampler.update_type == "tau_leaping": 
        update_func = poisson_jump_reject
    else:
        raise Exception(f"Unknown update type: {config.sampler.update_type}")

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
        x = update_func(p_key, x, rp * dt)

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
        x = update_func(p_key, x, rp * dt)

        # Update time
        t -= dt 

        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = update_func(c_key, x, rc * dt * corrector_step_size)

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
        # "rp": jnp.concatenate([out_1["rp"], out_2["rp"]]),
        # "rc": out_2["rc"]
    }
    
    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist

def k_gillespies_update(key, x, rates, k=1, cutoff=None):
    eps = 1e-6
    key_exp, key_cat = jr.split(key)
    
    # Get holding times for each dimension
    D = x.shape[0]
    rates = rates.at[jnp.arange(D), x].set(eps)
            
    # Compute total rate (D,)
    rates_sum = jnp.sum(rates, axis=-1)
    # Sample a holding time (D,)
    taus = jr.exponential(key, shape=(D,)) / rates_sum
    # Find which locations each dimension would transition to conditioning on a transition
    jump_target = jr.categorical(key_cat, jnp.log(rates + eps)).astype(jnp.int32)
    
    taus_sorted = jnp.sort(taus, axis=-1)
    # Obtains cut off threshold given the number of updates.
    cutoff = cutoff or taus_sorted[k] # jnp.take_along_axis(taus, k, axis=-1)

    # TODO: add safety that prevents updates if rates_sum is lower than some threshold
    out = jnp.where((taus <= cutoff), jump_target, x)
    return out, cutoff

def backward_process_pc_k_gillespies(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size
    corrector = config.sampler.corrector
    k = config.sampler.k
    corrector_entry_time = config.sampler.corrector_entry_time
    corrector_cutoff = config.sampler.corrector_step_cutoff
    
    t = ts[0]
    x = xT

    update_func = poisson_jump_reject
    
    if corrector == "barker":
        corrector_rate = barker_corrector
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "forward_backward":
        corrector_rate = forward_backward_corrector
    else:
        raise Exception(f"Unknown corrector: {corrector}")

    def _corrector_entry_cond(state):
        _, x, t, _ = state
        no_corrector = (jnp.sum(x == (S-1)) / D) > corrector_entry_time
        not_at_end = t > ts[-1]
        return jnp.logical_and(no_corrector, not_at_end)
    
    def _end_cond(state):
        key, x, t, i = state
        not_at_end = t > ts[-1]
        return not_at_end

    def _p_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x, dt = k_gillespies_update(p_key, x, rp, k=k)

        t -= dt 
        
        return (key, x, t, nfe+1)
    
    def _pc_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x, dt = k_gillespies_update(p_key, x, rp, k=k)

        t -= dt 
        
        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x, _ = k_gillespies_update(c_key, x, rc, cutoff=corrector_cutoff)

        return (key, x, t, nfe+2)

    key, x, t, nfe = jax.lax.while_loop(_corrector_entry_cond, _p_step, (key, xT, t, 0))
    _, x, t, nfe = jax.lax.while_loop(_end_cond, _pc_step, (key, x, t, nfe))
    
    x_hist = {
        "t": t,
        "nfe": nfe,
        "mask_count": jnp.sum(x == (S-1))
    }

    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist

def backward_process_pc_k_gillespies_euler(apply_fn, params, ts, config, xT, key, forward_process):
    """
    This implementation uses euler steps for the corrector
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.model.vocab_size
    corrector = config.sampler.corrector
    k = config.sampler.k
    corrector_entry_time = config.sampler.corrector_entry_time
    corrector_cutoff = config.sampler.corrector_step_cutoff

    # Scale the corrector step size with the average predictor step size
    corrector_step_size = 1 / D * k * config.sampler.corrector_step_size
    
    t = ts[0]
    x = xT

    update_func = poisson_jump_reject
    
    if corrector == "barker":
        corrector_rate = barker_corrector
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "forward_backward":
        corrector_rate = forward_backward_corrector
    else:
        raise Exception(f"Unknown corrector: {corrector}")

    def _corrector_entry_cond(state):
        _, x, t, _ = state
        no_corrector = (jnp.sum(x == (S-1)) / D) > corrector_entry_time
        not_at_end = t > ts[-1]
        return jnp.logical_and(no_corrector, not_at_end)
    
    def _end_cond(state):
        key, x, t, i = state
        not_at_end = t > ts[-1]
        return not_at_end

    def _p_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x, dt = k_gillespies_update(p_key, x, rp, k=k)

        t -= dt 
        
        return (key, x, t, nfe+1)
    
    def _pc_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x, dt = k_gillespies_update(p_key, x, rp, k=k)

        t -= dt 
        
        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = euler_update(c_key, x, rc * corrector_step_size)

        return (key, x, t, nfe+2)

    key, x, t, nfe = jax.lax.while_loop(_corrector_entry_cond, _p_step, (key, xT, t, 0))
    _, x, t, nfe = jax.lax.while_loop(_end_cond, _pc_step, (key, x, t, nfe))
    
    x_hist = {
        "t": t,
        "nfe": nfe,
        "mask_count": jnp.sum(x == (S-1))
    }

    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist