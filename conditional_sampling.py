import jax
from jax import numpy as jnp
from jax.nn import softmax
import jax.random as jr
from tqdm import trange
from functools import partial

from vdm.sampling import poisson_jump_reject, euler_update, mpf_corrector, barker_corrector, \
    forward_backward_corrector, k_gillespies_update

def compute_backward(y_with_label, t, apply_fn, params, config, forward_process):
    y_with_label = y_with_label.flatten()
    y = y_with_label[1:-1]

    D = y.shape[0]
    S = config.data.codebook_size + 1
    mask = -1
    # forward_process = config.forward_process
    min_t = config.training.min_t
    eps = config.training.eps
    qt0 = forward_process.transition(t)
    # R^d_t(*1,*2): (S, S) float array of instantenous transition rates
    # for a single dimension
    Rt = forward_process.rate(t)
    Rt_eval_y = Rt[:, y].T
    
    # Set corresponding values to mask 
    y_with_label = jnp.where((y_with_label == (S-1)), mask, y_with_label)
    x0_logits = apply_fn({"params": params}, y_with_label[None], t, 
        deterministic=True)
    # Only take the valid parts of the output
    x0_logits = x0_logits[0,1:-1,:S]
    
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
    backward_score_to_curr = st_eval_y[jnp.arange(D), y] + eps
    forward_score_from_curr = jnp.concatenate([jnp.zeros((D, S-1)), 1 / backward_score_to_curr[:, None]], axis=1)
    score = jnp.where((y != mask)[:,None], forward_score_from_curr, st_eval_y)

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
    
def backward_process_no_corrector(apply_fn, params, ts, config, xT, key, forward_process):

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
        p_key, key = jr.split(key)
        t = ts[idx]
        dt = t - ts[idx+1]

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x = x.at[1:-1].set(update_func(key, x[1:-1], rp * dt))

        return (x, key), x

    (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
    res = compute_backward(x, t, apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist

def backward_process_pc_single(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
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
        # Only update the data, do not update the label
        x = x.at[1:-1].set(update_func(p_key, x[1:-1], rp * dt))

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
        x = x.at[1:-1].set(update_func(p_key, x[1:-1], rp * dt))

        # Change current time (!!)
        t -= dt

        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = x.at[1:-1].set(update_func(c_key, x[1:-1], rc * dt * corrector_step_size))

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

def backward_process_pc_multiple(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that multiple corrector steps are used after each predictor step 
    """
    t = ts[0]
    x = xT

    corrector = config.sampler.corrector
    corrector_step_size = 1 / config.sampler.num_steps * config.sampler.corrector_step_size
    
    if config.sampler.update_type == "euler":
        update_func = euler_update
    elif config.sampler.update_type == "tau_leaping": 
        update_func = poisson_jump_reject
    else:
        raise Exception(f"Unknown update type: {config.sampler.update_type}")
    
    if corrector == "barker":
        corrector_rate = barker_corrector
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "forward_backward":
        corrector_rate = forward_backward_corrector
    else:
        raise Exception(f"Unknown corrector: {corrector}")

    def _c_step(i, carry):
        x, key, t = carry
        key, c_key = jr.split(key, 2)
        
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = x.at[1:-1].set(update_func(c_key, x[1:-1], rc * corrector_step_size))
        
        return (x, key, t)
    
    def _step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x = x.at[1:-1].set(update_func(p_key, x[1:-1], rp * dt))

        # Change current time (!!)
        t -= dt

        # Corrector
        x = jax.lax.cond(t <= config.sampler.corrector_entry_time,
                        lambda x: jax.lax.fori_loop(0, config.sampler.num_corrector_steps, _c_step, (x, c_key, t))[0],
                        lambda x: x, x) 

        return (x, key), x

    t_ids = jnp.arange(len(ts)-1)
    (x, key), x_hist = jax.lax.scan(_step, (xT, key), t_ids)
    
    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    x0_pred = jnp.argmax(x0_logits, axis=1)

    return x0_pred, x_hist

def backward_process_pc_k_gillespies(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.data.codebook_size
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
        x_update, dt = k_gillespies_update(p_key, x[1:-1], rp, k=k)
        x = x.at[1:-1].set(x_update)

        t -= dt 
        
        return (key, x, t, nfe+1)
    
    def _pc_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x_update, dt = k_gillespies_update(p_key, x[1:-1], rp, k=k)
        x = x.at[1:-1].set(x_update)

        t -= dt 
        
        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x_update, _ = k_gillespies_update(c_key, x[1:-1], rc, cutoff=corrector_cutoff)
        x = x.at[1:-1].set(x_update)

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
    We assume that 1 corrector step is always used after each predictor step 
    """
    # Assuming 1D data
    D = config.data.seq_length
    S = config.data.codebook_size
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
        x_update, dt = k_gillespies_update(p_key, x[1:-1], rp, k=k)
        x = x.at[1:-1].set(x_update)

        t -= dt 
        
        return (key, x, t, nfe+1)
    
    def _pc_step(state):
        key, x, t, nfe = state
        key, p_key, c_key = jr.split(key, 3)

        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rp = res["rates"]
        x_update, dt = k_gillespies_update(p_key, x[1:-1], rp, k=k)
        x = x.at[1:-1].set(x_update)

        t -= dt 
        
        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x_update = euler_update(c_key, x[1:-1], rc * corrector_step_size)
        x = x.at[1:-1].set(x_update)

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