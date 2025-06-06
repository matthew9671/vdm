import jax
from jax import numpy as jnp
from jax.nn import softmax
import jax.random as jr
from tqdm import trange
from functools import partial

from vdm.sampling import poisson_jump_reject, euler_update, mpf_corrector, barker_corrector, \
    forward_backward_corrector, k_gillespies_update, mpf_corrector_full, barker_corrector_full, \
    mpf_corrector_log

from vdm.parallel_decode import decode

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

def compute_backward(y_with_label, t, apply_fn, params, config, forward_process, rng=None):
    rng = jr.PRNGKey(0) if rng is None else rng

    y_with_label = y_with_label.flatten()
    y = y_with_label[1:-1]

    D = y.shape[0]
    S = config.data.codebook_size + 1
    mask = S-1
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
    y = y_with_label[1:-1]
    x0_logits = apply_fn({"params": params}, y_with_label[None], t, rngs={"permute": rng},
        deterministic=True)
    # Only take the valid parts of the output
    x0_logits = x0_logits[0,1:-1,:S]
    
    # Set mask logits to minus infinity and normalize
    x0_logits = x0_logits.at[:, mask].set(-jnp.inf)
    x0_logits -= jax.scipy.special.logsumexp(x0_logits, axis=-1, 
        keepdims=True)

    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)
    
    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    # However, now each dimension is assumed to be a mask token
    # This is the change that fixes everything!
    qt0_eval_y = qt0[:,mask][None] + eps

    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # Since every dimension considers itself as the mask, we set the ratio to 1
    st_eval_y = st_eval_y.at[:, mask].set(1.0)
    backward_score_to_curr = st_eval_y[jnp.arange(D), y] #+ eps
    # On mask dimensions this is dividing by 1, on non-mask it offsets the score function to be centered on y
    st_eval_y /= backward_score_to_curr[:,None]

    # log score is easier to compute
    alpha = qt0[0,0]
    log_score = x0_logits + jnp.log(alpha) - jnp.log(1-alpha)
    log_score = log_score.at[:, mask].set(0)
    log_score = log_score - log_score[jnp.arange(D), y][:, None]

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)
    
    results = {
        "score": st_eval_y,
        "log_score": log_score,
        "rates": (st_eval_y * Rt_eval_y) * y_mask,
        "x0_logits": x0_logits,
        "Rt_eval_y": Rt_eval_y,
        "Rt_eval_x": Rt[y],
        "rate_scalar": forward_process._rate_scalar(t)
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
    S = config.data.codebook_size + 1
    D = config.data.seq_length

    mask = S-1
    
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
    elif corrector == "barker_full":
        corrector_rate = barker_corrector_full
    elif corrector == "mpf":
        # corrector_rate = mpf_corrector
        corrector_rate = mpf_corrector_log
    elif corrector == "mpf_full":
        corrector_rate = mpf_corrector_full
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

        m1 = forward_process.mask_percentage(t)
        m2 = forward_process.mask_percentage(t-dt)
        unmask_prob = (m1 - m2) / m1
        update = md4_predictor_update(p_key, x[1:-1], res["x0_logits"], unmask_prob, mask=mask)

        # Only update the data, do not update the label
        x = x.at[1:-1].set(update)

        out = {
            "x": x,
            # "rp": rp
        }
        
        return (x, key), out
    
    def _pc_step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        # rp = res["rates"]

        # Changing update function from euler to MD4 (closed form?)
        # This means that we no longer need rates
        m1 = forward_process.mask_percentage(t)
        m2 = forward_process.mask_percentage(t-dt)
        unmask_prob = (m1 - m2) / m1
        update = md4_predictor_update(p_key, x[1:-1], res["x0_logits"], unmask_prob, mask=mask)

        x = x.at[1:-1].set(update)
        

        # Change current time (!!)
        t -= dt

        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = x.at[1:-1].set(update_func(c_key, x[1:-1], rc * dt * corrector_step_size))

        out = {
            "x": x,
            # "rp": rp,
            # "rc": rc
        }
        
        return (x, key), out

    t_ids = jnp.arange(len(ts)-1)
    (x, key), out_1 = jax.lax.scan(_p_step, (xT, key), t_ids[:start])
    (x, _), out_2 = jax.lax.scan(_pc_step, (x, key), t_ids[start:])

    x_hist = jnp.concatenate([out_1["x"], out_2["x"]])
    
    res = compute_backward(x, ts[-1], apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    if not config.sampler.restricted:
        x0_pred = jnp.argmax(x0_logits, axis=1)
    else:
        # Instead of potentially updating every position, update only the masks
        x0_pred = jnp.where(x[1:-1] == mask, jnp.argmax(x0_logits, axis=1), x[1:-1])

    return x0_pred, x_hist

def mask_conditional_k_gillespies_update(key, x, rates, mask=1024, k=1):
    eps = 1e-6
    key_exp, key_cat = jr.split(key)
    
    # Get holding times for each dimension
    D = x.shape[0]

    rates = rates.at[jnp.arange(D), x].set(0)
    rates = rates.at[jnp.arange(D), mask].set(0)
            
    # Compute total rate (D,)
    rates_sum = jnp.sum(rates, axis=-1)
    # Sample a holding time (D,)
    taus = jr.exponential(key, shape=(D,)) / (rates_sum + eps)
    taus = jnp.where(x == mask, jnp.inf, taus)
    # Find which locations each dimension would transition to conditioning on a transition
    jump_target = jr.categorical(key_cat, jnp.log(rates + eps)).astype(jnp.int32)

    taus_sorted = jnp.sort(taus, axis=-1)
    # Obtains cut off threshold given the number of updates.
    cutoff = taus_sorted[k-1]

    # TODO: add safety that prevents updates if rates_sum is lower than some threshold
    out = jnp.where((taus <= cutoff) & (x != mask), jump_target, x)
    return out

def mask_conditional_k_gillespies_update_mpf(key, x, x0_logits, mask=1024, k=1):
    """
    Compute MPF rates directly from logits
    """
    eps = 1e-6
    key_exp, key_cat = jr.split(key)
    
    D = x.shape[0]

    def _mpf_rates(logits):

        curr_logits = logits[jnp.arange(D), x]
        log_score = logits - curr_logits[:, None]

        log_mpf = log_score * 0.5
        log_mpf = log_mpf.at[jnp.arange(D), mask].set(-jnp.inf)

        rates = jnp.exp(log_mpf)
        return log_mpf, rates

    log_rates, rates = _mpf_rates(x0_logits)

    # Get holding times for each dimension
    rates = rates.at[jnp.arange(D), x].set(0)
    rates = rates.at[jnp.arange(D), mask].set(0)
            
    # Compute total rate (D,)
    rates_sum = jnp.sum(rates, axis=-1)
    # Sample a holding time (D,)
    taus = jr.exponential(key, shape=(D,)) / (rates_sum + eps)
    taus = jnp.where(x == mask, jnp.inf, taus)
    # Find which locations each dimension would transition to conditioning on a transition
    # It seems that it doesn't matter whether we include the self-transition or not
    # jump_target = jr.categorical(key_cat, jnp.log(rates + eps)).astype(jnp.int32)
    logits = x0_logits.at[:, mask].set(-jnp.inf)
    jump_target = jr.categorical(key_cat, logits).astype(jnp.int32)

    taus_sorted = jnp.sort(taus, axis=-1)
    # Obtains cut off threshold given the number of updates.
    cutoff = taus_sorted[k-1]

    # TODO: add safety that prevents updates if rates_sum is lower than some threshold
    out = jnp.where((taus <= cutoff) & (x != mask), jump_target, x)
    return out

def mask_conditonal_gibbs_update(key, x, x0_logits, k=1, mask=1024, temperature=0):
    D = x.shape[0]

    key1, key2 = jr.split(key)

    logits = x0_logits.at[:, mask].set(-jnp.inf)
    # Sample a bunch of new values according to denoising model
    jump_target = jr.categorical(key1, logits).astype(jnp.int32)

    # Margin trick: we use the difference between p(x) and max_{y\neq x} p(y) as confidence
    # this way we prioritize updating the dimensions that can be most improved
    x_logits = logits[jnp.arange(D), x].T
    logits = logits.at[jnp.arange(D), x].set(-jnp.inf)
    # Confidence is logits of x - logits of the best other option
    scores = x_logits - jnp.max(logits, axis=-1)
    # scores = x_logits
    
    # Add temperature annealing
    # This is minus since conventionally we add noise and take max
    scores -= temperature * jr.gumbel(key2, shape=(D,))

    scores = jnp.where(x == mask, jnp.inf, scores)
    # Want to take the k least confident dimensions
    # Trick: sort and then find the kth smallest
    thres = jnp.sort(scores, axis=-1)[k-1]
    out = jnp.where((scores <= thres) & (x != mask), jump_target, x)
    return out

def mask_conditonal_gibbs_update_uninformed(key, x, x0_logits, k=1, mask=1024, temperature=None):
    D = x.shape[0]

    key_dim, key_cat = jr.split(key)

    logits = x0_logits.at[:, mask].set(-jnp.inf)
    # Sample a bunch of new values according to denoising model
    jump_target = jr.categorical(key_cat, logits).astype(jnp.int32)
    # For random choice we just sample from uniform as the score
    scores = jr.uniform(key_dim, shape=(D,))
    # We don't want to choose the masked dimensions
    scores = jnp.where(x == mask, jnp.inf, scores)

    # Trick: sort and then find the kth smallest
    thres = jnp.sort(scores, axis=-1)[k-1]
    out = jnp.where((scores <= thres) & (x != mask), jump_target, x)
    return out

def gibbs_corrector(res):
    # Just return the denoising logits
    # Should only be used with the gibbs_update function
    return res["x0_logits"]

def md4_predictor_update(key, x, x0_logits, unmask_prob, mask=1024):
    """
    Given model predicted denoising probabilities, 
    compute and update transition probabilities from mask schedule
    """
    D = x.shape[0]
    denoising_probs = softmax(x0_logits, axis=-1)
    probs = unmask_prob * denoising_probs
    probs = probs.at[:,mask].set(1 - unmask_prob)

    to_unmask = tfd.Categorical(probs=probs).sample(seed=key)
    out = jnp.where(x == mask, to_unmask, x)
    return out

def remdm_predictor_update(key, x, x0_logits, unmask_prob, sigma_t, 
    mask=1024, softmax_temp=0.8):
    
    D = x.shape[0]
    denoising_probs = softmax(x0_logits / softmax_temp, axis=-1)
    probs = unmask_prob * denoising_probs
    probs = probs.at[:,mask].set(1 - unmask_prob)

    # Identity function but with a small probability of remasking
    remask_probs = jax.nn.one_hot(x, probs.shape[-1]) * (1 - sigma_t)
    remask_probs = remask_probs.at[:,mask].set(sigma_t)

    probs = jnp.where(x[...,None] != mask, remask_probs, probs)

    out = tfd.Categorical(probs=probs).sample(seed=key)
    return out

def backward_process_remdm(apply_fn, params, ts, config, xT, key, forward_process):

    S = config.data.codebook_size + 1
    D = config.data.seq_length
    mask = S - 1
    # k = config.sampler.k
    t = ts[0]
    x = xT
    
    # Always use the euler update for the predictor
    update_func = euler_update

    corrector = config.sampler.corrector
    if corrector == "gibbs":
        corrector_rate = gibbs_corrector
        corrector_update = mask_conditonal_gibbs_update
    elif corrector == "forward_backward" or corrector == "":
        corrector_rate = None
    else:
        raise Exception(f"Only gibbs and forward backward correctors are supported for REMDM")
        return None
    
    def _step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        
        # Changing update function from euler to MD4 (closed form?)
        # This means that we no longer need rates
        m1 = forward_process.mask_percentage(t)
        m2 = forward_process.mask_percentage(t-dt)
        unmask_prob = (m1 - m2) / m1

        if config.sampler.keep_updates_constant:
            expected_generations = jnp.round(D * (m1 - m2)).astype(int)
            # Keep the number of changes roughly constant
            k = config.sampler.k - expected_generations
            # Avoid k < 0
            k = jnp.maximum(k, 1)
        else:
            k = config.sampler.k

        if corrector == "gibbs":

            # Apply corrector updates without another forward pass
            rc = corrector_rate(res)
            temperature_coeff = t if config.sampler.anneal_temperature else 1

            c_update = corrector_update(c_key, x[1:-1], rc, k=k, mask=mask,
                temperature=config.sampler.top_k_temperature * temperature_coeff)
            x = x.at[1:-1].set(c_update)

            p_update = md4_predictor_update(p_key, x[1:-1], res["x0_logits"], unmask_prob, mask=mask)
            x = x.at[1:-1].set(p_update)

        else:

            alpha_t = 1 - m1
            alpha_s = 1 - m2

            # Recompute unmask probability because of remasking
            sigma_t_max = jnp.minimum((1 - alpha_s) / alpha_t, 1)
            sigma_t = config.sampler.sigma_scale * sigma_t_max

            unmask_prob = (alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t)

            p_update = remdm_predictor_update(p_key, x[1:-1], res["x0_logits"], unmask_prob, sigma_t, mask=mask)
            x = x.at[1:-1].set(p_update)

        out = { "x": x, }
        
        return (x, key), out

    (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
    res = compute_backward(x, t, apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    if not config.sampler.restricted:
        x0_pred = jnp.argmax(x0_logits, axis=1)
    else:
        # Instead of potentially updating every position, update only the masks
        x0_pred = jnp.where(x[1:-1] == mask, jnp.argmax(x0_logits, axis=1), x[1:-1])

    return x0_pred, x_hist["x"]

def backward_process_gibbs(p_apply_fn, p_params, ts, config, xT, key, forward_process,
    c_apply_fn=None, c_params=None):  

    S = config.data.codebook_size + 1
    D = config.data.seq_length
    mask = S - 1
    k = config.sampler.k
    t = ts[0]
    x = xT
    
    c_apply_fn = c_apply_fn or p_apply_fn
    c_params = c_params or p_params

    # Always use the euler update for the predictor
    update_func = euler_update

    corrector = config.sampler.corrector
    if corrector == "gibbs":
        corrector_rate = gibbs_corrector
        # k-Gibbs with temperature similar in MaskGIT
        # corrector_update = partial(mask_conditonal_gibbs_update, 
        #     temperature=config.sampler.top_k_temperature)
        corrector_update = mask_conditonal_gibbs_update
    elif corrector == "gibbs_uninformed":
        corrector_rate = gibbs_corrector
        corrector_update = mask_conditonal_gibbs_update_uninformed
    elif corrector == "gibbs_mpf":
        corrector_rate = gibbs_corrector
        corrector_update = mask_conditional_k_gillespies_update_mpf
    else:
        # Always use the full corrector because we allow transition between non-masks
        if "barker" in corrector:
            corrector_rate = barker_corrector_full
        elif "mpf" in corrector:
            corrector_rate = mpf_corrector_full
        else:
            raise Exception(f"Invalid corrector for Gibbs: {corrector}")
        corrector_update = mask_conditional_k_gillespies_update

    def _c_step(i, carry):
        x, key, t, k = carry
        key, c_key = jr.split(key, 2)
        
        res = compute_backward(x, t, c_apply_fn, c_params, config, forward_process)
        rc = corrector_rate(res)

        temperature_coeff = t if config.sampler.anneal_temperature else 1

        x_update = corrector_update(c_key, x[1:-1], rc, k=k, mask=mask,
            temperature=config.sampler.top_k_temperature * temperature_coeff)

        x = x.at[1:-1].set(x_update)
        
        return (x, key, t, k)
    
    def _step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, p_apply_fn, p_params, config, forward_process)
        
        # Changing update function from euler to MD4 (closed form?)
        # This means that we no longer need rates
        m1 = forward_process.mask_percentage(t)
        m2 = forward_process.mask_percentage(t-dt)
        unmask_prob = (m1 - m2) / m1
        update = md4_predictor_update(p_key, x[1:-1], res["x0_logits"], unmask_prob, mask=mask)

        # We're not using adaptive k at this time
        # # Figure out the number of changed dimension and setting k accordingly
        # k = jnp.round(jnp.sum(x[1:-1] != update)).astype(int)
        # # cap k
        # k = jnp.minimum(k, 16)
        x = x.at[1:-1].set(update)

        # Change current time (!!)
        t -= dt

        # Corrector
        if k > 0:
            # We only apply the corrector if k > 0 to avoid unnecessary computation
            x = jax.lax.cond(t <= config.sampler.corrector_entry_time,
                            lambda x: jax.lax.fori_loop(0, config.sampler.num_corrector_steps, _c_step, (x, c_key, t, k))[0],
                            lambda x: x, x)

        out = { "x": x, }
        
        return (x, key), out

    (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
    res = compute_backward(x, t, c_apply_fn, c_params, config, forward_process)
    x0_logits = res["x0_logits"]

    if not config.sampler.restricted:
        x0_pred = jnp.argmax(x0_logits, axis=1)
    else:
        # Instead of potentially updating every position, update only the masks
        x0_pred = jnp.where(x[1:-1] == mask, jnp.argmax(x0_logits, axis=1), x[1:-1])

    return x0_pred, x_hist["x"]

def maskgit(apply_fn, params, ts, config, xT, key, forward_process):

    S = config.data.codebook_size + 1
    mask = S-1

    def tokens_to_logits(y): 
        x0_logits = apply_fn({"params": params}, y, t=0, deterministic=True)
        # We keep the label dimensions because they won't be updated anyway
        return x0_logits[...,:S]

    # Add batch dimension
    # inputs = jnp.where((xT == (S-1)), mask, xT)[None]
    inputs = xT[None]
    rng = key

    x_hist = decode(inputs,
           rng,
           tokens_to_logits,
           mask_token_id=mask,
           num_iter=config.sampler.num_steps,
           start_iter=0,
           choice_temperature=config.sampler.maskgit_temperature,
           mask_scheduling_method="cosine")

    x = x_hist[0, -1]

    res = compute_backward(x, 0, apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    if not config.sampler.restricted:
        x0_pred = jnp.argmax(x0_logits, axis=1)
    else:
        # Instead of potentially updating every position, update only the masks
        x0_pred = jnp.where(x[1:-1] == mask, jnp.argmax(x0_logits, axis=1), x[1:-1])

    return x0_pred, x_hist[0]

# ---------------------------------------------------
# These are pretty much obsolete

def test_corrector_convergence(apply_fn, params, ts, config, xT, key, forward_process):
    """
    We assume that 1 corrector step is always used after each predictor step 
    """
    t = ts[0]
    x = xT
    S = config.data.codebook_size + 1
    D = config.data.seq_length
    
    update_func = euler_update

    corrector = config.sampler.corrector
    start = int(len(ts) * (1 - config.sampler.corrector_entry_time))
    end = int(len(ts) * (1 - config.sampler.predictor_cutoff_time))
    test_steps = config.sampler.convergence_steps

    if corrector == "barker":
        corrector_rate = barker_corrector
    elif corrector == "barker_full":
        corrector_rate = barker_corrector_full
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "mpf_full":
        corrector_rate = mpf_corrector_full
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

        out = {"x": x,
            # "rp": rp
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

        out = {"x": x,
            # "rp": rp,
            # "rc": rc
        }
        return (x, key), out

    def _c_step(carry, idx):
        x, key = carry
        key, c_key = jr.split(key)

        t = config.sampler.predictor_cutoff_time
        dt = 1 / config.sampler.num_steps

        # Corrector
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)
        x = x.at[1:-1].set(update_func(c_key, x[1:-1], rc * dt * corrector_step_size))

        out = {"x": x,
            # "rc": rc
        }
        return (x, key), out

    t_ids = jnp.arange(len(ts)-1)
    (x, key), out_1 = jax.lax.scan(_p_step, (xT, key), t_ids[:start])
    (x, _), out_2 = jax.lax.scan(_pc_step, (x, key), t_ids[start:end])
    (x, _), out_3 = jax.lax.scan(_c_step, (x, key), jnp.arange(test_steps))

    x_hist = jnp.concatenate([out_1["x"], out_2["x"], out_3["x"]])
    
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
    elif corrector == "barker_full":
        corrector_rate = barker_corrector_full
    elif corrector == "mpf":
        corrector_rate = mpf_corrector
    elif corrector == "mpf_full":
        corrector_rate = mpf_corrector_full
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

def maskgit_predictor_update(key, x, x0_logits, k=1, mask=1024, temperature=0, hollow=False):
    D = x.shape[0]

    key_dim, key_cat = jr.split(key)

    logits = x0_logits.at[:, mask].set(-jnp.inf)
    # Sample a bunch of new values according to denoising model
    jump_target = jr.categorical(key_cat, logits).astype(jnp.int32)
    # Compute the confidence score on each dimensions
    confidence = x0_logits[jnp.arange(D), jump_target].T
    # Add temperature annealing
    confidence += temperature * jr.gumbel(key_dim, shape=(D,))
    if not hollow:
        # This should be the original maskgit
        # Only update masks, set confidence for non-masks to 0
        confidence = jnp.where(x != mask, -jnp.inf, confidence)
        # Trick: sort and then find the kth largest
        thres = -jnp.sort(-confidence, axis=-1)[k-1]
        out = jnp.where((confidence >= thres), jump_target, x)
    else:
        # We don't care about the generated tokens
        k += jnp.sum(x != mask).astype(int)
        k = jnp.minimum(k, D)

        thres = -jnp.sort(-confidence, axis=-1)[k-1]
        out = jnp.where((confidence >= thres), jump_target, mask)
    return out

def backward_process_maskgit(apply_fn, params, ts, config, xT, key, forward_process):

    S = config.data.codebook_size + 1
    D = config.data.seq_length
    mask = S - 1
    k = config.sampler.k
    t = ts[0]
    x = xT

    predictor_update = partial(maskgit_predictor_update,
            # "Hollow maskgit" doesn't work yet
            hollow=False)
    
    corrector = config.sampler.corrector
    if corrector == "gibbs":
        corrector_rate = gibbs_corrector
        # k-Gibbs with temperature similar in MaskGIT
        # Note that the MaskGIT implementation doesn't do annealing
        corrector_update = partial(mask_conditonal_gibbs_update, 
            temperature=config.sampler.top_k_temperature)
    elif corrector == "gibbs_uninformed":
        corrector_rate = gibbs_corrector
        corrector_update = mask_conditonal_gibbs_update_uninformed
    elif corrector == "gibbs_mpf":
        corrector_rate = gibbs_corrector
        corrector_update = mask_conditional_k_gillespies_update_mpf
    elif corrector == "":
        corrector_update = None
    else:
        # Always use the full corrector because we allow transition between non-masks
        if "barker" in corrector:
            corrector_rate = barker_corrector_full
        elif "mpf" in corrector:
            corrector_rate = mpf_corrector_full
        else:
            raise Exception(f"Invalid corrector for Gibbs: {corrector}")
        corrector_update = mask_conditional_k_gillespies_update

    def _c_step(i, carry):
        x, key, t, k = carry
        key, c_key = jr.split(key, 2)
        
        res = compute_backward(x, t, apply_fn, params, config, forward_process)
        rc = corrector_rate(res)

        x_update = corrector_update(c_key, x[1:-1], rc, k=k, mask=mask)
        # This is just to test how changing k messes with recompilation

        x = x.at[1:-1].set(x_update)
        
        return (x, key, t, k)

    def _step(carry, idx):
        x, key = carry
        key, p_key, c_key = jr.split(key, 3)

        t = ts[idx]
        dt = t - ts[idx+1]
        res = compute_backward(x, t, apply_fn, params, config, forward_process)

        # Figure out how many dimensions need to be unmasked
        dm = forward_process.mask_percentage(t) - forward_process.mask_percentage(t-dt)
        k = jnp.floor(dm * D).astype(int)
        k = jnp.maximum(1, k)

        x_update = predictor_update(c_key, x[1:-1], res["x0_logits"], k=k, mask=mask,
        # TODO: This temperature annealing actually makes performance worse...?
            temperature=config.sampler.maskgit_temperature * t)

        x = x.at[1:-1].set(x_update)

        if corrector_update is not None:
            # Change current time (!!)
            t -= dt

            # Corrector
            k = config.sampler.k
            x = jax.lax.cond(t <= config.sampler.corrector_entry_time,
                            lambda x: jax.lax.fori_loop(0, config.sampler.num_corrector_steps, _c_step, (x, c_key, t, k))[0],
                            lambda x: x, x)

        out = { "x": x, }
        
        return (x, key), out

    (x, _), x_hist = jax.lax.scan(_step, (xT, key), jnp.arange(len(ts)-1))
    res = compute_backward(x, t, apply_fn, params, config, forward_process)
    x0_logits = res["x0_logits"]

    if not config.sampler.restricted:
        x0_pred = jnp.argmax(x0_logits, axis=1)
    else:
        # Instead of potentially updating every position, update only the masks
        x0_pred = jnp.where(x[1:-1] == mask, jnp.argmax(x0_logits, axis=1), x[1:-1])

    return x0_pred, x_hist["x"]