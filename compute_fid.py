import jax
import fidjax
from jax import numpy as jnp

if jax.process_index() == 0:
    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle?dl=1'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)

    all_acts = jnp.load("/home/yixiuz/logs/samples/converged_samples_10k_acts.npy", allow_pickle=True)

    stats = fid.compute_stats(all_acts)
    # We have to move these to the cpu since matrix sqrt is not supported by tpus yet
    stats_cpu = jax.device_put(stats, device=jax.devices("cpu")[0])
    ref_cpu = jax.device_put(fid.ref, device=jax.devices("cpu")[0])
    score = fid.compute_score(stats_cpu, ref_cpu)
    print(f"FID score: {score}")