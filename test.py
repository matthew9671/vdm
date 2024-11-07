# The following code snippet will be run on all TPU hosts
import jax
import socket
import fidjax
import jax.numpy as jnp
from absl import logging

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

# The psum is performed over all mapped devices across the Pod
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

# Print from a single host to avoid duplicated output
print(f"What is going on? Says process {jax.process_index()}")

if jax.process_index() == 0:

    print('global device count:', jax.device_count())
    print('local device count:', jax.local_device_count())
    print('pmap result:', r)
    
    print("Worker " + socket.gethostname() + " has process id 0.")

    weights = '/home/yixiuz/fid/inception_v3_weights_fid.pickle?dl=1'
    reference = '/home/yixiuz/fid/VIRTUAL_imagenet256_labeled.npz'
    fid = fidjax.FID(weights, reference)
      
    all_acts = jnp.load("/home/yixiuz/logs/samples/16psteps_1mpf_size=0.04_late_entry_acts.npy", allow_pickle=True)      
    print("Loaded activations")
    
    stats = fid.compute_stats(all_acts[:1000])
    print("Computed stats")
    
    # # We have to move these to the cpu since matrix sqrt is not supported by tpus yet
    # stats_cpu = jax.device_put(stats, device=jax.devices("cpu")[0])
    # ref_cpu = jax.device_put(fid.ref, device=jax.devices("cpu")[0])
    # print("Put the arrays on the cpu")
    
    # score = fid.compute_score(stats_cpu, ref_cpu)
    # print(f"FID score: {score}")