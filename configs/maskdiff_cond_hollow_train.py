# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get configuration
import ml_collections
import subprocess

def get_git_commit_message():
  try:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        cwd="/home/yixiuz/vdm/",
    )
    return result.stdout.strip()  # Return the commit message
  except subprocess.CalledProcessError as e:
    print("Error retrieving Git commit message:", e.stderr)
    return None


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the hyperparameters for the model"""
  config = ml_collections.ConfigDict()
  config.exp_name = "exp_vdm"
  config.model_type = "model_transformer"
  # config.ckpt_restore_dir = 'None'
  config.ckpt_restore_dir = "gs://maskdiff/logs/maskdiff_cond_hollow_train/20250128-021342/checkpoints-0/"
  # config.ckpt_restore_dir = "gs://maskdiff/logs/maskdiff_cond_hollow_train/20250127-095016/checkpoints-0/"
  # config.ckpt_restore_dir = "gs://maskdiff/logs/maskdiff_cond_hollow_train/20250120-115642/checkpoints-0/"
  # config.ckpt_restore_dir = "gs://maskdiff/cond_hollow/250111/checkpoints-0/"

  config.use_hollow_transformer = True

  config.loss = "mixed" # "mask" or "nonmask" or "mixed"

  config.git_commit_message = get_git_commit_message()

  config.data = d(
      dataset='tokenized_imagenet_256',  # cifar10/cifar10_aug/cifar10_aug_with_channel
      ignore_cache=False,
      seq_length=256,
      codebook_size=1024,
  )

  config.model = d(
    # tpu-v3 has less memory, use smaller network?
    vocab_size=1024 + 1000 + 1, # 1024 tokens, 1000 labels, 1 mask
    hidden_size=768,
    num_hidden_layers=24, # 24
    num_attention_heads=16,
    intermediate_size=2048,
    hidden_dropout_prob=0.1, 
    attention_probs_dropout_prob=0.1, # Same as hidden dropout prob
    max_position_embeddings=256 + 2, # label at start and end of sequence (because of the 2 streams)
    num_layers_per_mixed=8,
    # This should improve image generation
    # permute_positions=True,
    permute_positions=False,
  )

  config.sampler = d(
    seed=42,
    num_steps=16, # Cut the number of steps in half due to using correctors
    max_samples=10_000, # Stick with 10k samples for comparison
    # "tau_leaping", "gillespies", "euler", "gibbs", "test_convergence", "remdm"
    update_type="remdm", 
    # max_samples=128, update_type="test_convergence",
    tag="",
    corrector="gibbs", corrector_step_size=0.,
    # corrector="gibbs_uninformed", corrector_step_size=0,
    # corrector="forward_backward", corrector_step_size=4.,
    corrector_entry_time=0.9,
    num_corrector_steps=1,

    # Testing corrector convergence
    # predictor_cutoff_time=0.25, convergence_steps=100,
    # If set to true, only update masked tokens at the last argmax step
    restricted=False,
    k = 36,
    top_k_temperature=1.,
    maskgit_temperature=8.,
    # This only controls temperature for k-gibbs
    anneal_temperature=False,
    # This only applies to ReMDM
    keep_updates_constant=True,
  )

  config.noise = d(
    state_size=1024 + 1,
    rate_eps=1e-3
  )

  config.training = d(

      antithetic_time_sampling=True,

      min_t=0.01,
      max_t=1.,
      eps=1e-6,
      nll_weight=0.0,

      seed=1,
      substeps=1,
      num_steps_lr_warmup=100,
      num_steps_train=2_000_000, #100_000_000,
      num_steps_eval=100, # 512 * 100 ~ 50k val images
      batch_size_train=512, #1024 in paper version
      batch_size_eval=512,
      steps_per_logging=100,
      steps_per_eval=2500, # 1 full epoch
      steps_per_save=25_000, # ~1.5h of training time
      profile=False,
  )

  config.optimizer = d(
      name='adamw',
      args=d(
          b1=0.9,
          b2=0.99,
          eps=1e-8,
          weight_decay=0.01,
      ),
      learning_rate=1e-4, #2e-4 in paper version
      lr_decay=False,
      ema_rate=0.9999,
      # Trying gradient clipping
      # This shouldn't matter too much but should eliminate spikes
      gradient_clip_norm=1.0,
  )

  config.vqvae = ml_collections.ConfigDict()
  config.vqvae.quantizer = "vq"
  config.vqvae.codebook_size = 1024

  config.vqvae.entropy_loss_ratio = 0.1
  config.vqvae.entropy_temperature = 0.01
  config.vqvae.entropy_loss_type = "softmax"
  config.vqvae.commitment_cost = 0.25

  config.vqvae.filters = 128
  config.vqvae.num_res_blocks = 2
  config.vqvae.channel_multipliers = [1, 1, 2, 2, 4]
  config.vqvae.embedding_dim = 256
  config.vqvae.conv_downsample = False
  config.vqvae.activation_fn = "swish"
  config.vqvae.norm_type = "GN"

  return config
