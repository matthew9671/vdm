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


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the hyperparameters for the model"""
  config = ml_collections.ConfigDict()
  config.exp_name = "exp_vdm"
  config.model_type = "model_transformer"
  config.ckpt_restore_dir = 'None'
  # config.ckpt_restore_dir = 'gs://maskdiff/cond_hollow/240928/checkpoints-0/'

  config.use_hollow_transformer = True
  
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
    intermediate_size=3072 // 3,
    hidden_dropout_prob=0.1, 
    attention_probs_dropout_prob=0.1, # Same as hidden dropout prob
    max_position_embeddings=256 + 2, # label at start and end of sequence (because of the 2 streams)
    num_layers_per_mixed=24,
  )

  config.sampler = d(
    seed=42,
    num_steps=128, # Cut the number of steps in half due to using correctors
    max_samples=10_000,
    update_type="euler", # "tau_leaping", "gillespies", "euler"
    
    output_file_name="1mpf_128psteps",
    # corrector=None,
    corrector="mpf", corrector_step_size=.075,
    # corrector="barker", corrector_step_size=2.,
    # corrector="forward_backward", corrector_step_size=5.,
    corrector_entry_time=0.5,
    num_corrector_steps=1,

    # k-Gillespies
    # k=1,
    # corrector_step_cutoff=None,
  )

  config.noise = d(
    state_size=1024 + 1,
    rate_eps=1e-3
  )

  config.training = d(

      min_t=0.01,
      max_t=1.,
      eps=1e-6,
      nll_weight=0.0,

      seed=1,
      substeps=1,
      num_steps_lr_warmup=100,
      num_steps_train=700_000, #100_000_000,
      num_steps_eval=100,
      batch_size_train=768, #1024 in paper version
      batch_size_eval=1024,
      steps_per_logging=100,
      steps_per_eval=1_000,
      steps_per_save=10_000,
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
