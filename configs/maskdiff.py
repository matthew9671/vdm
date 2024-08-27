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

  config.data = d(
      dataset='tokenized_imagenet_256',  # cifar10/cifar10_aug/cifar10_aug_with_channel
      ignore_cache=False,
      seq_length=256,
  )

  config.model = d(
    # tpu-v3 has less memory, use smaller network?
    vocab_size=1024 + 1, # Caveat: conditional generation stuff
    hidden_size=768,
    num_hidden_layers=12, # 24
    num_attention_heads=16,
    intermediate_size=1024, #3072
    hidden_dropout_prob=0.1, 
    attention_probs_dropout_prob=0.1, # Same as hidden dropout prob
    max_position_embeddings=256 + 1, # seq length + 1?
    # Transformer configs
    # patch_size = 16,
    # mask_token_id = -1,
    # latent_size = 16,
  )

  config.noise = d(
    state_size=1024 + 1,
    rate_eps=1e-3
  )

  config.training = d(

      min_t=0.01,
      max_t=1.,
      eps=1e-6,
      nll_weight=.01,

      seed=1,
      substeps=1,
      num_steps_lr_warmup=100,
      num_steps_train=500_000, #100_000_000,
      num_steps_eval=100,
      batch_size_train=768, #1024 in paper version
      batch_size_eval=1024,
      steps_per_logging=1000,
      steps_per_eval=10_000,
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
      # TODO: we use super small learning rate for debugging
      learning_rate=1e-7, #2e-4 in paper version
      lr_decay=False,
      ema_rate=0.9999,
  )

  return config
