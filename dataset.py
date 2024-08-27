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

"""Dataset loader and processor."""
from typing import Tuple

from clu import deterministic_data
import jax
from jax import numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_dataset(config, data_rng):
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    rng1, rng2 = jax.random.split(data_rng)
    if config.data.dataset == 'cifar10':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'cifar10_aug':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'cifar10_aug_with_channel':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment_with_channel_flip)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'imagenet32':
      _, train_ds = create_train_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_eval,
          'validation',
          rng2,
          _preprocess_cifar10)
    elif config.data.dataset == 'tokenized_imagenet_256':
      _, train_ds = create_custom_train_dataset(
          '/home/yixiuz/tokenized_imagenet/256x256.npy',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          None)

      # _, eval_ds = create_custom_eval_dataset(
      #     'tokenized_imagenet/256x256',
      #     config.training.batch_size_eval,
      #     'validation',
      #     rng2,
      #     None)

      eval_ds = train_ds

      # # Debugging
      # host_data = list(train_ds.as_numpy_iterator())[0]
      # # Print the data this host has received
      # host_index = jax.process_index()
      # print(f"Host {host_index} received data: {host_data}")

    else:
      raise Exception("Unrecognized config.data.dataset")

    return iter(train_ds), iter(eval_ds)

def load_custom_data(file_path: str) -> jnp.ndarray:
  #   """Load data from .npy files and concatenate into a single array."""
  return jnp.load(file_path, allow_pickle=True)

def apply_split(dataset, size, split):
  start, end = split.indices(size)
  split_ds = dataset.skip(start).take(end - start)
  return split_ds

def apply_split_with_sharding(dataset):
    num_shards = jax.process_count()  # Total number of hosts
    shard_index = jax.process_index()  # Index of the current host
    return dataset.shard(num_shards=num_shards, index=shard_index)

def create_custom_train_dataset(
      file_path: str,
      batch_size: int,
      substeps: int,
      data_rng,
      preprocess_fn) -> tf.data.Dataset:
  """Create custom dataset for training from .npy files."""

  # Compute batch size per device from global batch size.
  if batch_size % jax.device_count() != 0:
      raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                       f"the number of devices ({jax.device_count()}).")
  
  per_device_batch_size = batch_size // jax.device_count()

  # Load custom data
  # data = load_custom_data(file_path)
  # TODO: we are taking a subset for debuging
  data = load_custom_data(file_path)[:10000]

  # Create TensorFlow dataset
  train_ds = tf.data.Dataset.from_tensor_slices(data)

  # We don't have preprocessing
  # train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # split = tfds.split_for_jax_process('train', drop_remainder=True)
  # train_ds = tfds.load('my_dataset', split=split)
  # train_ds = apply_split(train_ds, size, split)
  train_ds = apply_split_with_sharding(train_ds)

  # Shuffle, batch, and prefetch
  batch_dims = [jax.local_device_count(), substeps, per_device_batch_size]
  train_ds = train_ds.shuffle(buffer_size=len(data))

  train_ds = train_ds.repeat(None) # Repeat infinitely

  train_ds = train_ds.batch(batch_dims[-1], drop_remainder=True)
  train_ds = train_ds.batch(batch_dims[-2], drop_remainder=True)
  train_ds = train_ds.batch(batch_dims[-3], drop_remainder=True)

  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  return None, train_ds

def create_train_dataset(
        task: str,
        batch_size: int,
        substeps: int,
        data_rng,
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  """Create datasets for training."""
  # Compute batch size per device from global batch size..
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)
  dataset_builder.download_and_prepare()

  train_split = deterministic_data.get_read_instruction_for_host(
      "train", dataset_builder.info.splits["train"].num_examples)
  batch_dims = [jax.local_device_count(), substeps, per_device_batch_size]

  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, train_ds


def create_eval_dataset(
        task: str,
        batch_size: int,
        subset: str,
        data_rng,
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  eval_split = deterministic_data.get_read_instruction_for_host(
      subset, dataset_builder.info.splits[subset].num_examples)
  batch_dims = [jax.local_device_count(), per_device_batch_size]

  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, eval_ds


def _preprocess_cifar10(features):
  """Helper to extract images from dict."""
  conditioning = tf.zeros((), dtype='uint8')
  return {"images": features["image"], "conditioning": conditioning}


def _preprocess_cifar10_augment(features):
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  if False:
    _img = tf.transpose(img, [2, 0, 1])
    _img = tf.random.shuffle(_img)
    _img = tf.transpose(_img, [1, 2, 0])
    _aug = tf.random.uniform(shape=[]) > 0.5
    img = tf.where(_aug, _img, img)
    aug = aug | _aug

  aug = tf.cast(aug, 'uint8')

  return {'images': img, 'conditioning': aug}


def _preprocess_cifar10_augment_with_channel_flip(features):
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  # random color channel flips
  _img = tf.transpose(img, [2, 0, 1])
  _img = tf.random.shuffle(_img)
  _img = tf.transpose(_img, [1, 2, 0])
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  aug = tf.cast(aug, 'uint8')

  return {'images': img, 'conditioning': aug}
