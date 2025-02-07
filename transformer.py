# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""MaskGIT Transformer for masked visual token modeling (MVTM) based on BERT.

The transformer is implemented based on a simplified version of BERT
[https://arxiv.org/abs/1810.04805]. Specifically, the part on next sentence
prediction and segment ids are removed from BERT. Taking the masked tokens as
inputs, the model predicts the probability of all individual tokens.

For details, please see https://arxiv.org/abs/2012.09841.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

import math

LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT
RMSNORM_EPSILON = 1e-5  # RMSNorm from MD4

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


def truncated_normal(stddev: Union[float, jnp.ndarray], dtype=jnp.float32):

  def init(key: jnp.ndarray, shape: Iterable[int], dtype: jnp.dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev

  return init

def cosine_fixed_positional_embedding(seq_len: int, dim: int):
    """
    Generate cosine fixed positional embeddings.
    
    Args:
        seq_len (int): The maximum sequence length.
        dim (int): The embedding dimension.
    
    Returns:
        jnp.ndarray: A (seq_len, dim) tensor containing the positional embeddings.
    """
    positions = jnp.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    div_term = 10000 ** (jnp.arange(dim) / dim)  # Shape: (dim,)

    pos_embedding = jnp.cos(positions / div_term)  # Shape: (seq_len, dim)
    
    return pos_embedding

class Attention(nn.Module):
  """Attention layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_mask = nn.make_attention_mask(input_mask, input_mask)
    attention_output = nn.attention.SelfAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(layer_input, attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic)
    attention_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + layer_input)

    return attention_output


class Mlp(nn.Module):
  """MLP layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  intermediate_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, attention_output: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    intermediate_output = nn.Dense(
        features=self.intermediate_size,
        kernel_init=self.initializer_fn,
        name='intermediate_output')(
            attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    layer_output = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='layer_output')(
            intermediate_output)
    layer_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        layer_output, deterministic=deterministic)
    layer_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='layer_output_ln')(
            layer_output + attention_output)

    return layer_output


class TransformerLayer(nn.Module):
  """A single Transformer layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_output = Attention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            layer_input=layer_input,
            input_mask=input_mask,
            deterministic=deterministic)

    layer_output = Mlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output, deterministic=deterministic)

    return layer_output


class Embed(nn.Module):
  """Embeds visual tokens."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None
  use_position_embeddings: bool = True

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray,
               deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)

    if self.use_position_embeddings:
        position_embeddings = nn.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='position_embeddings')(
                position_ids)
    else:
        position_embeddings = jnp.zeros_like(word_embeddings)

    input_embeddings = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class Bias(nn.Module):
  """Adds a learnable bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class MlmLayer(nn.Module):
  """MLM layer for masked token prediction."""
  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, last_layer: jnp.ndarray,
               embeddings: jnp.ndarray) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense')(
            last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='mlm_ln')(
            mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Transformer(nn.Module):
  """Transformer modified from BERT."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 256
  initializer_range: float = 0.02

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               # Add time-conditioning
               t: float,
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
    input_ids = input_ids.astype('int32')
    input_embeddings = Embed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids, deterministic=deterministic)

    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = TransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=jnp.ones_like(input_ids, dtype=jnp.int32),
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = MlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits

# ---------------------------------------------------------------------
# Hollow transformer implementation based on the maskgit implementation
# 2025/1/19 Update: added rotary position encoding from MD4
# ---------------------------------------------------------------------

class Dropout1d(nn.Module):

  dropout_rate: float = 0.0

  def __call__(self, x, deterministic=True):
    if (self.dropout_rate > 0.0) and not deterministic:
      drop = jax.random.bernoulli(
          self.make_rng('dropout'),
          1 - self.dropout_rate,
          (x.shape[0], 1, x.shape[-1]),
      )
      x = x * drop / (1 - self.dropout_rate)
    return x

def precompute_freqs_cis(dim, end, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
  t = jnp.arange(end)
  freqs = jnp.outer(t, freqs)
  freqs_cos = jnp.cos(freqs)
  freqs_sin = jnp.sin(freqs)
  return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis, x):
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.reshape(shape)


def jax_unstack(x, axis=0):
  return [
      jax.lax.index_in_dim(x, i, axis, keepdims=False)
      for i in range(x.shape[axis])
  ]

def apply_rotary_emb(x, freqs_cos, freqs_sin):
  # reshape x to match the complex representation
  # [bs, seq_len, n_head, head_dim // 2]
  x_r, x_i = jax_unstack(x.reshape(x.shape[:-1] + (-1, 2)), -1)

  # reshape freqs_cos and freqs_sin for broadcasting
  # [1, seq_len, 1, head_dim // 2]
  freqs_cos = reshape_for_broadcast(freqs_cos, x_r)
  freqs_sin = reshape_for_broadcast(freqs_sin, x_r)

  # apply rotation using real numbers
  x_out_r = x_r * freqs_cos - x_i * freqs_sin
  x_out_i = x_r * freqs_sin + x_i * freqs_cos

  # flatten last two dimensions
  # [bs, seq_len, n_head, head_dim // 2, 2] -> [bs, seq_len, n_head, head_dim]
  x_out = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(
      x_out_r.shape[:3] + (-1,)
  )
  return x_out

# Not used if n_heads == n_kv_heads (n_rep == 1)
def repeat_kv(x, n_rep):
  bs, slen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return jnp.tile(x[:, :, :, None, :], [1, 1, 1, n_rep, 1]).reshape(
      bs, slen, n_kv_heads * n_rep, head_dim
  )

class RMSNorm(nn.Module):

  dim: int
  eps: float

  def setup(self):
    self.scale = self.param(
        'scale', lambda key, shape: jnp.ones(shape), (self.dim,)
    )

  def _norm(self, x):
    return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

  def __call__(self, x):
    output = self._norm(x)
    return output * self.scale

class MaskedAttentionWithRoPE(nn.Module):
  """
  Attention that takes in an additive mask, with the "causal" flag removed.
  Everything else is identical with the original MD4 implementation
  """

  dim: int
  n_heads: int
  n_kv_heads: int | None = None
  dropout_rate: float = 0.0
  qkv_bias: bool = True # In line with flax implementation

  def setup(self):
    self._n_kv_heads = (
        self.n_heads if self.n_kv_heads is None else self.n_kv_heads
    )
    assert self.n_heads % self._n_kv_heads == 0
    self.n_rep = self.n_heads // self._n_kv_heads
    self.head_dim = self.dim // self.n_heads
    self.wq = nn.Dense(self.n_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wk = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wv = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wo = nn.Dense(self.dim, use_bias=False)
    if self.dropout_rate > 0.0:
      self.attn_dropout = nn.Dropout(self.dropout_rate)
      self.resid_dropout = Dropout1d(self.dropout_rate)

  def __call__(self, x_q, x_kv, attn_mask, freqs_cos_q, freqs_sin_q, 
    freqs_cos_kv=None, freqs_sin_kv=None, deterministic=True):
    bsz, seqlen_q, _ = x_q.shape
    _, seqlen_kv, _ = x_kv.shape

    # QKV
    xq, xk, xv = self.wq(x_q), self.wk(x_kv), self.wv(x_kv)
    xq = xq.reshape(bsz, seqlen_q, self.n_heads, self.head_dim)
    xk = xk.reshape(bsz, seqlen_kv, self._n_kv_heads, self.head_dim)
    xv = xv.reshape(bsz, seqlen_kv, self._n_kv_heads, self.head_dim)

    # RoPE relative positional embeddings
    xq = apply_rotary_emb(xq, freqs_cos_q, freqs_sin_q)
    
    freqs_cos_kv = freqs_cos_q if freqs_cos_kv is None else freqs_cos_kv
    freqs_sin_kv = freqs_sin_q if freqs_sin_kv is None else freqs_cos_kv
    xk = apply_rotary_emb(xk, freqs_cos_kv, freqs_sin_kv)

    # grouped multiquery attention: expand out keys and values
    xk = repeat_kv(xk, self.n_rep)
    xv = repeat_kv(xv, self.n_rep)

    # make heads into a batch dimension
    xq = xq.swapaxes(1, 2)  # (bs, n_heads, seqlen_q, head_dim)
    xk = xk.swapaxes(1, 2)
    xv = xv.swapaxes(1, 2)

    scores = jnp.matmul(xq, xk.swapaxes(2, 3)) / math.sqrt(self.head_dim)

    # Assuming attn_mask already has the right shape
    scores = (
          scores + attn_mask
      )  # (bs, n_heads, seqlen_q, seqlen_kv)

    scores = nn.softmax(scores, axis=-1)
    if self.dropout_rate > 0.0:
      scores = self.attn_dropout(scores, deterministic=deterministic)
    output = jnp.matmul(scores, xv)  # (bs, n_heads, seqlen_q, head_dim)

    # restore time as batch dimension and concat heads
    output = output.swapaxes(1, 2).reshape(bsz, seqlen_q, -1)

    # final projection into the residual stream
    output = self.wo(output)
    if self.dropout_rate > 0.0:
      output = self.resid_dropout(output, deterministic=deterministic)
    return output

class CausalAttention(nn.Module):
  """Attention layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, q: jnp.ndarray, kv: jnp.ndarray,
               attention_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:

    attention_output = nn.attention.MultiHeadAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        out_features=None,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='multi_head_attention',
    )(q, kv, kv, mask=attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic)
    attention_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + q)

    return attention_output

class GenericTransformerLayer(nn.Module):
  """A single Transformer layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray,
               freqs_cos: jnp.ndarray=None, freqs_sin: jnp.ndarray=None,
               freqs_cos_kv: jnp.ndarray=None, freqs_sin_kv: jnp.ndarray=None,
               deterministic: bool=True) -> jnp.ndarray:
      
    attention_output = CausalAttention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            q=q, kv=kv, attention_mask=mask,
            deterministic=deterministic)

    # norm_layer = RMSNorm(dim=self.hidden_size, eps=RMSNORM_EPSILON)

    # # Note that this function from MD4 applies dropout
    # # but not the residual connection and layer norm
    # attention_output = MaskedAttentionWithRoPE(
    #     dim=self.hidden_size,
    #     n_heads=self.num_attention_heads,
    #     dropout_rate=self.attention_probs_dropout_prob,
    #     qkv_bias=True)(
    #         norm_layer(q), 
    #         norm_layer(kv), 
    #         mask, freqs_cos, freqs_sin, 
    #         freqs_cos_kv=freqs_cos_kv, freqs_sin_kv=freqs_sin_kv,
    #         deterministic=deterministic)
    # # Apply layer norm and residual connection
    # attention_output = norm_layer(attention_output + q)

    layer_output = Mlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output, deterministic=deterministic)

    return layer_output

class HollowTransformer(nn.Module):
  """Hollow transformer modified from BERT."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 256
  initializer_range: float = 0.02
  num_layers_per_mixed: int = 4 
  permute_positions: bool = False

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               t: float, # This is not currently used
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:

    B, L = input_ids.shape
    pad = jnp.zeros((B, 1))
    input_ids = jnp.concatenate([pad, input_ids, pad], axis=1)

    input_ids = input_ids.astype('int32')
    x = Embed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings + 2, # Including the padded values
        initializer_fn=truncated_normal(self.initializer_range),
        # No positional embeddings if we use rotary embeddings
        use_position_embeddings=True
        )(input_ids=input_ids, deterministic=deterministic)
    
    H = self.num_attention_heads

    # freqs_cos, freqs_sin = precompute_freqs_cis(self.hidden_size // self.num_attention_heads, L+2)

    # # Offset the two streams and initialize the mixed stream to None
    # freqs_cos_f, freqs_sin_f = freqs_cos[:-2], freqs_sin[:-2]
    # freqs_cos_b, freqs_sin_b = freqs_cos[2:], freqs_sin[2:]
    # freqs_cos_m, freqs_sin_m = freqs_cos[1:-1], freqs_sin[1:-1]
    # freqs_cos_m_kv = jnp.concatenate([freqs_cos_f, freqs_cos_b], axis=0)
    # freqs_sin_m_kv = jnp.concatenate([freqs_sin_f, freqs_sin_b], axis=0)

    forward_mask = jnp.tile(jnp.tril(jnp.ones((L, L)))[None, None], (B, H, 1, 1))
    backward_mask = jnp.tile(jnp.triu(jnp.ones((L, L)))[None, None], (B, H, 1, 1))

    # # Different conventions between MD4 attention and linen.MultiHeadAttention
    # forward_mask = jnp.where(forward_mask == 0, -jnp.inf, 0)
    # backward_mask = jnp.where(backward_mask == 0, -jnp.inf, 0)

    mixing_mask = jnp.concatenate([forward_mask, backward_mask], axis=-1)   

    if self.permute_positions:
      # Randomly permute the sequence after positional embeddings
      key = self.make_rng('permute')
      # Keep the first and last positions (labels) fixed
      rand_perm = jax.random.permutation(key, jnp.arange(L-2))
      rand_perm = jnp.concatenate([jnp.array([0]), rand_perm + 1, jnp.array([L-1])], axis=0)
      # Permute the sequence
      # Also keeping the paddings fixed
      x = jnp.concatenate([x[:,:1], x[:,1:-1][:,rand_perm], x[:,-1:]], axis=1)
      # In order for each position to "remember" the permutation
      # The Q vectors for the first layer is a special position embedding-only vector

      # Learnable embeddings
      position_ids = jnp.arange(L)[None, :] + 1 # +1 because of the padding
      p_emb = nn.Embed(
              num_embeddings=self.max_position_embeddings,
              features=self.hidden_size,
              embedding_init=truncated_normal(self.initializer_range),
              name='init_position_embeddings')(position_ids)
      p_emb = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='init_embeddings_ln')(p_emb)

      # Fixed embeddings
      # p_emb = cosine_fixed_positional_embedding(L, self.hidden_size)
      # p_emb = p_emb[rand_perm]
      # p_emb = jnp.tile(p_emb[None], (B, 1, 1))

      # Permute position embeddings in the same way
      p_emb = p_emb[rand_perm]
      p_emb = jnp.tile(p_emb, (B, 1, 1))

      # init_fb_layer = GenericTransformerLayer(
      #     intermediate_size=self.intermediate_size,
      #     hidden_size=self.hidden_size,
      #     hidden_dropout_prob=self.hidden_dropout_prob,
      #     num_attention_heads=self.num_attention_heads,
      #     attention_probs_dropout_prob=self.attention_probs_dropout_prob,
      #     initializer_fn=truncated_normal(self.initializer_range))

      # xf = init_fb_layer(q=p_emb, kv=x[:,:-2], mask=forward_mask, 
      #               deterministic=deterministic)
      # xb = init_fb_layer(q=p_emb, kv=x[:,2:], mask=backward_mask, 
      #               deterministic=deterministic)

      xf = x[:,:-2] + p_emb
      xb = x[:,2:] + p_emb

    else:
      xf = x[:,:-2]
      xb = x[:,2:]
    xm = jnp.zeros((B, L, self.hidden_size * 2))
    
    for i in range(self.num_hidden_layers):
      fb_layer = GenericTransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))
    #   b_layer = GenericTransformerLayer(
    #       intermediate_size=self.intermediate_size,
    #       hidden_size=self.hidden_size,
    #       hidden_dropout_prob=self.hidden_dropout_prob,
    #       num_attention_heads=self.num_attention_heads,
    #       attention_probs_dropout_prob=self.attention_probs_dropout_prob,
    #       initializer_fn=truncated_normal(self.initializer_range))
      xf = fb_layer(q=xf, kv=xf, mask=forward_mask, 
                    # freqs_cos=freqs_cos_f, freqs_sin=freqs_sin_f,
                    deterministic=deterministic)
      xb = fb_layer(q=xb, kv=xb, mask=backward_mask, 
                    # freqs_cos=freqs_cos_b, freqs_sin=freqs_sin_b,
                    deterministic=deterministic)

      if (i + 1) % self.num_layers_per_mixed == 0:
        # xm += xf + xb
        xm += jnp.concatenate([xf, xb], axis=2)
        xfb = jnp.concatenate([xf, xb], axis=1)
        m_layer = GenericTransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size * 2,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))
        xm = m_layer(q=xm, kv=xfb, mask=mixing_mask, 
                    # freqs_cos=freqs_cos_m, freqs_sin=freqs_sin_m,
                    # freqs_cos_kv=freqs_cos_m_kv, freqs_sin_kv=freqs_sin_m_kv,
                    deterministic=deterministic)

    layer_output = xm
    if self.permute_positions:
      layer_output = layer_output[:, jnp.argsort(rand_perm)]
      
    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = MlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits