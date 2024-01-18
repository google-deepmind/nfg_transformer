# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of self-attention and cross-attention networks."""

import haiku as hk
import jax
import jax.numpy as jnp


def attend(q, k, v, attention_mask=None):
  """Computes multi-head attention using a query, key and value.

  Args:
    q: Query with shape [q_indices, num_heads, head_dim].
    k: Key with shape [kv_indices, num_heads, head_dim].
    v: Value with shape [kv_indices, num_heads, head_dim].
    attention_mask: Array of shape [q_indices, kv_indices] indicating which
      attentions are valid

  Returns:
    Output of the attention with shape [q_indices, hiddens]
  """
  q_indices, num_heads, q_head_dim = q.shape
  kv_indices, _, v_head_dim = v.shape
  hiddens = num_heads * v_head_dim

  attention = jnp.einsum('thd,Thd->htT', q, k)

  scale = 1.0 / jnp.sqrt(q_head_dim)
  attention *= scale

  if attention_mask is not None:
    assert attention_mask.shape == (q_indices, kv_indices)
    # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
    # left-padded sampling. For more, see the colab below.
    # //experimental/users/tycai/lrl/NINF_NaN_investigation.ipynb
    large_k = jnp.array(
        1e4 if attention.dtype == jnp.float16 else 1e30, dtype=attention.dtype
    )
    attention = jnp.where(attention_mask[None, :, :], attention, -large_k)

  normalized = jax.nn.softmax(attention)
  summed = jnp.einsum('htT,Thd->thd', normalized, v)
  return jnp.reshape(summed, [q_indices, hiddens])


def conv_1d(output_channels, init_scale=1.0, with_bias=True, name=None):
  """A 1D convolution."""
  return hk.Linear(
      output_size=output_channels,
      with_bias=with_bias,
      w_init=hk.initializers.VarianceScaling(init_scale),
      name=name,
  )


def layer_norm(x, name=None):
  """Normalisation using RMS (https://arxiv.org/pdf/1910.07467.pdf)."""
  return hk.RMSNorm(axis=-1, name=name)(x)


class Dense(hk.Module):
  """A Transformer-style dense module to follow attention."""

  def __init__(
      self, widening_factor=4, init_scale=1.0, output_channels=None, name=None
  ):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._init_scale = init_scale
    self._output_channels = output_channels

  def __call__(self, x):
    output_channels = self._output_channels
    if output_channels is None:
      output_channels = x.shape[-1]
    x = conv_1d(
        output_channels=self._widening_factor * output_channels,
        init_scale=self._init_scale,
    )(x)
    x = jax.nn.gelu(x)
    return conv_1d(
        output_channels=output_channels, init_scale=self._init_scale
    )(x)


class Attention(hk.Module):
  """Multi-headed {cross, self}-attention."""

  def __init__(
      self,
      num_heads=8,
      init_scale=1.0,
      with_final_bias=True,
      final_init_scale_multiplier=1.0,
      channels_per_head=None,
      qkv_multi_head=False,
      qk_channels=None,
      v_channels=None,
      output_channels=None,
      name='attn',
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._init_scale = init_scale
    self._with_final_bias = with_final_bias
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._qkv_multi_head = qkv_multi_head

    # If none of these are passed, the Q input determines the output shape:
    self._qk_channels = qk_channels
    self._v_channels = v_channels
    self._output_channels = output_channels

  def __call__(self, inputs_q, inputs_kv, attention_mask=None):
    # Q and K must have the same number of channels.
    # Default to preserving Q's input's shape.
    if self._qk_channels is None:
      self._qk_channels = inputs_q.shape[-1]
    # V's num_channels determines the shape of the output of QKV-attention.
    # Default to the same number of channels used in the key-query operation.
    if self._v_channels is None:
      self._v_channels = self._qk_channels
    # Project the output of QKV attention to a desired number of channels.
    # Default to the same number as the output of the QKV attention operation.
    if self._output_channels is None:
      self._output_channels = self._v_channels

    assert self._qk_channels % self._num_heads == 0
    assert self._v_channels % self._num_heads == 0
    qk_channels_per_head = self._qk_channels // self._num_heads
    v_channels_per_head = self._v_channels // self._num_heads

    # Project QKV to a common feature dimension *without bias*.
    # https://arxiv.org/pdf/2302.05442.pdf
    q = conv_1d(self._qk_channels, self._init_scale, False, name='q')(inputs_q)
    k = conv_1d(self._qk_channels, self._init_scale, False, name='k')(inputs_kv)
    v = conv_1d(self._v_channels, self._init_scale, False, name='v')(inputs_kv)

    # Normalise QK prior to softmax layer.
    # https://arxiv.org/pdf/2302.05442.pdf
    q = layer_norm(q, name='layer_norm_q')
    k = layer_norm(k, name='layer_norm_k')

    # Reshape channels for multi-head attention.
    q_time, _ = q.shape
    kv_time, _ = k.shape
    q = jnp.reshape(q, [q_time, self._num_heads, qk_channels_per_head])
    k = jnp.reshape(k, [kv_time, self._num_heads, qk_channels_per_head])
    v = jnp.reshape(v, [kv_time, self._num_heads, v_channels_per_head])

    result = attend(q, k, v, attention_mask=attention_mask)
    return conv_1d(
        self._output_channels,
        with_bias=self._with_final_bias,
        init_scale=self._final_init_scale,
        name='dense',
    )(result)


class CrossAttention(hk.Module):
  """A cross-attention module, including a dense block."""

  def __init__(
      self,
      widening_factor=4,
      num_heads=8,
      att_init_scale=1.0,
      dense_init_scale=1.0,
      shape_for_attn='kv',
      use_query_residual=False,
      qk_channels=None,
      v_channels=None,
      name='cross_attn',
  ):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._shape_for_attn = shape_for_attn
    self._use_query_residual = use_query_residual
    self._qk_channels = qk_channels
    self._v_channels = v_channels

  def __call__(self, inputs_q, inputs_kv, attention_mask=None):
    output_channels = inputs_q.shape[-1]
    if self._shape_for_attn == 'q':
      qk_channels = inputs_q.shape[-1]
    elif self._shape_for_attn == 'kv':
      qk_channels = inputs_kv.shape[-1]
    else:
      raise ValueError(
          f'Unknown value {self._shape_for_attn} for shape_for_attention.'
      )

    v_channels = None
    if self._qk_channels is not None:
      qk_channels = self._qk_channels
    if self._v_channels is not None:
      v_channels = self._v_channels

    attention = Attention(
        num_heads=self._num_heads,
        init_scale=self._att_init_scale,
        qk_channels=qk_channels,
        v_channels=v_channels,
        output_channels=output_channels,
    )(
        layer_norm(inputs_q, name='layer_norm_q'),
        layer_norm(inputs_kv, name='layer_norm_kv'),
        attention_mask=attention_mask,
    )
    # Optionally include a residual to the query.
    # Consider omitting the residual if the semantics of query and output
    # are different, e.g. if queries are positions and outputs are pixels.
    if self._use_query_residual:
      x = inputs_q + attention
    else:
      x = attention

    x += Dense(
        widening_factor=self._widening_factor, init_scale=self._dense_init_scale
    )(layer_norm(x, name='layer_norm_x'))
    return x


class SelfAttention(hk.Module):
  """A self-attention module, including a dense block."""

  def __init__(
      self,
      widening_factor=4,
      num_heads=8,
      att_init_scale=1.0,
      dense_init_scale=1.0,
      qk_channels=None,
      v_channels=None,
      name='self_attn',
  ):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._qk_channels = qk_channels
    self._v_channels = v_channels

  def __call__(self, inputs, attention_mask=None):
    x = inputs
    output_channels = x.shape[-1]
    qkv_inputs = layer_norm(inputs, name='layer_norm_qkv')
    attention = Attention(
        num_heads=self._num_heads,
        init_scale=self._att_init_scale,
        qk_channels=self._qk_channels,
        v_channels=self._v_channels,
        output_channels=output_channels,
    )(qkv_inputs, qkv_inputs, attention_mask=attention_mask)
    x += attention

    x += Dense(
        widening_factor=self._widening_factor, init_scale=self._dense_init_scale
    )(layer_norm(x, name='layer_norm_x'))
    return x
