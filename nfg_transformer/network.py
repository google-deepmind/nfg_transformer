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

"""Transformer architecture for equivariant NFG representation learning."""

from typing import Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


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
      qk_channels=None,
      v_channels=None,
      name='cross_attn',
  ):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._qk_channels = qk_channels
    self._v_channels = v_channels

  def __call__(self, inputs_q, inputs_kv, attention_mask=None):
    output_channels = inputs_q.shape[-1]
    qk_channels = inputs_kv.shape[-1]

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

    # Include the residual to the query.
    x = inputs_q + attention

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


######
# NfgTransformer
######


def _to_joint_actions(action_embeddings: Sequence[jnp.ndarray]) -> jnp.ndarray:
  """Tabulate all joint actions from action embeddings."""
  joint_actions = []
  for p, s_p in enumerate(
      jnp.mgrid[tuple(map(lambda e: slice(len(e)), action_embeddings))]
  ):
    joint_actions.append(jnp.take(action_embeddings[p], s_p, axis=0))
  joint_actions = jnp.stack(joint_actions, axis=0)  # [N] + [T1, ..., TN] + [E]
  return joint_actions


class NfgTransformerBlock(hk.Module):
  """Returns a refined set of action embeddings for each player."""

  def __init__(
      self,
      num_heads,
      qk_channels: int,
      v_channels: int,
      num_self_attend_per_block: int,
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name=name)
    self._num_heads = num_heads
    self._qk_channels = qk_channels
    self._v_channels = v_channels
    self._num_self_attend_per_block = num_self_attend_per_block

  def __call__(
      self,
      payoffs: jnp.ndarray,
      action_embeddings: Sequence[jnp.ndarray],
      mask: Optional[jnp.ndarray] = None,
  ) -> Sequence[jnp.ndarray]:
    """Refine action embeddings from payoffs."""
    action_embeddings_tm1 = action_embeddings
    num_players = len(action_embeddings)
    num_channels = action_embeddings[0].shape[-1]

    if mask is None:
      mask = jnp.ones(payoffs.shape, dtype=jnp.bool_)
    else:
      mask = jnp.broadcast_to(mask, payoffs.shape)  # [N, T1, ..., TN]

    # Player to co-player: each player action attends to all co-player actions
    # and the payoffs each player receives under the joint action.
    actions = _to_joint_actions(action_embeddings_tm1)  # [N, T1, ..., TN, E]
    action_value = jnp.concatenate([actions, payoffs[..., jnp.newaxis]], -1)
    action_value = jax.nn.gelu(
        conv_1d(num_channels, name='action_value')(action_value)
    )
    action_value = jnp.reshape(action_value, [num_players, -1, num_channels])
    attention_mask = jnp.reshape(mask, [num_players, -1, 1])
    attention_mask = jax.vmap(jnp.outer, in_axes=1, out_axes=1)(
        attention_mask, attention_mask
    )
    action_to_joint = jax.vmap(
        SelfAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            name='a2j',
        ),
        in_axes=1,
        out_axes=1,
    )(action_value, attention_mask)
    action_to_joint = jnp.reshape(
        action_to_joint, payoffs.shape + (num_channels,)
    )  # [N, T1, ..., TN, E]

    # Action to play: each player action attends to all joint-actions to which
    # it has participated.
    cross_attention = jax.vmap(
        CrossAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            name='a2p',
        )
    )
    action_masks = []
    action_embeddings = []
    for p in range(num_players):
      # Compute key-values for each of player p's actions.
      # [T1, ..., TN, E] --> [Tp, prod(T_notp), E]
      kv = jnp.moveaxis(action_to_joint[p], p, 0)
      kv = jnp.reshape(kv, [kv.shape[0], -1, kv.shape[-1]])

      # [T1, ..., TN] --> [Tp, prod(T_notp)]
      kv_mask = jnp.moveaxis(mask[p], p, 0)
      kv_mask = jnp.reshape(kv_mask, [kv_mask.shape[0], -1])

      # Mask out information from action embeddings who did not participate in
      # any joint-action under the payoffs.
      action_masks.append(kv_mask.any(axis=-1))

      q = jnp.expand_dims(action_embeddings_tm1[p], axis=1)  # [Tp, 1, E]
      qkv_mask = jnp.expand_dims(kv_mask, axis=1)  # [Tp, 1, prod(T_notp)]

      action_to_plays_p = cross_attention(q, kv, qkv_mask)  # [Tp, 1, E]
      action_embeddings.append(jnp.squeeze(action_to_plays_p, axis=1))

    # Action to action: each player action attends to all actions of all
    # players or attends to all actions of that player.
    # [Sum(Tp), E]
    action_embeddings = jnp.concatenate(action_embeddings, axis=0)
    # [Sum(Tp)]
    action_mask = jnp.concatenate(action_masks, axis=0)
    qkv_mask = jnp.outer(action_mask, action_mask)
    for _ in range(self._num_self_attend_per_block):
      action_embeddings = SelfAttention(
          num_heads=self._num_heads,
          qk_channels=self._qk_channels,
          v_channels=self._v_channels,
          name='a2a',
      )(action_embeddings, qkv_mask)
    action_embeddings = jnp.split(
        action_embeddings,
        np.cumsum([len(e) for e in action_embeddings_tm1])[:-1],
    )

    # Preserve input action-embeddings when they do not participate in any
    # joint-action described by the payoffs.
    #
    # This should be optional. Invalid actions would have arbitrary embeddings
    # values which should not be used. And they shouldn't be used in the
    # sequence of transformer blocks if the masking is done correctly.
    # Nevertheless, we ensure invalid embeddings are unchanged from the inputs
    # to avoid misuse downstream (e.g. at decoding time).
    action_embeddings_tp1 = []
    for p in range(num_players):
      action_embeddings_tp1.append(
          jnp.where(
              action_masks[p][..., jnp.newaxis],
              action_embeddings[p],
              action_embeddings_tm1[p],
          )
      )
    return action_embeddings_tp1


class NfgTransformer(hk.Module):
  """Transformer architectuer for representing normal-form games."""

  def __init__(
      self,
      num_action_channels: int = 16,
      num_blocks: int = 2,
      num_heads: int = 4,
      num_qkv_channels: int = 16,
      num_self_attend_per_block: int = 1,
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name=name)
    self._num_action_channels = num_action_channels
    self._num_qkv_channels = num_qkv_channels
    self._num_heads = num_heads
    self._num_blocks = num_blocks
    self._num_self_attend_per_block = num_self_attend_per_block

  def __call__(
      self,
      payoffs: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      initial_action_embeddings: Optional[Sequence[jnp.ndarray]] = None,
  ) -> Sequence[jnp.ndarray]:
    """Returns refined action embeddings from payoffs."""
    num_players, *num_actions = payoffs.shape

    if initial_action_embeddings is None:
      action_embeddings = [
          jnp.zeros((na, self._num_action_channels), dtype=jnp.float32)
          for na in num_actions
      ]
    else:
      assert len(initial_action_embeddings) == num_players
      action_embeddings = initial_action_embeddings

    for t in range(self._num_blocks):
      block = NfgTransformerBlock(
          num_heads=self._num_heads,
          qk_channels=self._num_qkv_channels,
          v_channels=self._num_qkv_channels,
          num_self_attend_per_block=self._num_self_attend_per_block,
          name=f'block_{t}',
      )
      action_embeddings = block(payoffs, action_embeddings, mask)

    action_embeddings = [
        layer_norm(embeddings, f'layer_norm_final_{p}')
        for p, embeddings in enumerate(action_embeddings)
    ]
    return action_embeddings


class NfgPerJoint(hk.Module):
  """Decode joint logits over joint actions from action embeddings."""

  def __call__(self, action_embeddings: Sequence[jnp.ndarray]) -> jnp.ndarray:
    actions = _to_joint_actions(action_embeddings)  # [N, T1, ..., TN, E]
    actions = jnp.reshape(actions, [actions.shape[0], -1, actions.shape[-1]])
    actions = actions.sum(axis=0)  # [prod(T_p), E]

    per_joint_outputs = hk.nets.MLP(
        output_sizes=(4 * action_embeddings[0].shape[-1], 1),
        activation=jax.nn.gelu,
    )(actions)
    return jnp.reshape(
        jnp.squeeze(per_joint_outputs, -1), [len(e) for e in action_embeddings]
    )


class NfgPerAction(hk.Module):
  """Decode an output for each action for each player."""

  def __call__(
      self, action_embeddings: Sequence[jnp.ndarray]
  ) -> Sequence[jnp.ndarray]:
    outputs = []
    net = conv_1d(1)
    for action_embeddings_p in action_embeddings:
      outputs_p = jnp.squeeze(net(action_embeddings_p), -1)
      outputs.append(outputs_p)
    return outputs


class NfgPerPayoff(hk.Module):
  """Decodes a payoff estimation for each player under each joint-action."""

  def __init__(
      self,
      num_heads: int = 4,
      qk_channels: int = 16,
      v_channels: int = 16,
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name=name)
    self._num_heads = num_heads
    self._qk_channels = qk_channels
    self._v_channels = v_channels

  def __call__(self, action_embeddings: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Returns per-payoff outputs under all joint actions."""
    joint = _to_joint_actions(action_embeddings)  # [N, T1, ..., TN, E]
    action_to_joint = jax.vmap(
        SelfAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            name='a2j',
        ),
        in_axes=1,
        out_axes=1,
    )(jnp.reshape(joint, [joint.shape[0], -1, joint.shape[-1]]))
    per_payoff = jnp.squeeze(conv_1d(1)(action_to_joint), axis=-1)
    return jnp.reshape(per_payoff, joint.shape[:-1])
