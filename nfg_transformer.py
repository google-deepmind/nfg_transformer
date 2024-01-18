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

import attention as attn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


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
        attn.conv_1d(num_channels, name="action_value")(action_value)
    )
    action_value = jnp.reshape(action_value, [num_players, -1, num_channels])
    attention_mask = jnp.reshape(mask, [num_players, -1, 1])
    attention_mask = jax.vmap(jnp.outer, in_axes=1, out_axes=1)(
        attention_mask, attention_mask
    )
    player_to_joint = jax.vmap(
        attn.SelfAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            name="p2j",
        ),
        in_axes=1,
        out_axes=1,
    )(action_value, attention_mask)
    player_to_joint = jnp.reshape(
        player_to_joint, payoffs.shape + (num_channels,)
    )  # [N, T1, ..., TN, E]

    # Action to play: each player action attends to all joint-actions to which
    # it has participated.
    cross_attention = jax.vmap(
        attn.CrossAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=True,
            name="a2p",
        )
    )
    action_masks = []
    action_embeddings = []
    for p in range(num_players):
      # Compute key-values for each of player p's actions.
      # [T1, ..., TN, E] --> [Tp, prod(T_notp), E]
      kv = jnp.moveaxis(player_to_joint[p], p, 0)
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
      action_embeddings = attn.SelfAttention(
          num_heads=self._num_heads,
          qk_channels=self._qk_channels,
          v_channels=self._v_channels,
          name="a2a",
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
          name=f"block_{t}",
      )
      action_embeddings = block(payoffs, action_embeddings, mask)

    action_embeddings = [
        attn.layer_norm(embeddings, f"layer_norm_final_{p}")
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
    net = attn.conv_1d(1)
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
    player_to_joint = jax.vmap(
        attn.SelfAttention(
            num_heads=self._num_heads,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            name="p2j",
        ),
        in_axes=1,
        out_axes=1,
    )(jnp.reshape(joint, [joint.shape[0], -1, joint.shape[-1]]))
    per_payoff = jnp.squeeze(attn.conv_1d(1)(player_to_joint), axis=-1)
    return jnp.reshape(per_payoff, joint.shape[:-1])
