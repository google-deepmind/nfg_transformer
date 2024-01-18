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

"""Objective functions related to learning to solve for equilibria."""

import functools
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


def _mul_exp(x: jnp.ndarray, logp: jnp.ndarray) -> jnp.ndarray:
  p = jnp.exp(logp)
  # If p==0, the gradient with respect to logp is zero,
  # so we can replace the possibly non-finite `x` with zero.
  x = jnp.where(p == 0, 0.0, x)
  return x * p


def _make_cce_gain_swapaxis(
    payoffs: jnp.ndarray,
    player: int,
    joint_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Returns CCE deviation gain.

  Args:
    payoffs: Array with shape [N,|T1|,...,|TN|].
    player: Integer with value p.
    joint_mask: Optional array with shape [|T1|,...,|TN|].

  Returns:
    gain: Array with shape [|S'_p|,|T1|,...,|TN|].
  """
  # Shape: [    1,|T1|,...,|Tp-1|,|Tp|,|Tp+1|,...,|TN|]
  payoff = payoffs[player : player + 1]
  # The swap is changing shapes:
  # From:  [    1,|T1|,...,|Tp-1|,|Tp|,|Tp+1|,...,|TN|]
  # To:    [|Tp|,|T1|,...,|Tp-1|,    1,|Tp+1|,...,|TN|]
  gain = jnp.swapaxes(payoff, 0, player + 1) - payoff
  if joint_mask is not None:
    # [|T1|,...,|Tp-1|,    1,|Tp+1|,...,|TN|]
    marginal_mask = jnp.any(
        joint_mask,
        axis=(*range(player), *range(player + 1, joint_mask.ndim)),
    )
    # [|Tp|,  1, ... ,    1]
    marginal_mask = jnp.expand_dims(marginal_mask, tuple(range(1, gain.ndim)))
    gain = marginal_mask * gain * joint_mask
  return gain


def _cce_gain_per_player(
    payoffs: jnp.ndarray,
    *,
    joint_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, ...]:
  """Returns CCE deviation gains.

  Args:
    payoffs: Array with shape [N,|T1|,...,|TN|].
    joint_mask: Optional array with shape [|T1|,...,|TN|].

  Returns:
    cce_gain_per_player: Tuple of arrays with shape
      [[|T'_p|,|T1|,...,|TN|]_p=1:N].
  """
  num_players = len(payoffs)
  joint_mask_per_player = (joint_mask,) * num_players
  return jax.tree_util.tree_map(
      functools.partial(_make_cce_gain_swapaxis, payoffs),
      tuple(range(num_players)),
      tuple(joint_mask_per_player),
  )


def max_deviation_gain(
    payoffs: jnp.ndarray, pred: jnp.ndarray
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
  """Returns the loss for predicting the total deviation gain per joint action.

  Args:
    payoffs: a tensor of shape [N, T, ..., T] describing the payoffs to each
      player.
    pred: a tensor of shape [T, ..., T] estimating the sum of deviation gains
      across players for each joint action.

  Returns:
    a scalar value describing the NLL loss for a joint-action being a pure NE.

  Raises:
    ValueError: one and only one of the logits and marginals can be specified.
  """
  gain = _cce_gain_per_player(payoffs)
  max_gain = jnp.max(jnp.stack([jnp.max(g_p, axis=0) for g_p in gain]), axis=0)
  loss = jnp.mean(jnp.square(max_gain - pred))
  return loss, dict(
      pred=pred,
      max_deviation_gain=max_gain,
      num_pure_ne=(max_gain == 0.0).sum(),
  )


def nash_approx(
    payoffs: jnp.ndarray,
    logits: Optional[Sequence[jnp.ndarray]] = None,
    marginals: Optional[Sequence[jnp.ndarray]] = None,
    joint_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
  """Returns approximation of softmax(logits) to a NE.

  Args:
    payoffs: a tensor of shape [N, T, ..., T] describing the payoffs to each
      player.
    logits: a tensor of shape [N, T] describing the marginal action distribution
      for each player.
    marginals: a tensor of shape [N, T] describing the marginal action
      distribution for each player.
    joint_mask: a tensor of shape [T, ..., T] masking out invalid payoffs
      entries.

  Returns:
    a scalar value describing the Nash exploitability of the action marginals
    and extra metadata.

  Raises:
    ValueError: one and only one of the logits and marginals can be specified.
  """

  if logits is None and marginals is None:
    raise ValueError("One of logits and marginals should be specified.")

  if logits is not None and marginals is not None:
    raise ValueError("Only one of logits and marginals should be specified.")

  if logits is not None:
    log_marginals = [jax.nn.log_softmax(l, -1) for l in logits]
  else:
    log_marginals = [jnp.log(m) for m in marginals]

  log_joint = sum(jnp.ix_(*log_marginals))
  entropy = -jnp.sum(_mul_exp(log_joint, log_joint))

  num_players = payoffs.ndim - 1
  cce_gain = _cce_gain_per_player(payoffs, joint_mask=joint_mask)
  joint_axes = tuple(range(1, num_players + 1))
  deviation_gain_per_player = jax.tree_map(
      lambda gain: jnp.max(jnp.sum(_mul_exp(gain, log_joint), axis=joint_axes)),
      cce_gain,
  )
  deviation_gain_per_player = jnp.stack(deviation_gain_per_player)
  nash_conv = deviation_gain_per_player.max()

  return nash_conv, dict(
      entropy=entropy, total_deviation_gain=deviation_gain_per_player.sum()
  )
