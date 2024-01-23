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

"""Tests for equilibria computation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from nfg_transformer import equilibria


RPS = np.asarray(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

UNIFORM = np.zeros((2, 3), dtype=np.float32)
ONE_TWO = np.asarray([[1e9, 0.0, 0.0], [0.0, 1e9, 0.0]], dtype=np.float32)
ONE_ONE = np.asarray(
    [[0.0, -np.inf, -np.inf], [0.0, -np.inf, -np.inf]], dtype=np.float32
)

UNIFORM_JOINT = np.zeros((3, 3), dtype=np.float32)
ROCK_PAPER_SCISSORS_JOINT = np.asarray(
    [[1e9, 0.0, 0.0], [0.0, 1e9, 0.0], [0.0, 0.0, 1e9]], dtype=np.float32
)
ROCK_PAPER_JOINT = np.asarray(
    [[0.0, 1e9, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)

NO_MASK = np.ones((3, 3), dtype=np.float32)
ONE_ONE_MASK = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)
ONE_TWO_MASK = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)


class EquilibriaTest(parameterized.TestCase):

  @parameterized.parameters([
      (RPS, UNIFORM, 0.0),
      (RPS, ONE_TWO, 2.0),
  ])
  def test_nash_approx(self, payoffs, logits, expected):
    approx, _ = equilibria.nash_approx(
        payoffs=np.asarray([payoffs, -payoffs]), logits=logits
    )
    self.assertAlmostEqual(approx, expected)

  @parameterized.parameters([
      (ONE_ONE, ONE_ONE_MASK),
      (ONE_TWO, ONE_TWO_MASK),
  ])
  def test_nash_approx_mask(self, logits, joint_mask):
    batch_size = 32
    payoffs = jax.random.uniform(
        key=jax.random.PRNGKey(42), minval=-1.0, shape=(batch_size, 2, 3, 3)
    )
    approx, _ = jax.vmap(equilibria.nash_approx, in_axes=(0, None, None, None))(
        payoffs,
        logits,
        None,
        joint_mask,
    )
    self.assertAlmostEqual(approx.sum(), 0.0)

  def test_nash_approx_subgame_mask(self):
    key = jax.random.PRNGKey(41)
    k_s, k_l, k_p = jax.random.split(key, 3)
    subgame_payoffs = jax.random.uniform(key=k_s, minval=-1.0, shape=(2, 3, 3))
    payoffs = jax.random.uniform(key=k_p, minval=-1.0, shape=(2, 8, 8))
    payoffs = payoffs.at[:, :3, :3].set(subgame_payoffs)

    subgame_logits = jax.random.uniform(k_l, (2, 3), minval=-10.0, maxval=10.0)
    subgame_approx, _ = equilibria.nash_approx(subgame_payoffs, subgame_logits)

    padding_logits = jnp.full(shape=(2, 5), fill_value=-jnp.inf)
    logits = jnp.concatenate([subgame_logits, padding_logits], axis=-1)
    joint_mask = jnp.zeros(payoffs.shape[1:])
    joint_mask = joint_mask.at[:3, :3].set(1.0)
    approx, _ = equilibria.nash_approx(payoffs, logits, joint_mask=joint_mask)
    self.assertAlmostEqual(approx, subgame_approx)

  def test_nash_pure_strategies(self):
    _, extra = equilibria.max_deviation_gain(
        payoffs=np.asarray([RPS, -RPS]), pred=np.zeros(shape=(3, 3))
    )
    self.assertEqual(extra["num_pure_ne"], 0)


if __name__ == "__main__":
  absltest.main()
