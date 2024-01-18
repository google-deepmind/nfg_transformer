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

"""Unit tests for game sampling."""

from absl.testing import absltest
from absl.testing import parameterized
import games
import jax
import numpy as np


class GamesTest(parameterized.TestCase):

  @parameterized.parameters([((16, 16),), ((64, 64, 64),), ((3, 4, 5),)])
  def test_sample_l2_invariant(self, num_strategies):
    key = jax.random.PRNGKey(42)
    payoffs, mask = games.l2_invariant(key, num_strategies)

    self.assertEqual(payoffs.shape, (len(num_strategies),) + num_strategies)
    self.assertEqual(mask.shape, num_strategies)

    np.testing.assert_allclose(payoffs.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(payoffs.std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(mask, 1.0)

  def test_empirical_disc_game(self):
    num_strategies = (16, 16)
    latent_size = 8
    key = jax.random.PRNGKey(42)
    payoffs, mask = games.empirical_disc_game(
        key, num_strategies, 0.5, latent_size
    )

    self.assertEqual(payoffs.shape, (len(num_strategies),) + num_strategies)
    self.assertEqual(mask.shape, num_strategies)
    self.assertEqual(mask.max(), 1.0)

    np.testing.assert_allclose(mask, mask.T)
    np.testing.assert_allclose(payoffs.sum(0), 1.0)


if __name__ == "__main__":
  absltest.main()
