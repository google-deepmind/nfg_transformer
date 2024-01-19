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

"""Tests for the NfgTransformer architecture implementation."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp

from nfg_transformer import games
from nfg_transformer import nfg_transformer


NUM_STRATEGIES = [4, 5, 6]


class NfgTransformerTest(parameterized.TestCase):

  def test_payoff(self):
    num_strategies = NUM_STRATEGIES
    key = jax.random.PRNGKey(42)
    random_payoff, _ = games.l2_invariant(key, num_strategies)
    f = hk.transform(
        lambda p: nfg_transformer.NfgPerPayoff()(
            nfg_transformer.NfgTransformer()(p)
        )
    )
    params = f.init(key, random_payoff)
    payoff = f.apply(params, key, random_payoff)
    chex.assert_trees_all_equal_shapes_and_dtypes(payoff, random_payoff)

  def test_joint(self):
    num_strategies = NUM_STRATEGIES
    key = jax.random.PRNGKey(42)
    random_payoff, _ = games.l2_invariant(key, num_strategies)
    f = hk.transform(
        lambda p: nfg_transformer.NfgPerJoint()(
            nfg_transformer.NfgTransformer()(p)
        )
    )
    params = f.init(key, random_payoff)
    joint = f.apply(params, key, random_payoff)
    chex.assert_shape(joint, num_strategies)

  def test_per_action(self):
    num_strategies = NUM_STRATEGIES
    key = jax.random.PRNGKey(42)
    random_payoff, _ = games.l2_invariant(key, num_strategies)
    f = hk.transform(
        lambda p: nfg_transformer.NfgPerAction()(
            nfg_transformer.NfgTransformer()(p)
        )
    )
    params = f.init(key, random_payoff)
    outputs = f.apply(params, key, random_payoff)
    self.assertLen(outputs, len(num_strategies))
    for output, n_a in zip(outputs, num_strategies):
      chex.assert_shape(output, (n_a,))

  def test_mask_joint_actions(self):
    num_strategies = NUM_STRATEGIES
    key = hk.PRNGSequence(42)
    payoff, _ = games.l2_invariant(next(key), num_strategies)
    mask = jax.random.bernoulli(next(key), shape=num_strategies)
    f = lambda p, m: nfg_transformer.NfgTransformer()(p, m)  # pylint: disable=unnecessary-lambda
    f = hk.without_apply_rng(hk.transform(f))
    params = f.init(next(key), payoff, mask)
    embeddings = f.apply(params, payoff, mask)

    # Compare embeddings of two payoffs where only masked values differ.
    new_payoff, _ = games.l2_invariant(next(key), num_strategies)
    new_payoff = jnp.where(mask[jnp.newaxis], payoff, new_payoff)
    self.assertFalse(jnp.all(payoff == new_payoff))

    new_embeddings = f.apply(params, new_payoff, mask)
    for emb1, emb2 in zip(embeddings, new_embeddings):
      self.assertTrue(jnp.all(emb1 == emb2))

  def test_mask_player_actions(self):
    num_strategies = NUM_STRATEGIES
    key = hk.PRNGSequence(42)
    payoff, _ = games.l2_invariant(next(key), num_strategies)
    mask = jnp.ones(shape=num_strategies)
    mask = mask.at[:2, :, :].set(0)
    mask = mask.at[:, :, 2:3].set(0)

    f = lambda p, m: nfg_transformer.NfgTransformer()(p, m)  # pylint: disable=unnecessary-lambda
    f = hk.without_apply_rng(hk.transform(f))
    params = f.init(next(key), payoff, mask)
    embeddings = f.apply(params, payoff, mask)

    with self.subTest("masked_actions_have_zero_embeddings"):
      self.assertTrue(jnp.all(embeddings[0][:2] == 0))
      self.assertTrue(jnp.all(embeddings[2][2:3] == 0))

    # Compare embeddings of two payoffs where only masked values differ.
    new_payoff, _ = games.l2_invariant(next(key), num_strategies)
    new_payoff = jnp.where(mask[jnp.newaxis], payoff, new_payoff)
    self.assertFalse(jnp.all(payoff == new_payoff))

    with self.subTest("other_actions_have_same_embeddings"):
      new_embeddings = f.apply(params, new_payoff, mask)
      for emb1, emb2 in zip(embeddings, new_embeddings):
        self.assertTrue(jnp.all(emb1 == emb2))


if __name__ == "__main__":
  absltest.main()
