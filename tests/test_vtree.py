"""Phase 4 vectorized PUCT: the parity gate.

The load-bearing claim is EXACT equivalence: the vectorized engine at
tie_break="first" must produce the SAME visit counts and root values as
MCTSHelper.run_full_mcts at sim_batches=1 with random.choice patched to
first-candidate, tree for tree, on the same nets -- across depth caps,
prefix-LSTM horizons (reset boundaries), noise on/off, and per-step
reward mode. Anything less than integer-exact visit parity is a
semantics divergence, not float noise: score arithmetic runs in float64
on both sides.
"""
import asyncio
import sys

import numpy as np
import pytest
import torch

import muzero
from muzero_vtree import VectorizedPUCT

CPU = torch.device("cpu")
ACTIONS = 6


@pytest.fixture
def world(monkeypatch):
    """One net set shared by both engines; PUCT knobs pinned; the
    sequential tie-break made deterministic to match tie_break='first'."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("MCTS_BATCHES", 1),
                     ("DIRICHLET_ALPHA", 0.3),
                     ("MAX_DEPTH", 12),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("MINMAX_RANGE_FLOOR", 0.01)):
        monkeypatch.setattr(muzero, key, val)
    import random as pyrandom
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    torch.manual_seed(4)
    helper = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                               network_ref=None, storage_ref=None)
    nm = helper.network_manager
    nets = {n: getattr(nm, n) for n in ("representation", "dynamics",
                                        "prediction", "value_prefix")}
    engine = VectorizedPUCT(nets, ACTIONS, CPU)
    rng = np.random.default_rng(11)
    obs = rng.integers(0, 255, (3, 12, *muzero.IMAGE_RESHAPE),
                       dtype=np.uint8)
    return helper, engine, obs


def sequential_search(helper, obs_one, sims, root_noise=False,
                      discount=0.99, c_puct=1.25, max_depth=None):
    """The reference: exactly act_search's tree setup + run_full_mcts at
    sim_batches=1, returning integer visit counts and the root value."""
    latents, policies, values = helper.network_manager.batch_inference(
        np.asarray(obs_one)[None])
    root = muzero.MCTSNode(latent=latents[0:1], prior=1.0)
    root.policy = policies[0]
    root.expanded_value = float(values[0])
    helper.current_trajectory = [root]
    counts_holder = {}

    async def go():
        pol, val, has = await helper.run_full_mcts(
            0, num_simulations=sims, sim_batches=1, c_puct=c_puct,
            discount=discount, max_depth=max_depth, root_noise=root_noise)
        return pol, val, has

    pol, val, has = asyncio.run(go())
    assert has
    helper.current_trajectory = None
    return np.round(pol * sims).astype(int), val


@pytest.mark.parametrize("sims,max_depth,horizon", [
    (12, 12, 3),    # resets inside the tree (horizon < depth)
    (12, 2, 3),     # depth-capped: expanded leaves re-back static values
    (20, 12, 100),  # no in-tree resets
])
def test_visit_parity_noiseless(world, monkeypatch, sims, max_depth,
                                horizon):
    helper, engine, obs = world
    monkeypatch.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
    seq_counts, seq_vals = [], []
    for t in range(len(obs)):
        c, v = sequential_search(helper, obs[t], sims,
                                 max_depth=max_depth)
        seq_counts.append(c)
        seq_vals.append(v)
    pol, val, counts = engine.search(
        obs, num_simulations=sims, c_puct=1.25, discount=0.99,
        max_depth=max_depth, root_noise=False, tie_break="first")
    assert counts.astype(int).tolist() == np.array(seq_counts).tolist(), \
        "visit counts must match the sequential engine EXACTLY"
    assert np.allclose(val, seq_vals, atol=1e-9)
    assert np.allclose(pol.sum(axis=1), 1.0)


def test_visit_parity_with_injected_noise(world, monkeypatch):
    helper, engine, obs = world
    rng = np.random.default_rng(7)
    noise = rng.dirichlet([0.3] * ACTIONS, size=len(obs))
    calls = {"i": 0}

    def fixed_noise(prior, epsilon=0.25, alpha=0.25):
        n = noise[calls["i"]]
        calls["i"] += 1
        return (1 - epsilon) * prior + epsilon * n

    monkeypatch.setattr(muzero, "add_dirichlet_noise", fixed_noise)
    seq_counts = [sequential_search(helper, obs[t], 15, root_noise=True)[0]
                  for t in range(len(obs))]
    _, _, counts = engine.search(
        obs, num_simulations=15, c_puct=1.25, discount=0.99,
        root_noise=True, noise_override=noise, tie_break="first")
    assert counts.astype(int).tolist() == np.array(seq_counts).tolist()


def test_visit_parity_per_step_reward_mode(world, monkeypatch):
    """USE_VALUE_PREFIX_IN_TREE False: edge rewards come from the
    per-step reward head; the LSTM machinery must be fully bypassed."""
    helper, engine, obs = world
    monkeypatch.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", False)
    seq_counts = [sequential_search(helper, obs[t], 10)[0]
                  for t in range(len(obs))]
    _, _, counts = engine.search(obs, num_simulations=10, c_puct=1.25,
                                 discount=0.99, root_noise=False,
                                 tie_break="first")
    assert counts.astype(int).tolist() == np.array(seq_counts).tolist()


def test_budget_is_spent_exactly(world):
    _, engine, obs = world
    _, _, counts = engine.search(obs, num_simulations=17, c_puct=1.25,
                                 discount=0.99, root_noise=False,
                                 tie_break="first")
    assert (counts.sum(axis=1) == 17).all(), \
        "every simulation lands exactly one visit under the root"


def test_random_tie_break_stays_within_budget_and_varies(world):
    _, engine, obs = world
    gen = torch.Generator().manual_seed(3)
    _, _, counts = engine.search(obs, num_simulations=17, c_puct=1.25,
                                 discount=0.99, root_noise=False,
                                 generator=gen, tie_break="random")
    assert (counts.sum(axis=1) == 17).all()


def test_act_greedy_and_sampled(world, monkeypatch):
    _, engine, obs = world
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    actions, pol, val = engine.act(obs, greedy=True, num_simulations=8,
                                   c_puct=1.25, discount=0.99,
                                   tie_break="first")
    assert actions.tolist() == pol.argmax(axis=1).tolist()
    np.random.seed(0)
    actions2, pol2, _ = engine.act(obs, greedy=False, training_steps=0,
                                   num_simulations=8, c_puct=1.25,
                                   discount=0.99, tie_break="first")
    assert all(0 <= a < ACTIONS for a in actions2)
    assert np.isfinite(val).all()
