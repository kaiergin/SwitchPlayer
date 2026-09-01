"""Phase 4 vectorized PUCT: the parity gate, post-audit form.

The audited parity contract (see muzero_vtree's docstring): EXACT --
integer visit counts, 1e-12 values -- at matched net batch shapes, so
the exact gate runs the engine at T=1 per tree (the representation conv
stack is not batch-size-invariant; ~1e-7 latent wobble at B>1 decodes
to ~1e-3 value differences that are batch-shape noise, not semantics).
Heads are DE-ZEROED first: factory nets zero-init every final head,
which makes all values one shared constant and would leave the score
arithmetic untested (audit F1). The all-ties degenerate case keeps its
own test -- it is precisely the regime that pins tie-break and lockstep
mask machinery hardest.
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


def dezero_heads(nets, seed):
    """Audit F1: zero-init final heads make every value/reward a shared
    constant; re-randomize them so priors, values, and rewards genuinely
    differ and the score arithmetic is exercised."""
    gen = torch.Generator().manual_seed(seed)
    heads = []
    p = nets["prediction"]
    for attr in ("policy_fc", "value_fc"):
        if hasattr(p, attr):
            heads.append(getattr(p, attr))
    for owner, attr in ((nets["dynamics"], "reward_fc"),
                        (nets["value_prefix"], "fc")):
        if hasattr(owner, attr):
            heads.append(getattr(owner, attr))
    assert heads, "head layout changed; update dezero_heads"
    with torch.no_grad():
        for layer in heads:
            for prm in layer.parameters():
                prm.normal_(0, 0.5, generator=gen)


def build_world(monkeypatch, dezero=True):
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("MCTS_BATCHES", 1),
                     ("DIRICHLET_ALPHA", 0.3),
                     ("MAX_DEPTH", 12),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("MINMAX_RANGE_FLOOR", 0.01)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    torch.manual_seed(4)
    helper = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                               network_ref=None, storage_ref=None)
    nm = helper.network_manager
    nets = {n: getattr(nm, n) for n in ("representation", "dynamics",
                                        "prediction", "value_prefix")}
    if dezero:
        dezero_heads(nets, seed=21)
    engine = VectorizedPUCT(nets, ACTIONS, CPU)
    rng = np.random.default_rng(11)
    obs = rng.integers(0, 255, (3, 12, *muzero.IMAGE_RESHAPE),
                       dtype=np.uint8)
    return helper, engine, obs


@pytest.fixture
def world(monkeypatch):
    return build_world(monkeypatch, dezero=True)


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

    async def go():
        return await helper.run_full_mcts(
            0, num_simulations=sims, sim_batches=1, c_puct=c_puct,
            discount=discount, max_depth=max_depth, root_noise=root_noise)

    pol, val, has = asyncio.run(go())
    assert has
    helper.current_trajectory = None
    return np.round(pol * sims).astype(int), val


def engine_per_tree(engine, obs, **kw):
    """The exact gate runs T=1 per tree: matched net batch shapes are
    the parity contract's condition (audit F2)."""
    counts, vals = [], []
    for t in range(len(obs)):
        _, v, c = engine.search(obs[t:t + 1], tie_break="first", **kw)
        counts.append(c[0].astype(int))
        vals.append(v[0])
    return np.array(counts), np.array(vals)


@pytest.mark.parametrize("sims,max_depth,horizon", [
    (12, 12, 3),    # resets inside the tree (horizon < depth)
    (12, 2, 3),     # depth-capped: expanded leaves re-back static values
    (20, 12, 100),  # no in-tree resets
])
def test_exact_parity_real_heads(world, monkeypatch, sims, max_depth,
                                 horizon):
    helper, engine, obs = world
    monkeypatch.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
    seq = [sequential_search(helper, obs[t], sims, max_depth=max_depth)
           for t in range(len(obs))]
    counts, vals = engine_per_tree(engine, obs, num_simulations=sims,
                                   c_puct=1.25, discount=0.99,
                                   max_depth=max_depth, root_noise=False)
    assert counts.tolist() == [c.tolist() for c, _ in seq], \
        "visit counts must match the sequential engine EXACTLY"
    assert np.allclose(vals, [v for _, v in seq], atol=1e-12)


def test_exact_parity_with_injected_noise(world, monkeypatch):
    helper, engine, obs = world
    rng = np.random.default_rng(7)
    noise = rng.dirichlet([0.3] * ACTIONS, size=len(obs))
    calls = {"i": 0}

    def fixed_noise(prior, epsilon=0.25, alpha=0.25):
        n = noise[calls["i"]]
        calls["i"] += 1
        # float64 mix on BOTH sides: the sequential engine's float32
        # prior would round differently at ~1e-8 (audit F3).
        return (1 - epsilon) * np.asarray(prior, dtype=np.float64) \
            + epsilon * n

    monkeypatch.setattr(muzero, "add_dirichlet_noise", fixed_noise)
    seq = [sequential_search(helper, obs[t], 15, root_noise=True)
           for t in range(len(obs))]
    counts = []
    for t in range(len(obs)):
        _, _, c = engine.search(obs[t:t + 1], num_simulations=15,
                                c_puct=1.25, discount=0.99,
                                root_noise=True,
                                noise_override=noise[t:t + 1],
                                tie_break="first")
        counts.append(c[0].astype(int))
    assert np.array(counts).tolist() == [c.tolist() for c, _ in seq]


def test_exact_parity_per_step_reward_mode(world, monkeypatch):
    helper, engine, obs = world
    monkeypatch.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", False)
    seq = [sequential_search(helper, obs[t], 10)[0]
           for t in range(len(obs))]
    counts, _ = engine_per_tree(engine, obs, num_simulations=10,
                                c_puct=1.25, discount=0.99,
                                root_noise=False)
    assert counts.tolist() == [c.tolist() for c in seq]


def test_lockstep_parity_on_the_all_ties_tree(monkeypatch):
    """Zero-init heads: every selection is an exact 6-way tie, the
    hardest regime for the tie-break and lockstep mask machinery -- and
    value-free, so batch-shape wobble cannot move it: T=3 lockstep must
    match the sequential engine exactly (audit: keep this degenerate
    case as its own pin)."""
    helper, engine, obs = build_world(monkeypatch, dezero=False)
    seq = [sequential_search(helper, obs[t], 25)[0]
           for t in range(len(obs))]
    _, _, counts = engine.search(obs, num_simulations=25, c_puct=1.25,
                                 discount=0.99, root_noise=False,
                                 tie_break="first")
    assert counts.astype(int).tolist() == [c.tolist() for c in seq]


def test_lockstep_real_heads_is_sane_and_budget_exact(world):
    """T>1 with real heads: counts are batch-shape-sensitive at the
    ~5e-6-margin tail (documented), so no exact cross-shape assert --
    but the budget, shapes, and finiteness must hold with heavy lane
    divergence (small depth cap forces it)."""
    _, engine, obs = world
    obs16 = np.concatenate([obs] * 6)[:16]
    pol, val, counts = engine.search(obs16, num_simulations=40,
                                     c_puct=1.25, discount=0.99,
                                     max_depth=2, root_noise=False,
                                     tie_break="first")
    assert (counts.sum(axis=1) == 40).all()
    assert np.allclose(pol.sum(axis=1), 1.0)
    assert np.isfinite(val).all() and np.isfinite(pol).all()


def test_sims_zero_clamps_like_sequential(world):
    """Audit F4: the sequential engine always runs >= 1 simulation; a
    zero budget must not emit NaN policies."""
    _, engine, obs = world
    pol, val, counts = engine.search(obs[:1], num_simulations=0,
                                     c_puct=1.25, discount=0.99,
                                     root_noise=False, tie_break="first")
    assert counts.sum() == 1
    assert np.isfinite(pol).all() and np.isfinite(val).all()


def test_gumbel_scope_is_refused(world, monkeypatch):
    """Audit F5: wiring this engine under a Gumbel arm must fail loudly,
    never silently swap algorithms."""
    _, engine, obs = world
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", True)
    with pytest.raises(AssertionError, match="PUCT-only"):
        engine.search(obs[:1], num_simulations=4)


def test_random_tie_break_stays_within_budget(world):
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
    actions2, _, _ = engine.act(obs, greedy=False, training_steps=0,
                                num_simulations=8, c_puct=1.25,
                                discount=0.99, tie_break="first")
    assert all(0 <= a < ACTIONS for a in actions2)
    assert np.isfinite(val).all()
