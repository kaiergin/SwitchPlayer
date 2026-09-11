"""vtree v2 -- the COMMUNAL chain-tree parity gate.

The acceptance criterion: VectorizedCommunalPUCT must reproduce
MCTSHelper.create_tree_from_game + run_communal_mcts (the archfix arm's
refresh engine) on the same trajectory, nets, seeds and windows --
BITWISE on the root children's visit counts, 1e-12 on root values and
policies, identical has_policy.

Why the gate can be bitwise even at multi-root windows (v1 needed T=1):
the sequential communal engine ALREADY forms its network batch across
the window's roots (one chunk of <= 40 leaves), so both engines run the
conv stack at identical batch shapes; there is no cross-shape wobble to
excuse. What does have to be reproduced is the intra-round ORDER --
selections run root by root with virtual loss visible across roots on
the shared chain, backups in root order -- and NumPy 2's float32
promotion on chain scalars; the two negative-control tests below prove
both are load-bearing (break either and the gate fails).

Conditions, as in the v1 gate: de-zeroed heads (factory nets zero-init
every final head), tie_break="first" with random.choice patched to the
first candidate, Dirichlet noise either patched to identity or drawn by
BOTH engines from the same np.random stream through the real
muzero.add_dirichlet_noise (the engine calls that function itself, in
root order, so the RNG consumption matches).
"""
import asyncio

import numpy as np
import pytest
import torch

import muzero
import muzero_vtree
from muzero_vtree import VectorizedCommunalPUCT, F32, WEAK
from tests.test_vtree import dezero_heads

CPU = torch.device("cpu")
ACTIONS = 6
C_PUCT = 1.25
DISCOUNT = 0.99


def build_world(monkeypatch, noise=False, dezero=True, horizon=3,
                max_depth=12, prefix=True):
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_VALUE_PREFIX_IN_TREE", prefix),
                     ("MCTS_BATCHES", 1),
                     ("DIRICHLET_ALPHA", 0.3),
                     ("MAX_DEPTH", max_depth),
                     ("PREFIX_LSTM_HORIZON", horizon),
                     ("MINMAX_RANGE_FLOOR", 0.01),
                     ("COMMUNAL_WINDOW_ROOTS", 16)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    if not noise:
        monkeypatch.setattr(muzero, "add_dirichlet_noise",
                            lambda prior, epsilon=0.25, alpha=0.25: prior)
    torch.manual_seed(4)
    helper = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                               network_ref=None, storage_ref=None)
    nm = helper.network_manager
    nets = {n: getattr(nm, n) for n in ("representation", "dynamics",
                                        "prediction", "value_prefix")}
    if dezero:
        dezero_heads(nets, seed=21)
    engine = VectorizedCommunalPUCT(nets, ACTIONS, CPU)
    return helper, engine


def make_traj(seed, length, terminated, zero_rewards=False):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 255, (length, 12, *muzero.IMAGE_RESHAPE),
                       dtype=np.uint8)
    actions = rng.integers(0, ACTIONS, length).tolist()
    actions[-1] = 0     # the tail record's action is 0 in stored trajectories
    if zero_rewards:
        rewards = [0.0] * length
    else:
        rewards = [float(x) for x in rng.choice([-1.0, 0.0, 0.0, 1.0], length)]
    return {"obs": obs, "actions": actions, "rewards": rewards,
            "terminated": terminated}


def sequential_windows(helper, traj, windows, sims, seed=5,
                       seed_values=True):
    """The reference: exactly _search_traj_communal's drive of the
    sequential engine, plus the raw (pre-bookkeeping) visit counts of
    every root's children read straight off the tree."""
    np.random.seed(seed)
    L = len(traj["actions"])
    evaluated = helper.eval_obs([(traj["obs"][i], traj["actions"][i],
                                  traj["rewards"][i]) for i in range(L)])
    helper.create_tree_from_game(evaluated, terminated=traj["terminated"],
                                 seed_values=seed_values)
    out, raw = {}, {}
    released = -1
    for widx in windows:
        res = asyncio.run(helper.run_communal_mcts(
            widx, num_simulations=sims, c_puct=C_PUCT, discount=DISCOUNT,
            chain_seeded=seed_values))
        for idx in widx:
            root = helper.current_trajectory[idx]
            raw[idx] = [root.children[a].visit_count if a in root.children
                        else 0 for a in range(ACTIONS)]
        out.update(res)
        helper.release_span(released + 1, widx[-1])
        released = widx[-1]
    helper.release_span(released + 1, L - 1)
    helper.current_trajectory = None
    return out, raw


def vectorized_windows(engine, traj, windows, sims, seed=5,
                       seed_values=True):
    np.random.seed(seed)
    lat, pol, val = engine.evaluate(traj["obs"])
    forest = engine.build([{"latents": lat, "policies": pol, "values": val,
                            "actions": traj["actions"],
                            "rewards": traj["rewards"],
                            "terminated": traj["terminated"]}],
                          seed_values=seed_values)
    out = {}
    for widx in windows:
        res = engine.search_window(forest, widx, num_simulations=sims,
                                   c_puct=C_PUCT, discount=DISCOUNT,
                                   tie_break="first", chain_seeded=seed_values)
        out.update({i: v for (_, i), v in res.items()})
        forest.release({0: widx[-1]})
    return out, forest


def assert_parity(seq, raw, vec):
    assert set(seq) == set(vec)
    for idx in sorted(seq):
        spol, sval, shas = seq[idx]
        vpol, vval, vhas, vraw = vec[idx]
        assert vraw.tolist() == raw[idx], \
            f"root {idx}: visit counts must match the sequential engine EXACTLY"
        assert shas == vhas, f"root {idx}: has_policy"
        assert np.allclose(spol, vpol, rtol=0, atol=1e-12), f"root {idx}: policy"
        # rtol=0: the true diff is bitwise 0.0 (type-faithful arithmetic);
        # numpy's default rtol would loosen the gate 1e8x at |v|~30.
        assert float(sval) == pytest.approx(vval, rel=0, abs=1e-12), \
            f"root {idx}: root value {float(sval)!r} vs {vval!r}"


def run_both(monkeypatch, traj, windows, sims, **world):
    helper, engine = build_world(monkeypatch, **world)
    seq, raw = sequential_windows(helper, traj, windows, sims)
    vec, forest = vectorized_windows(engine, traj, windows, sims)
    assert_parity(seq, raw, vec)
    return seq, vec, forest


# --------------------------------------------------------------------------
# The exact gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("root,terminated", [
    (0, True),      # window at the trajectory start
    (3, True),      # interior
    (6, True),      # gameplay child is the terminal node
    (7, False),     # the truncated tail itself (no gameplay child)
])
def test_window_of_one_exact_gate(monkeypatch, root, terminated):
    traj = make_traj(1, 8, terminated)
    run_both(monkeypatch, traj, [[root]], sims=10)


@pytest.mark.parametrize("name,length,terminated,windows,sims", [
    ("adjacent-pair", 8, True, [[2, 3]], 12),
    ("four-at-start", 10, True, [[0, 1, 2, 3]], 12),
    ("four-at-truncated-end", 10, False, [[6, 7, 8, 9]], 12),
    ("window-overlapping-terminal-end", 10, True, [[6, 7, 8, 9]], 12),
    ("eight-interior", 14, True, [list(range(3, 11))], 16),
])
def test_multi_root_window_exact(monkeypatch, name, length, terminated,
                                 windows, sims):
    """Roots share chain nodes: root j's descent reads the virtual loss
    roots < j left this round, backups land in root order. Bitwise."""
    traj = make_traj(2, length, terminated)
    seq, vec, _ = run_both(monkeypatch, traj, windows, sims)
    if name == "window-overlapping-terminal-end":
        # The terminal root: no children, no policy, RNG still consumed.
        assert seq[9][2] is False and vec[9][2] is False


@pytest.mark.parametrize("name,windows,sims", [
    # Gapped windows (audit F1): REFRESH_NO_SHARPEN_GUARD's guarded steps
    # leave holes in the eligible set; the stagger must grow with the
    # CHAIN distance between roots, not their window slot.
    ("stride-2", [list(range(0, 32, 2))], 50),
    ("irregular", [[0, 1, 4, 5, 6, 11, 12, 20, 21, 22, 23, 29, 30, 33, 35, 36]], 50),
    ("two-gapped-windows", [[0, 3, 5, 9, 10, 14], [15, 16, 21, 27, 28, 31]], 30),
])
def test_gapped_windows_exact_at_production_shape(monkeypatch, name, windows,
                                                  sims):
    traj = make_traj(70, 40, True)
    run_both(monkeypatch, traj, windows, sims=sims, noise=True, horizon=5,
             max_depth=50)


def test_two_windows_through_one_shared_chain_tree(monkeypatch):
    """Windows advance with release_span between them; the second
    window's roots may have been expanded (and visited) by the first
    window's traffic -- communal statistics, not a fresh tree."""
    traj = make_traj(3, 12, True)
    seq, vec, forest = run_both(monkeypatch, traj,
                                [[0, 1, 2, 3], [4, 5, 6, 7]], sims=10)
    # Chain nodes survive release; imagined nodes up to the last root of
    # the second window are gone.
    n = forest.count
    assert (forest.f["cidx"][:n] >= 0).sum() == 12
    owners = forest.f["owner"][:n][forest.f["cidx"][:n] < 0]
    assert (owners > 7).all()


@pytest.mark.parametrize("horizon,max_depth", [
    (1, 12),     # reset at every imagined ply
    (3, 12),     # resets inside imagined subtrees (horizon < depth)
    (100, 12),   # no in-tree resets: every chain node is still a reset
    (3, 2),      # depth-capped: expanded leaves re-back static values
])
def test_prefix_horizon_and_is_reset_edges(monkeypatch, horizon, max_depth):
    traj = make_traj(4, 10, True)
    run_both(monkeypatch, traj, [[1, 2, 3, 4]], sims=12, horizon=horizon,
             max_depth=max_depth)


def test_per_step_reward_mode(monkeypatch):
    traj = make_traj(5, 10, True)
    run_both(monkeypatch, traj, [[1, 2, 3, 4]], sims=12, prefix=False)


def test_root_noise_from_the_shared_numpy_stream(monkeypatch):
    """Noise ON: both engines call the real muzero.add_dirichlet_noise
    once per root in index order from the same np.random seed, so the
    draws AND the float32 (1-eps)*prior mixing are identical."""
    traj = make_traj(6, 10, True)
    run_both(monkeypatch, traj, [[1, 2, 3, 4]], sims=12, noise=True)


def test_trajectory_shorter_than_the_window(monkeypatch):
    """5 positions, COMMUNAL_WINDOW_ROOTS=16: one window of 4 roots (the
    terminated tail is not eligible)."""
    traj = make_traj(7, 5, True)
    helper, engine = build_world(monkeypatch)
    idx = list(range(4))
    seq, raw = sequential_windows(helper, traj, [idx], sims=8)
    np.random.seed(5)
    got = engine.search_trajectories(
        [traj], [idx], num_simulations=8, c_puct=C_PUCT, discount=DISCOUNT,
        tie_break="first")[0]
    assert set(got) == set(idx)
    for i in idx:
        assert np.allclose(got[i][0], seq[i][0], rtol=0, atol=1e-12)
        assert got[i][1] == pytest.approx(float(seq[i][1]), rel=0, abs=1e-12)


def test_search_trajectories_matches_the_communal_window_drive(monkeypatch):
    """The whole-trajectory driver: windows of COMMUNAL_WINDOW_ROOTS in
    ascending order, release after each, terminated tail excluded --
    _search_traj_communal's exact drive of the sequential engine."""
    traj = make_traj(8, 14, True)
    helper, engine = build_world(monkeypatch)
    monkeypatch.setattr(muzero, "COMMUNAL_WINDOW_ROOTS", 4)
    idx = list(range(13))
    windows = [idx[w:w + 4] for w in range(0, 13, 4)]
    seq, raw = sequential_windows(helper, traj, windows, sims=8)
    np.random.seed(5)
    got = engine.search_trajectories(
        [traj], [idx], num_simulations=8, c_puct=C_PUCT, discount=DISCOUNT,
        tie_break="first")[0]
    assert set(got) == {i for i in idx if seq[i][2]}
    for i in got:
        assert np.allclose(got[i][0], seq[i][0], rtol=0, atol=1e-12)
        assert got[i][1] == pytest.approx(float(seq[i][1]), rel=0, abs=1e-12)


# --------------------------------------------------------------------------
# Seeding and the bookkeeping visit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("terminated", [True, False])
def test_seed_values_anchor_the_behaviour_child(monkeypatch, terminated):
    """The chain backup: one visit per chain node and its realized
    Monte-Carlo return (0 at a terminal, the np.float32 network value at
    a truncated tail -- which makes the whole chain's value_sum float32
    under NumPy 2, reproduced by type code)."""
    helper, engine = build_world(monkeypatch)
    traj = make_traj(9, 8, terminated)
    L = 8
    evaluated = helper.eval_obs([(traj["obs"][i], traj["actions"][i],
                                  traj["rewards"][i]) for i in range(L)])
    helper.create_tree_from_game(evaluated, terminated=terminated,
                                 seed_values=True)
    lat, pol, val = engine.evaluate(traj["obs"])
    forest = engine.build([{"latents": lat, "policies": pol, "values": val,
                            "actions": traj["actions"],
                            "rewards": traj["rewards"],
                            "terminated": terminated}])
    f = forest.f
    for k, node in enumerate(helper.current_trajectory):
        assert node.visit_count == 1 and f["visit"][k] == 1
        assert float(node.value_sum) == f["wsum"][k].item()
        assert isinstance(node.value_sum, np.float32) == (int(f["wsum_t"][k]) == F32)
        assert node.q_value() == f["wsum"][k].item()
        assert f["done"][k].item() == node.done
        if k > 0:
            assert float(node.prior) == f["prior"][k].item()
            assert int(f["prior_t"][k]) == F32
    assert (int(f["wsum_t"][0]) == F32) == (not terminated)
    # The behaviour child's realized return is what anchors selection.
    root = helper.current_trajectory[2]
    child = root.children[traj["actions"][2]]
    assert child.q_value() == f["wsum"][3].item()
    helper.current_trajectory = None


def test_readout_subtracts_exactly_one_bookkeeping_visit(monkeypatch):
    traj = make_traj(10, 10, True)
    seq, vec, _ = run_both(monkeypatch, traj, [[1, 2, 3, 4]], sims=12)
    for idx in (1, 2, 3, 4):
        pol, val, has, raw = vec[idx]
        beh = traj["actions"][idx]
        counts = raw.astype(np.float64).copy()
        counts[beh] -= 1                       # the one bookkeeping visit
        assert counts.sum() >= 12              # own sims (+ inherited traffic)
        assert np.allclose(pol, counts / counts.sum(), rtol=0, atol=1e-12)
        assert raw[beh] >= 1


def test_neutral_chain_subtracts_nothing(monkeypatch):
    """seed_values=False / chain_seeded=False: no bookkeeping visit exists;
    readout is the raw visit fraction (parity with the sequential
    neutral chain)."""
    helper, engine = build_world(monkeypatch)
    traj = make_traj(11, 10, True)
    seq, raw = sequential_windows(helper, traj, [[2, 3, 4]], sims=10,
                                  seed_values=False)
    vec, _ = vectorized_windows(engine, traj, [[2, 3, 4]], sims=10,
                                seed_values=False)
    assert_parity(seq, raw, vec)
    for idx in (2, 3, 4):
        pol, _, _, rawc = vec[idx]
        assert rawc.sum() >= 10
        assert np.allclose(pol, rawc / rawc.sum(), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Negative controls: the gate must FAIL when the reproduced order or the
# type-faithful arithmetic is removed -- otherwise it tests nothing.
# --------------------------------------------------------------------------
def _diverges(helper, engine, traj, windows, sims):
    seq, raw = sequential_windows(helper, traj, windows, sims)
    vec, _ = vectorized_windows(engine, traj, windows, sims)
    try:
        assert_parity(seq, raw, vec)
    except AssertionError:
        return True
    return False


def test_intra_round_order_is_load_bearing(monkeypatch):
    """Pure lockstep (every root at wave 0) drops the cross-root virtual
    loss the sequential index-ordered selection leaves on shared chain
    nodes; visit counts then differ."""
    monkeypatch.setattr(VectorizedCommunalPUCT, "_ready",
                        lambda self, ws: ws["active"])
    hits = 0
    for seed, L, windows, sims in ((2, 10, [[0, 1, 2, 3]], 12),
                                   (3, 12, [[0, 1, 2, 3], [4, 5, 6, 7]], 10),
                                   (12, 12, [list(range(4, 12))], 16)):
        helper, engine = build_world(monkeypatch)
        hits += _diverges(helper, engine, make_traj(seed, L, seed != 12),
                          windows, sims)
    assert hits > 0, "removing the stagger must break parity somewhere"


def test_slot_stagger_is_wrong_for_gapped_windows(monkeypatch):
    """Audit F1's reproduction as a standing control: the ORIGINAL rule
    (stagger by window slot) must diverge on a gapped window at
    production shape, so the chain-offset rule is load-bearing."""
    def by_slot(self, ws):
        if "slot2" not in ws:
            seen, out = {}, []
            for t, _ in ws["roots"]:
                out.append(2 * seen.get(t, 0))
                seen[t] = seen.get(t, 0) + 1
            ws["slot2"] = torch.tensor(out, device=ws["active"].device)
        return ws["active"] & (ws["slot2"] <= ws["wave"])

    monkeypatch.setattr(VectorizedCommunalPUCT, "_ready", by_slot)
    hits = 0
    for seed in (70, 78):
        helper, engine = build_world(monkeypatch, noise=True, horizon=5,
                                     max_depth=50)
        hits += _diverges(helper, engine, make_traj(seed, 40, True),
                          [list(range(0, 32, 2))], 50)
    assert hits > 0, "slot-based stagger must break a gapped window"


def test_static_chain_offset_stagger_is_also_exact(monkeypatch):
    """The static chain-offset stagger (root j at wave d + 2*(i_j - i_0))
    is the proof's special case of _ready; pin that it is itself exact
    on gapped and contiguous windows (it bounds the wave count)."""
    def static(self, ws):
        return ws["active"] & (ws["slots2"] <= ws["wave"])

    monkeypatch.setattr(VectorizedCommunalPUCT, "_ready", static)
    for windows in ([list(range(0, 32, 2))], [list(range(4, 20))]):
        traj = make_traj(77, 40, True)
        run_both(monkeypatch, traj, windows, sims=20, noise=True, horizon=5,
                 max_depth=50)


def test_type_faithful_arithmetic_is_load_bearing(monkeypatch):
    """float64-everywhere differs from the sequential engine's NumPy 2
    float32 promotion on chain scalars at the 1e-6 level -- the value
    gate must catch it."""
    monkeypatch.setattr(muzero_vtree, "_tb",
                        lambda op, a, ta, b, tb: (op(a, b), torch.maximum(ta, tb)))
    monkeypatch.setattr(muzero_vtree, "_tc", lambda op, a, ta, b, tb: op(a, b))
    helper, engine = build_world(monkeypatch)
    assert _diverges(helper, engine, make_traj(2, 10, True), [[0, 1, 2, 3]], 12)


# --------------------------------------------------------------------------
# Scope, multi-tree, and the fresh-root path
# --------------------------------------------------------------------------
def test_gumbel_scope_is_refused(monkeypatch):
    helper, engine = build_world(monkeypatch)
    traj = make_traj(13, 6, True)
    lat, pol, val = engine.evaluate(traj["obs"])
    forest = engine.build([{"latents": lat, "policies": pol, "values": val,
                            "actions": traj["actions"],
                            "rewards": traj["rewards"], "terminated": True}])
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", True)
    with pytest.raises(AssertionError, match="PUCT-only"):
        engine.search_window(forest, [0, 1], num_simulations=4)


def test_released_span_cannot_be_searched_again(monkeypatch):
    """Audit F5: a root inside a released span would gather through -1
    child links (garbage on CPU, a device assert on CUDA); refuse."""
    helper, engine = build_world(monkeypatch)
    traj = make_traj(14, 8, True)
    lat, pol, val = engine.evaluate(traj["obs"])
    forest = engine.build([{"latents": lat, "policies": pol, "values": val,
                            "actions": traj["actions"],
                            "rewards": traj["rewards"], "terminated": True}])
    engine.search_window(forest, [0, 1, 2], num_simulations=4)
    forest.release({0: 2})
    with pytest.raises(AssertionError, match="released span"):
        engine.search_window(forest, [2, 3], num_simulations=4)
    with pytest.raises(AssertionError, match="outside tree"):
        engine.search_window(forest, [8], num_simulations=4)
    engine.search_window(forest, [3, 4], num_simulations=4)   # still fine


def test_rewards_must_not_be_numpy_integers(monkeypatch):
    """Audit F4: np.int64 is a STRONG scalar under NEP 50 (int + float ->
    float64), unlike a Python int; refuse rather than mistype."""
    helper, engine = build_world(monkeypatch)
    traj = make_traj(15, 4, True)
    lat, pol, val = engine.evaluate(traj["obs"])
    bad = dict(latents=lat, policies=pol, values=val, actions=traj["actions"],
               rewards=[np.int64(1)] * 4, terminated=True)
    with pytest.raises(AssertionError, match="np.floating"):
        engine.build([bad])
    with pytest.raises(TypeError):
        muzero_vtree._scalar_type(np.int64(1))
    assert muzero_vtree._scalar_type(1) == WEAK
    assert muzero_vtree._scalar_type(np.float32(1)) == F32


def test_forest_allocates_once_and_never_shrinks(monkeypatch):
    """Audit F3: a window sequence over one forest reserves up front,
    compacts IN PLACE at release (capacity never drops), and grows at
    most once."""
    helper, engine = build_world(monkeypatch, noise=True)
    traj = make_traj(16, 40, True)
    caps = []
    orig_release = muzero_vtree.ChainForest.release

    def spy_release(self, bounds):
        orig_release(self, bounds)
        caps.append(self.capacity)

    monkeypatch.setattr(muzero_vtree.ChainForest, "release", spy_release)
    forests = []
    orig_sw = VectorizedCommunalPUCT.search_window

    def grab(self, forest, *a, **kw):
        forests.append(forest)
        return orig_sw(self, forest, *a, **kw)

    monkeypatch.setattr(VectorizedCommunalPUCT, "search_window", grab)
    engine.search_trajectories([traj], [list(range(39))], window=4,
                               num_simulations=8, c_puct=C_PUCT,
                               discount=DISCOUNT, tie_break="first")
    forest = forests[0]
    assert len(caps) == 10 and all(c == caps[0] for c in caps), \
        "capacity must never shrink across releases"
    assert forest.grows <= 1
    assert forest.count >= 40      # the chain survives every release


def test_two_trees_in_one_forest_match_their_solo_runs_on_the_all_ties_tree(
        monkeypatch):
    """Zero-init heads + zero rewards: every selection is an exact tie,
    value-free, so batch-shape wobble cannot move it and the multi-tree
    lockstep (slot-aligned waves across trees) must reproduce each
    tree's solo result exactly."""
    helper, engine = build_world(monkeypatch, dezero=False)
    trajs = [make_traj(20, 9, True, zero_rewards=True),
             make_traj(21, 7, False, zero_rewards=True)]
    idxs = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    solo = []
    for tr, ix in zip(trajs, idxs):
        np.random.seed(5)
        solo.append(engine.search_trajectories(
            [tr], [ix], num_simulations=9, c_puct=C_PUCT, discount=DISCOUNT,
            tie_break="first")[0])
    np.random.seed(5)
    both = engine.search_trajectories(
        trajs, idxs, num_simulations=9, c_puct=C_PUCT, discount=DISCOUNT,
        tie_break="first")
    for t in range(2):
        assert set(both[t]) == set(solo[t])
        for i in both[t]:
            assert np.allclose(both[t][i][0], solo[t][i][0], rtol=0, atol=1e-12)
            assert both[t][i][1] == pytest.approx(solo[t][i][1], rel=0,
                                                  abs=1e-12)


def test_two_trees_real_heads_budget_and_finiteness(monkeypatch):
    helper, engine = build_world(monkeypatch, noise=True)
    trajs = [make_traj(22, 9, True), make_traj(23, 12, False)]
    idxs = [list(range(8)), list(range(12))]
    monkeypatch.setattr(muzero, "COMMUNAL_WINDOW_ROOTS", 4)
    got = engine.search_trajectories(trajs, idxs, num_simulations=8,
                                     c_puct=C_PUCT, discount=DISCOUNT,
                                     tie_break="random",
                                     generator=torch.Generator().manual_seed(1))
    for t, ix in enumerate(idxs):
        assert set(got[t]) == set(ix)
        for i in ix:
            pol, val = got[t][i]
            assert np.isfinite(val) and np.isfinite(pol).all()
            assert pol.sum() == pytest.approx(1.0)


def test_fresh_root_search_is_inherited_unchanged(monkeypatch):
    """The communal engine is a VectorizedPUCT: its fresh-root search()
    is the v1 method itself (the fresh-root refresh path stays
    byte-identical when USE_COMMUNAL_REANALYSIS is off)."""
    assert VectorizedCommunalPUCT.search is muzero_vtree.VectorizedPUCT.search
    assert VectorizedCommunalPUCT.act is muzero_vtree.VectorizedPUCT.act


# --------------------------------------------------------------------------
# Integration: experiments/ablation.py's refresh under the communal flags
# --------------------------------------------------------------------------
def _ablation():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent / "experiments"))
    import ablation
    return ablation


def _make_run(ablation, replay, dezero_seed=None):
    from tests.test_paper_mode import ACTIONS as HA, make_trainer
    run = object.__new__(ablation.Run)
    run.trainer = make_trainer()
    run.trainer.training_steps = 500
    if dezero_seed is not None:
        dezero_heads({"prediction": run.trainer.prediction,
                      "dynamics": run.trainer.dynamics,
                      "value_prefix": run.trainer.value_prefix},
                     seed=dezero_seed)
    run.searcher = muzero.MCTSHelper(action_space=HA, device=CPU,
                                     network_ref=None, storage_ref=None)
    run.replay = replay
    return run


def _harness_replay(seed):
    from tests.test_paper_mode import ACTIONS as HA
    from tests.test_reanalyze_refresh import obs_traj
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    base = [obs_traj([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], [5.0] * 7),
            obs_traj([0.0, 1.0, 0.0, 0.0, 1.0], [2.0] * 5, truncated=True)]
    for traj in base:
        for t in traj:
            t["born_step"] = 0
            t["action"] = int(rng.integers(HA))
    return base


@pytest.fixture
def quiet_communal(monkeypatch):
    """The communal arm's refresh flags, noise ON from a shared numpy
    seed, deterministic tie-breaks on both engines, windows of 3 so a
    7-step trajectory spans several windows."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_COMMUNAL_REANALYSIS", True),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("REFRESH_ENTROPY_PRIORITY", False),
                     ("REFRESH_NO_SHARPEN_GUARD", False),
                     ("MCTS_BATCHES", 1),
                     ("ACT_SEARCH_SIMS", 10),
                     ("MAX_DEPTH", 8),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("DIRICHLET_ALPHA", 0.3),
                     ("COMMUNAL_WINDOW_ROOTS", 3),
                     ("OFFPOLICY_DECAY_STEPS", None)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    orig = VectorizedCommunalPUCT.search_trajectories

    def forced(self, *a, **kw):
        kw["tie_break"] = "first"
        return orig(self, *a, **kw)

    monkeypatch.setattr(VectorizedCommunalPUCT, "search_trajectories", forced)


def test_vtree_communal_refresh_matches_inline_communal_exactly(
        quiet_communal, monkeypatch):
    """refresh_targets_vtree under USE_COMMUNAL_REANALYSIS must equal the
    inline refresh_targets (-> _search_traj_communal -> the sequential
    run_communal_mcts) on every applied field: same candidates, same
    eligibility, same _apply_refresh -- so any divergence would be the
    search, and the engine gate above covers that."""
    import copy
    ablation = _ablation()
    base = _harness_replay(3)
    inline_replay = copy.deepcopy(base)
    run_a = _make_run(ablation, inline_replay, dezero_seed=21)
    monkeypatch.setattr(muzero, "REFRESH_VTREE", False)
    np.random.seed(11)
    run_a.refresh_targets(2)

    monkeypatch.setattr(muzero, "REFRESH_VTREE", True)
    vtree_replay = copy.deepcopy(base)
    run_b = _make_run(ablation, vtree_replay)
    run_b.trainer = run_a.trainer          # identical target model
    engine = run_b._vtree_engine()
    assert isinstance(engine, VectorizedCommunalPUCT)
    np.random.seed(11)
    run_b.refresh_targets_vtree(2)

    for ta, tb in zip(inline_replay, vtree_replay):
        assert len(ta) == len(tb)
        for a, b in zip(ta, tb):
            assert np.allclose(a["target_policy"], b["target_policy"],
                               rtol=0, atol=1e-12)
            assert a["root_value"] == pytest.approx(b["root_value"],
                                                    rel=0, abs=1e-12)
            assert a["priority"] == pytest.approx(b["priority"],
                                                  rel=0, abs=1e-12)
            assert a["refreshed_step"] == b["refreshed_step"] == 500
            assert a["has_policy_target"] == b["has_policy_target"]
    # The searches actually changed something (not a vacuous match).
    assert any(not np.allclose(a["target_policy"], b["target_policy"])
               for a, b in zip(base[0], vtree_replay[0]))


def test_guarded_gaps_route_through_the_communal_engine_exactly(
        quiet_communal, monkeypatch):
    """Audit F1 end to end: REFRESH_NO_SHARPEN_GUARD's refresh_guarded
    records make _eligible_indices GAPPED; the vectorized communal path
    must still equal the inline sequential one field by field."""
    import copy
    ablation = _ablation()
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    monkeypatch.setattr(muzero, "COMMUNAL_WINDOW_ROOTS", 16)
    from tests.test_reanalyze_refresh import obs_traj
    from tests.test_paper_mode import ACTIONS as HA
    rng = np.random.default_rng(71)
    np.random.seed(71)
    long = obs_traj([float(x) for x in rng.choice([-1.0, 0.0, 1.0], 24)],
                    [3.0] * 24)
    guarded = (1, 2, 5, 9, 10, 11, 12, 17, 20)
    for t in long:
        t["born_step"] = 0
        t["action"] = int(rng.integers(HA))
    for i in guarded:
        long[i]["refresh_guarded"] = True
    base = [long]
    inline_replay = copy.deepcopy(base)
    run_a = _make_run(ablation, inline_replay, dezero_seed=21)
    assert run_a._eligible_indices(inline_replay[0]) == \
        [i for i in range(24) if i not in guarded]
    monkeypatch.setattr(muzero, "REFRESH_VTREE", False)
    np.random.seed(21)
    run_a.refresh_targets(1)
    monkeypatch.setattr(muzero, "REFRESH_VTREE", True)
    vtree_replay = copy.deepcopy(base)
    run_b = _make_run(ablation, vtree_replay)
    run_b.trainer = run_a.trainer
    np.random.seed(21)
    run_b.refresh_targets_vtree(1)
    for a, b in zip(inline_replay[0], vtree_replay[0]):
        assert np.allclose(a["target_policy"], b["target_policy"],
                           rtol=0, atol=1e-12)
        assert a["root_value"] == pytest.approx(b["root_value"], rel=0,
                                                abs=1e-12)
        assert a["priority"] == pytest.approx(b["priority"], rel=0,
                                              abs=1e-12)
        assert a.get("refresh_guarded") == b.get("refresh_guarded")
        assert a.get("refresh_no_drop_strikes") == b.get("refresh_no_drop_strikes")


def test_sweep_and_quota_callers_take_the_communal_path(quiet_communal,
                                                        monkeypatch):
    """refresh_sweep (the async driver's contract sweep) and
    refresh_targets_vtree both go through _refresh_traj_vtree, so under
    the communal flag both produce the communal engine's targets."""
    import copy
    ablation = _ablation()
    monkeypatch.setattr(muzero, "REFRESH_VTREE", True)
    base = _harness_replay(4)
    quota_replay = copy.deepcopy(base)
    run_q = _make_run(ablation, quota_replay, dezero_seed=21)
    np.random.seed(12)
    run_q.refresh_targets_vtree(2)
    sweep_replay = copy.deepcopy(base)
    run_s = _make_run(ablation, sweep_replay)
    run_s.trainer = run_q.trainer
    np.random.seed(12)
    assert run_s.refresh_sweep(2) == 2
    for tq, ts in zip(quota_replay, sweep_replay):
        for a, b in zip(tq, ts):
            assert np.allclose(a["target_policy"], b["target_policy"],
                               rtol=0, atol=1e-12)
            assert a["root_value"] == pytest.approx(b["root_value"],
                                                    rel=0, abs=1e-12)
    stats = run_s._refresh_offload_stats
    assert stats["refresh_applied"] == 2
    # fake_paper_traj marks EVERY record policy-carrying (7 + 5 eligible);
    # the terminal root is searched, yields no policy, and is skipped.
    assert stats["refresh_steps_total"] == 7 + 5
    assert "target_policy" in sweep_replay[0][-1]
    assert np.allclose(sweep_replay[0][-1]["target_policy"],
                       base[0][-1]["target_policy"])


def test_batched_forest_callers_refresh_every_trajectory(quiet_communal,
                                                          monkeypatch):
    """VTREE_COMMUNAL_TRAJS=2: the sweep searches both candidates in one
    forest (lockstep windows) and applies each through _apply_refresh;
    batch shapes differ from the one-per-forest exact path, so this
    pins structure and accounting, not bitwise values."""
    ablation = _ablation()
    monkeypatch.setattr(muzero, "REFRESH_VTREE", True)
    monkeypatch.setattr(muzero, "VTREE_COMMUNAL_TRAJS", 2)
    replay = _harness_replay(5)
    old = [[t["target_policy"].copy() for t in traj] for traj in replay]
    run = _make_run(ablation, replay, dezero_seed=21)
    np.random.seed(13)
    assert run.refresh_sweep(2) == 2
    for traj, before in zip(replay, old):
        for t, b in zip(traj, before):
            assert t["refreshed_step"] == 500
            assert np.isfinite(t["root_value"])
            assert t["target_policy"].sum() == pytest.approx(1.0)
        assert any(not np.allclose(t["target_policy"], b)
                   for t, b in zip(traj, before))
    assert run._refresh_offload_stats["refresh_steps_total"] == 7 + 5
    # A trajectory with no eligible steps still rotates (stamp only).
    for t in replay[1]:
        t["has_policy_target"] = False
    run.trainer.training_steps = 600
    assert run.refresh_sweep(2) == 1
    assert all(t["refreshed_step"] == 600 for t in replay[1])


def test_vtree_engine_class_follows_the_communal_flag(monkeypatch):
    ablation = _ablation()
    run = _make_run(ablation, [])
    monkeypatch.setattr(muzero, "USE_COMMUNAL_REANALYSIS", False)
    assert type(run._vtree_engine()) is muzero_vtree.VectorizedPUCT
    monkeypatch.setattr(muzero, "USE_COMMUNAL_REANALYSIS", True)
    assert type(run._vtree_engine()) is VectorizedCommunalPUCT
