"""Search-based value estimation (EfficientZero V2 SVE, stage 1: stored root
values) -- VALUE_TARGET_MODE "mixed" and the prioract-sve arm (2026-09-28).

The rule under test: from SVE_START_TRAIN_STEPS on, a sampled window ROOT
whose global buffer index lies before the newest SVE_FRESH_TRANSITIONS
transitions takes each real unroll position's stored root_value as that
position's value target; every other value (fresh roots, padding) is the
TD target exactly as in "td" mode.
"""
import asyncio
import copy
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_checkpoint_resume import make_run
from tests.test_paper_mode import ACTIONS, fake_paper_traj, make_trainer

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parents[1] / "experiments"))
import ablation

ARM = "effzero-repro-faithful-prioract-sve"
K = 2                                        # unroll steps in these tests


def test_arm_is_prioract_plus_exactly_the_sve_flags():
    base = ablation.ARMS["effzero-repro-faithful-prioract"]
    arm = ablation.ARMS[ARM]
    delta = {k for k in set(base) | set(arm) if base.get(k) != arm.get(k)}
    assert delta == {"VALUE_TARGET_MODE", "SVE_START_TRAIN_STEPS",
                     "SVE_FRESH_TRANSITIONS", "SVE_SOURCE", "SVE_SEARCH_SIMS",
                     "_doc"}
    assert (arm["VALUE_TARGET_MODE"], arm["SVE_START_TRAIN_STEPS"],
            arm["SVE_FRESH_TRANSITIONS"], arm["SVE_SOURCE"],
            arm["SVE_SEARCH_SIMS"]) == ("mixed", 40000, 20000, "search", 16)


def test_default_is_td():
    assert muzero.VALUE_TARGET_MODE == "td"
    assert muzero.SVE_SOURCE == "stored"      # stage-1 tests below use it


def _buffer():
    """Three terminated trajectories, 12 transitions, with root values that
    differ from every TD target so a swap is always visible."""
    a = fake_paper_traj([0.0, 1.0, 0.0, 1.0], [11.0, 12.0, 13.0, 0.0])
    b = fake_paper_traj([1.0, 0.0, 1.0, 0.0], [21.0, 22.0, 23.0, 0.0])
    c = fake_paper_traj([0.0, 0.0, 1.0, 1.0], [31.0, 32.0, 33.0, 0.0])
    for t in a + b + c:
        t["born_step"] = 0
    return [a, b, c]


def _batch(trainer, replay, mode, ts, fresh=None, start=0, seed=0):
    """Every legal root exactly once (batch = #roots, without replacement),
    uniform, USE_TARGET_MODEL off (TD from stored values: no net needed)."""
    n_roots = sum(len(t) - 1 for t in replay)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(muzero, "PER_ENABLED", False)
        mp.setattr(muzero, "USE_TARGET_MODEL", False)
        mp.setattr(muzero, "VALUE_TARGET_MODE", mode)
        mp.setattr(muzero, "SVE_START_TRAIN_STEPS", start)
        if fresh is not None:
            mp.setattr(muzero, "SVE_FRESH_TRANSITIONS", fresh)
        trainer.training_steps = ts
        np.random.seed(seed)
        b = trainer.prepare_batch_paper(replay, n_roots, K, ACTIONS)
        state = np.random.get_state()
    where = {id(t): (ti, p) for ti, traj in enumerate(replay)
             for p, t in enumerate(traj)}
    return b, [where[id(r)] for r in b["sample_refs"]], state


def _same_batch(x, y):
    for k in ("obs", "actions", "rewards", "target_policy", "target_value",
              "has_policy_target", "real_mask", "is_weights"):
        assert np.array_equal(x[k], y[k]), k


@pytest.mark.parametrize("case", ["before_T1", "all_fresh"])
def test_mixed_mode_without_stale_roots_is_the_td_batch_bitwise(case):
    """Nothing to swap -> the exact "td" batch, values, masks, weights, and
    the global RNG left in the same state."""
    trainer = make_trainer()
    replay = _buffer()
    td, roots_td, rng_td = _batch(trainer, replay, "td", ts=100)
    if case == "before_T1":
        mx, roots_mx, rng_mx = _batch(trainer, replay, "mixed", ts=100,
                                      fresh=1, start=101)
    else:
        mx, roots_mx, rng_mx = _batch(trainer, replay, "mixed", ts=100,
                                      fresh=12, start=0)
    _same_batch(td, mx)
    assert roots_td == roots_mx
    assert rng_td[0] == rng_mx[0] and np.array_equal(rng_td[1], rng_mx[1]) \
        and rng_td[2:] == rng_mx[2:]


# Hand-listed stale roots (trajectory, position). The buffer holds 9
# TRANSITIONS (3 per trajectory; a trajectory's last record is not one), at
# global indices a: 0-2, b: 3-5, c: 6-8; stale iff g <= 9 - T2 (EZ-V2).
STALE = {3: {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)},  # g<=6
         5: {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},                  # g<=4
         7: {(0, 0), (0, 1), (0, 2)},                                  # g<=2
         9: {(0, 0)},                                                  # g<=0
         10: set()}


@pytest.mark.parametrize("fresh", [3, 5, 7, 9, 10])
def test_stale_roots_take_stored_values_at_every_real_position(fresh):
    """Cutoffs inside c, inside b, at the a/b boundary, at g = 0 (T2 = N:
    EZ-V2 still marks index 0 stale) and past the start. The rule is per
    ROOT: a stale root's whole window uses stored values, even positions
    past the cutoff; fresh roots and padding keep TD."""
    trainer = make_trainer()
    replay = _buffer()
    td, roots_td, _ = _batch(trainer, replay, "td", ts=100)
    mx, roots, _ = _batch(trainer, replay, "mixed", ts=100, fresh=fresh)
    assert roots == roots_td
    assert {r for r in roots if r in STALE[fresh]} == STALE[fresh]  # all sampled
    for row, (ti, p) in enumerate(roots):
        stale = (ti, p) in STALE[fresh]
        for step in range(K + 1):
            pos = p + step
            if stale and pos < len(replay[ti]):
                assert mx["target_value"][row, step] == replay[ti][pos]["root_value"]
            else:
                assert mx["target_value"][row, step] == td["target_value"][row, step]
    # Everything except value targets is untouched.
    for k in ("obs", "actions", "rewards", "target_policy",
              "has_policy_target", "real_mask", "is_weights"):
        assert np.array_equal(td[k], mx[k]), k


def test_mixed_mode_uses_the_fixed_td_horizon(monkeypatch):
    """EZ-V2 zeroes delta_td for the mixed target (before T1 too): fresh
    roots' TD keeps the full TD_STEPS horizon where "td" mode would shrink
    it with data age."""
    monkeypatch.setattr(muzero, "TD_STEPS", 3)
    monkeypatch.setattr(muzero, "OFFPOLICY_DECAY_STEPS", 50)  # age 100 -> 1
    trainer = make_trainer()
    replay = _buffer()
    td, roots, _ = _batch(trainer, replay, "td", ts=100)
    mx, roots_mx, _ = _batch(trainer, replay, "mixed", ts=100, fresh=1,
                             start=10 ** 9)                 # before T1
    assert roots == roots_mx
    for row, (ti, p) in enumerate(roots):
        for step in range(K + 1):
            pos = p + step
            if pos >= len(replay[ti]):
                continue
            want_td = muzero.paper_value_target(replay[ti], pos, 1, muzero.DISCOUNT)
            want_mx = muzero.paper_value_target(replay[ti], pos, 3, muzero.DISCOUNT)
            assert td["target_value"][row, step] == pytest.approx(want_td)
            assert mx["target_value"][row, step] == pytest.approx(want_mx)
    assert not np.array_equal(td["target_value"], mx["target_value"])


def test_t1_gate_uses_the_batch_step_including_the_prefetch_path():
    trainer = make_trainer()
    replay = _buffer()
    before, _, _ = _batch(trainer, replay, "mixed", ts=499, fresh=4, start=500)
    at, roots, _ = _batch(trainer, replay, "mixed", ts=500, fresh=4, start=500)
    td, _, _ = _batch(trainer, replay, "td", ts=500)
    assert np.array_equal(before["target_value"], td["target_value"])
    assert not np.array_equal(at["target_value"], td["target_value"])
    # The prefetch worker passes training_steps explicitly (base + i): the
    # gate must read THAT, not the trainer's counter.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(muzero, "PER_ENABLED", False)
        mp.setattr(muzero, "USE_TARGET_MODEL", False)
        mp.setattr(muzero, "VALUE_TARGET_MODE", "mixed")
        mp.setattr(muzero, "SVE_START_TRAIN_STEPS", 500)
        mp.setattr(muzero, "SVE_FRESH_TRANSITIONS", 4)
        trainer.training_steps = 0
        np.random.seed(0)
        pre = trainer.finish_batch_paper(trainer.assemble_batch_paper(
            replay, 9, K, ACTIONS, training_steps=500))
    assert np.array_equal(pre["target_value"], at["target_value"])


def test_padding_keeps_the_td_value():
    """Past a trajectory's end nothing is swapped: absorbing 0 after a
    terminal, today's value for a short truncated tail."""
    trainer = make_trainer()
    replay = _buffer()
    replay[0][-1]["root_value"] = 7.0          # would leak if padding swapped
    mx, roots, _ = _batch(trainer, replay, "mixed", ts=100, fresh=1)
    td, _, _ = _batch(trainer, replay, "td", ts=100)
    for row, (ti, p) in enumerate(roots):
        for step in range(K + 1):
            if p + step >= len(replay[ti]):
                assert mx["target_value"][row, step] == td["target_value"][row, step]
    short = [fake_paper_traj([0.0, 1.0], [3.0, 44.0], truncated=True)]
    mx, roots, _ = _batch(trainer, short, "mixed", ts=100, fresh=0)
    td, _, _ = _batch(trainer, short, "td", ts=100)
    assert mx["target_value"][0, 0] == 3.0                       # real: SVE
    assert mx["target_value"][0, 1] == 44.0      # the truncated tail is real
    assert np.array_equal(mx["target_value"][0, 2:], td["target_value"][0, 2:])


def test_swap_follows_the_target_model_bootstrap_and_feeds_diagnostics():
    """USE_TARGET_MODEL on (the arm): the TD value is completed by the
    lagged target nets in finish_batch_paper, and only THEN replaced -- so
    the diagnostics' |SVE - TD| is against the exact target replaced."""
    torch.manual_seed(0)
    trainer = make_trainer()
    replay = _buffer()
    for traj in replay:
        for t in traj:
            t["obs"] = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                         dtype=np.uint8)
        traj[1]["root_value_step"] = 40                     # age 60 at ts 100
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(muzero, "PER_ENABLED", False)
        mp.setattr(muzero, "USE_TARGET_MODEL", True)
        mp.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
        mp.setattr(muzero, "SVE_FRESH_TRANSITIONS", 4)
        trainer.training_steps = 100
        out = {}
        for mode in ("td", "mixed"):
            mp.setattr(muzero, "VALUE_TARGET_MODE", mode)
            trainer._sve_window = None
            np.random.seed(0)
            out[mode] = trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    td, mx = out["td"], out["mixed"]
    where = {id(t): (ti, p) for ti, traj in enumerate(replay)
             for p, t in enumerate(traj)}
    roots = [where[id(r)] for r in mx["sample_refs"]]
    gap = bias = age = 0.0
    n_sve = n_real = 0
    for row, (ti, p) in enumerate(roots):
        for step in range(K + 1):
            pos = p + step
            if pos >= len(replay[ti]):
                continue
            n_real += 1
            if ti < 2:                                          # a, b stale
                sv = replay[ti][pos]["root_value"]
                assert mx["target_value"][row, step] == pytest.approx(sv)
                d = sv - td["target_value"][row, step]
                gap, bias, n_sve = gap + abs(d), bias + d, n_sve + 1
                age += 100 - replay[ti][pos].get("root_value_step", 0)
            else:
                assert mx["target_value"][row, step] == td["target_value"][row, step]
    w = trainer._sve_window
    assert (w["n_real"], w["n_sve"]) == (n_real, n_sve)
    assert w["gap"] == pytest.approx(gap, rel=1e-6)
    assert w["bias"] == pytest.approx(bias, rel=1e-6)
    assert w["age"] == pytest.approx(age)
    # The row path emits the means and restarts the window.
    run = make_run()
    run.trainer = trainer
    row = {}
    run._merge_row_stats(row)
    assert row["sve_frac"] == pytest.approx(n_sve / n_real, abs=1e-4)
    assert row["sve_td_gap"] == pytest.approx(gap / n_sve, abs=1e-4)
    assert row["sve_td_bias"] == pytest.approx(bias / n_sve, abs=1e-4)
    assert row["sve_value_age"] == pytest.approx(age / n_sve, abs=0.1)
    # 9 transitions, T2 = 4 -> g <= 5 stale: 6 of 9 (uniform buffer share).
    assert row["sve_buffer_frac"] == pytest.approx(6 / 9, abs=1e-4)
    assert trainer._sve_window is None


def test_before_t1_rows_report_zero_sve_and_td_rows_report_nothing():
    trainer = make_trainer()
    replay = _buffer()
    run = make_run()
    run.trainer = trainer
    _batch(trainer, replay, "mixed", ts=10, fresh=1, start=100)
    row = {}
    run._merge_row_stats(row)
    assert row["sve_frac"] == 0.0 and "sve_td_gap" not in row
    trainer._sve_window = None
    _batch(trainer, replay, "td", ts=10)
    row = {}
    run._merge_row_stats(row)
    assert not any(k.startswith("sve_") for k in row)


def _five_field_traj(n):
    rng = np.random.default_rng(0)
    return [(np.zeros((12, *muzero.IMAGE_RESHAPE), dtype=np.uint8), 0,
             float(i % 2), np.ones(ablation.ACTIONS) / ablation.ACTIONS,
             float(rng.normal())) for i in range(n)]


@pytest.mark.parametrize("mode", ["td", "mixed"])
def test_root_value_step_is_stamped_only_in_mixed_mode(monkeypatch, mode):
    monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", mode)
    run = make_run()
    out = run.paper_targets(_five_field_traj(5), True, born_step=1234)
    if mode == "mixed":
        assert all(t["root_value_step"] == 1234 for t in out)
    else:
        assert not any("root_value_step" in t for t in out)
    run._apply_refresh(out, {1: (np.ones(ablation.ACTIONS) / ablation.ACTIONS,
                                 5.0)}, now=2000)
    if mode == "mixed":
        assert out[1]["root_value_step"] == 2000
        assert out[0]["root_value_step"] == 1234      # not re-searched
    else:
        assert not any("root_value_step" in t for t in out)


def test_refresh_priorities_are_the_same_in_both_modes(monkeypatch):
    """Priorities stay TD-based under SVE (a stale position's SVE target is
    its own root value, which would zero its priority and starve it)."""
    prios = {}
    for mode in ("td", "mixed"):
        monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", mode)
        run = make_run()
        out = run.paper_targets(_five_field_traj(6), True, born_step=0)
        run._apply_refresh(out, {i: (np.ones(ablation.ACTIONS) / ablation.ACTIONS,
                                     0.5 * i) for i in range(5)}, now=100)
        prios[mode] = [t["priority"] for t in out]
        assert all(p > muzero.PER_EPS for p in prios[mode][:-1])
    assert prios["td"] == prios["mixed"]


def test_priority_writeback_uses_the_sve_target(monkeypatch):
    """train_step writes |v(s_0) - target_value[:, 0]| + eps. Two identical
    trainers see the same batch, one in "td" mode and one with every root
    stale: both priorities must be consistent with ONE shared v(s_0) -- the
    td run's against the TD target, the mixed run's against the stored
    root value."""
    monkeypatch.setattr(muzero, "PER_ENABLED", True)
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", False)
    monkeypatch.setattr(muzero, "USE_AUGMENTATION", False)
    monkeypatch.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
    monkeypatch.setattr(muzero, "SVE_FRESH_TRANSITIONS", 0)   # all stale
    rng = np.random.default_rng(3)
    obs = [rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)
           for _ in range(6)]
    got = {}
    for mode in ("td", "mixed"):
        monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", mode)
        torch.manual_seed(0)
        trainer = make_trainer()
        trainer.training_steps = 0
        traj = fake_paper_traj([0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                               [9.0, 8.0, 7.0, 6.0, 5.0, 0.0])
        for t, o in zip(traj, obs):
            t["obs"] = o
        np.random.seed(1)
        batch = trainer.prepare_batch_paper([traj], 4, muzero.UNROLL_STEPS,
                                            ACTIONS)
        torch.manual_seed(1)
        asyncio.run(trainer.train_step(batch))
        idx = [next(i for i, t in enumerate(traj) if t is r)
               for r in batch["sample_refs"]]
        got[mode] = (idx, batch["target_value"][:, 0].copy(),
                     [r["priority"] - muzero.PER_EPS
                      for r in batch["sample_refs"]])
    (i_td, z_td, p_td), (i_mx, z_mx, p_mx) = got["td"], got["mixed"]
    assert i_td == i_mx
    assert np.allclose(z_mx, [[9.0, 8.0, 7.0, 6.0, 5.0][i] for i in i_mx])
    assert not np.allclose(z_td, z_mx)
    for a, pa, b, pb in zip(z_td, p_td, z_mx, p_mx):
        v_td = (a - pa, a + pa)                  # v0 candidates (td run)
        v_mx = (b - pb, b + pb)                  # v0 candidates (mixed run)
        assert min(abs(x - y) for x in v_td for y in v_mx) < 1e-4



def test_diagnostics_panel_is_td_referenced_and_leaves_the_window_alone(monkeypatch):
    """SVE audit r1 M1: evaluate_diagnostics' batches must be built with TD
    targets (value / value_control comparable with the td arms) and must
    not enter the per-row SVE window."""
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)    # the SVE builder
    monkeypatch.setattr(muzero, "PER_ENABLED", True)
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", True)
    monkeypatch.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
    monkeypatch.setattr(muzero, "SVE_FRESH_TRANSITIONS", 12)
    rng = np.random.default_rng(0)
    base = []
    for k in range(3):
        tr = fake_paper_traj([float(rng.integers(0, 2)) for _ in range(12)],
                             [float(rng.normal()) * 3 for _ in range(11)] + [0.0])
        for t in tr:
            t["obs"] = rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                    dtype=np.uint8)
        base.append(tr)
    rows = {}
    for mode in ("td", "mixed"):
        monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", mode)
        torch.manual_seed(0)
        run = make_run()
        run.replay = copy.deepcopy(base)
        run.trainer.training_steps = 100
        np.random.seed(0)
        rows[mode] = run.evaluate_diagnostics(batches=2)
        assert getattr(run.trainer, "_sve_window", None) is None
        assert muzero.VALUE_TARGET_MODE == mode          # restored
    for k in ("value", "value_control"):
        assert rows["td"][k] == rows["mixed"][k], k
    assert not any(k.startswith("sve_") for k in rows["mixed"])
    # Non-vacuous: the same setup's TRAINING batch does swap.
    monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", "mixed")
    run.trainer._sve_window = None
    run.trainer.prepare_batch_paper(run.replay, 32, muzero.UNROLL_STEPS,
                                    ACTIONS)
    assert run.trainer._sve_window["n_sve"] > 0


@pytest.mark.parametrize("key,value,pattern", [
    ("VALUE_TARGET_MODE", "Mixed", "must be 'td' or 'mixed'"),
    ("PAPER_TARGETS", False, "requires PAPER_TARGETS"),
])
def test_run_one_refuses_an_invalid_sve_setting(monkeypatch, key, value, pattern):
    """L5: a typo must not silently run TD; the gate fires in run_one before
    any file work (out=None: nothing can be touched)."""
    spec = dict(ablation.ARMS[ARM])
    spec[key] = value
    if key == "PAPER_TARGETS":
        spec["BIRTH_SEARCH"] = False          # isolate the SVE gate
        spec["USE_SEARCH_ACTING"] = True
    monkeypatch.setitem(ablation.ARMS, "sve-variant", spec)
    with pytest.raises(SystemExit, match=pattern):
        ablation.run_one("sve-variant", 0, 1, 1, 1, 1, None)



def test_truncated_tail_is_swapped_and_its_padding_keeps_the_target_net_value():
    """Round-2 LOW-A: with the target model on (the arm), a stale window
    over a short TRUNCATED trajectory swaps the real tail record to its
    stored value, while the padding past it keeps the target-net bootstrap
    of the tail -- both differ from what the other rule would give."""
    torch.manual_seed(0)
    trainer = make_trainer()
    rng = np.random.default_rng(1)
    short = [fake_paper_traj([0.0, 1.0], [3.0, 44.0], truncated=True)]
    for t in short[0]:
        t["obs"] = rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                dtype=np.uint8)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(muzero, "PER_ENABLED", False)
        mp.setattr(muzero, "USE_TARGET_MODEL", True)
        mp.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
        mp.setattr(muzero, "SVE_FRESH_TRANSITIONS", 1)        # g=0 stale
        trainer.training_steps = 100
        for mode in ("td", "mixed"):
            mp.setattr(muzero, "VALUE_TARGET_MODE", mode)
            np.random.seed(0)
            out[mode] = trainer.prepare_batch_paper(short, 1, K, ACTIONS)
    td, mx = out["td"]["target_value"][0], out["mixed"]["target_value"][0]
    assert mx[0] == 3.0 and mx[1] == 44.0                    # real: stored
    assert td[1] != pytest.approx(44.0, abs=1e-3)             # TD tail differs
    assert np.array_equal(mx[2:], td[2:])                     # padding: TD
    assert not np.any(np.isclose(mx[2:], 44.0, atol=1e-3))


def test_sve_gate_rejects_an_empty_fresh_window(monkeypatch):
    spec = dict(ablation.ARMS[ARM])
    spec["SVE_FRESH_TRANSITIONS"] = 0
    monkeypatch.setitem(ablation.ARMS, "sve-variant", spec)
    with pytest.raises(SystemExit, match="SVE_FRESH_TRANSITIONS >= 1"):
        ablation.run_one("sve-variant", 0, 1, 1, 1, 1, None)



# --------------------------------------------------------------------------
# Stage 2: SVE_SOURCE "search" (fresh target-net search at batch time)
# --------------------------------------------------------------------------
V0_OFFSET = 1000.0          # the spy's root network value sits this far above q


class _SpyEngine:
    """Stands in for muzero_vtree.VectorizedPUCT: q = obs mean, the root's
    own network value = q + V0_OFFSET, n = num_simulations root visits, so
    the N+1 root mean is q + V0_OFFSET / (n + 1). Records every call."""
    made, calls = [], []

    def __init__(self, nets, action_space, device):
        _SpyEngine.made.append((nets, action_space, device))

    @staticmethod
    def _q(obs):
        return np.asarray(obs, dtype=np.float64).reshape(len(obs), -1).mean(1)

    def search(self, obs, **kw):
        import time as _t
        _t.sleep(0.002)
        _SpyEngine.calls.append((len(obs), kw))
        counts = np.zeros((len(obs), ACTIONS))
        counts[:, 0] = kw["num_simulations"]
        return None, self._q(obs), counts

    def _initial(self, obs):
        return None, None, torch.tensor(self._q(obs) + V0_OFFSET)


def _search_setup(monkeypatch, fresh=4, replay=None):
    import muzero_vtree
    _SpyEngine.made, _SpyEngine.calls = [], []
    monkeypatch.setattr(muzero_vtree, "VectorizedPUCT", _SpyEngine)
    monkeypatch.setattr(muzero, "PER_ENABLED", False)
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", False)
    monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", "mixed")
    monkeypatch.setattr(muzero, "SVE_SOURCE", "search")
    monkeypatch.setattr(muzero, "SVE_SEARCH_SIMS", 16)
    monkeypatch.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
    monkeypatch.setattr(muzero, "SVE_FRESH_TRANSITIONS", fresh)
    replay = replay or _buffer()
    for ti, traj in enumerate(replay):
        for p, t in enumerate(traj):
            t["obs"] = np.full((2,), 10 * ti + p, dtype=np.uint8)
    trainer = make_trainer()
    trainer.training_steps = 100
    trainer._ensure_target_model()
    trainer._target_synced_at = 60          # value age = 100 - 60 = 40
    return trainer, replay


def _spy_value(ti, pos):
    return 10 * ti + pos + V0_OFFSET / 17.0      # n = 16 root visits


def test_search_source_takes_fresh_target_net_values(monkeypatch):
    """Stale real positions get the N+1 root mean of a search of their own
    obs; the terminal record is 0 and never searched; fresh roots keep TD;
    the engine runs on the TARGET nets with the configured sims, no noise,
    deterministic tie-breaks, in VTREE_REFRESH_BATCH chunks; the value age
    is steps since the target sync."""
    monkeypatch.setattr(muzero, "VTREE_REFRESH_BATCH", 2)
    trainer, replay = _search_setup(monkeypatch)
    td, roots_td, _ = _batch(trainer, replay, "td", ts=100)
    trainer._sve_window = None
    np.random.seed(0)
    mx = trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    where = {id(t): (ti, p) for ti, traj in enumerate(replay)
             for p, t in enumerate(traj)}
    roots = [where[id(r)] for r in mx["sample_refs"]]
    assert roots == roots_td
    for row, (ti, p) in enumerate(roots):
        stale = ti < 2                                   # g <= 5 (T2 = 4)
        for step in range(K + 1):
            pos = p + step
            got = mx["target_value"][row, step]
            if stale and pos < len(replay[ti]):
                want = 0.0 if pos == len(replay[ti]) - 1 else _spy_value(ti, pos)
                assert got == pytest.approx(want, abs=1e-9), (ti, p, step)
            else:
                assert got == td["target_value"][row, step]
    nets, A, dev = _SpyEngine.made[0]
    assert nets["representation"] is trainer.target_representation
    assert nets["dynamics"] is trainer.target_dynamics
    assert nets["prediction"] is trainer.target_prediction
    assert nets["value_prefix"] is trainer.target_value_prefix
    for n, kw in _SpyEngine.calls:
        assert n <= 2                                    # chunked
        assert kw["num_simulations"] == 16 and kw["root_noise"] is False
        assert kw["tie_break"] == "first"
        assert kw["c_puct"] == muzero.C_PUCT and kw["discount"] == muzero.DISCOUNT
    # every distinct stale non-terminal position, once: a0-2, b0-2
    assert sum(n for n, _ in _SpyEngine.calls) == 6
    assert len(_SpyEngine.calls) == 3
    w = trainer._sve_window
    assert w["age"] / w["n_sve"] == pytest.approx(40.0)
    assert w["search_s"] > 0


def test_search_source_searches_a_truncated_tail(monkeypatch):
    """A truncated tail is a genuine state: searched, not zeroed (EZ-V2
    would give it 0 -- a documented deviation)."""
    trunc = fake_paper_traj([0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0],
                            truncated=True)
    c = fake_paper_traj([0.0, 0.0, 1.0, 1.0], [31.0, 32.0, 33.0, 0.0])
    trainer, replay = _search_setup(monkeypatch, fresh=3, replay=[trunc, c])
    np.random.seed(0)
    mx = trainer.prepare_batch_paper(replay, 5, K, ACTIONS)
    where = {id(t): (ti, p) for ti, traj in enumerate(replay)
             for p, t in enumerate(traj)}
    tails = [row for row, r in enumerate(mx["sample_refs"])
             if where[id(r)] == (0, 1)]            # root 1: window 1..3
    assert tails
    for row in tails:
        assert mx["target_value"][row, 2] == pytest.approx(_spy_value(0, 3))


def test_search_cache_lives_until_the_next_sync_or_block(monkeypatch):
    trainer, replay = _search_setup(monkeypatch)
    trainer._sve_window = None
    np.random.seed(0)
    trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    first = sum(n for n, _ in _SpyEngine.calls)
    np.random.seed(1)
    trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    assert sum(n for n, _ in _SpyEngine.calls) == first      # all cached
    trainer._target_synced_at = 80
    np.random.seed(2)
    trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    assert sum(n for n, _ in _SpyEngine.calls) == 2 * first  # new sync
    trainer.invalidate_bootstrap_cache()
    assert trainer._sve_cache is None
    w = trainer._sve_window
    assert w["searches"] == 2 * first and w["search_s"] > 0
    run = make_run()
    run.trainer = trainer
    row = {}
    run._merge_row_stats(row)
    assert row["sve_searches"] == 2 * first and row["sve_search_s"] >= 0


def test_search_source_real_engine_at_the_arms_tree_config(monkeypatch):
    """The real engine, heads de-zeroed (fresh heads return one constant for
    every state), online nets perturbed away from the target nets, the
    arm's tree config (value prefix in the tree, PER, target model): each
    swapped value equals EZ-V2's N+1 root mean of a direct TARGET-net
    search of THAT position, values differ across positions and from an
    online-net search, and the batch draws nothing from torch's global
    RNG."""
    import muzero_vtree
    from tests.test_vtree import dezero_heads
    monkeypatch.setattr(muzero, "PER_ENABLED", True)
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", True)
    monkeypatch.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", True)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)   # the arm's PUCT
    monkeypatch.setattr(muzero, "VALUE_TARGET_MODE", "mixed")
    monkeypatch.setattr(muzero, "SVE_SOURCE", "search")
    monkeypatch.setattr(muzero, "SVE_SEARCH_SIMS", 8)
    monkeypatch.setattr(muzero, "SVE_START_TRAIN_STEPS", 0)
    monkeypatch.setattr(muzero, "SVE_FRESH_TRANSITIONS", 4)
    torch.manual_seed(0)
    trainer = make_trainer()
    online = {"prediction": trainer.prediction, "dynamics": trainer.dynamics,
              "value_prefix": trainer.value_prefix}
    dezero_heads(online, seed=5)
    trainer.training_steps = 100
    trainer._ensure_target_model()            # target := these weights
    with torch.no_grad():                     # online now differs
        for prm in trainer.prediction.parameters():
            prm.add_(0.3 * torch.randn_like(prm))
    rng = np.random.default_rng(0)
    replay = _buffer()
    for traj in replay:
        for t in traj:
            t["obs"] = rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                    dtype=np.uint8)
    np.random.seed(0)
    t_rng = torch.get_rng_state()
    mx = trainer.prepare_batch_paper(replay, 9, K, ACTIONS)
    assert torch.equal(t_rng, torch.get_rng_state())

    def n1_mean(nets, obs):
        eng = muzero_vtree.VectorizedPUCT(nets, ACTIONS, trainer.device)
        _, q, counts = eng.search(obs, num_simulations=8, c_puct=muzero.C_PUCT,
                                  discount=muzero.DISCOUNT, root_noise=False,
                                  tie_break="first")
        _, _, v0 = eng._initial(obs)
        n = counts.sum(axis=1)
        return (v0.cpu().numpy() + n * q) / (n + 1)

    target = {"representation": trainer.target_representation,
              "dynamics": trainer.target_dynamics,
              "prediction": trainer.target_prediction,
              "value_prefix": trainer.target_value_prefix}
    live = {"representation": trainer.representation,
            "dynamics": trainer.dynamics, "prediction": trainer.prediction,
            "value_prefix": trainer.value_prefix}
    stale_pos = [(ti, p) for ti in (0, 1) for p in range(3)]
    obs = np.stack([replay[ti][p]["obs"] for ti, p in stale_pos])
    for net in live.values():
        net.eval()
    want = dict(zip(stale_pos, n1_mean(target, obs)))
    other = dict(zip(stale_pos, n1_mean(live, obs)))
    spread = max(want.values()) - min(want.values())
    tol = 1e-4 * spread + 1e-6
    assert spread > 1e-3                                   # non-degenerate
    assert max(abs(want[k] - other[k]) for k in want) > 100 * tol
    where = {id(t): (ti, p) for ti, traj in enumerate(replay)
             for p, t in enumerate(traj)}
    checked = 0
    for row, r in enumerate(mx["sample_refs"]):
        ti, p = where[id(r)]
        for step in range(K + 1):
            if (ti, p + step) in want and ti < 2:
                assert abs(mx["target_value"][row, step] - want[(ti, p + step)]) <= tol
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("key,value,pattern", [
    ("SVE_SOURCE", "fresh", "SVE_SOURCE must be"),
    ("SVE_SEARCH_SIMS", 0, "SVE_SEARCH_SIMS >= 1"),
    ("USE_GUMBEL_SEARCH", True, "PUCT"),
    ("USE_TARGET_MODEL", False, "requires USE_TARGET_MODEL"),
])
def test_sve_gate_validates_the_source(monkeypatch, key, value, pattern):
    spec = dict(ablation.ARMS[ARM])
    spec[key] = value
    if key == "USE_GUMBEL_SEARCH":
        spec["BIRTH_SEARCH"] = False          # isolate the SVE gate
        spec["USE_SEARCH_ACTING"] = True
    monkeypatch.setitem(ablation.ARMS, "sve-variant", spec)
    with pytest.raises(SystemExit, match=pattern):
        ablation.run_one("sve-variant", 0, 1, 1, 1, 1, None)
