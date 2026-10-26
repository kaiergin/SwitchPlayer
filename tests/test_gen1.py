"""Test 2 of the per-generation clip (Kai, 2026-10-24; CURRICULUM.md "Next
experiment (pre-registered) -- Test 2: the first swap under the
per-generation clip"): G0 the master; two generation-1 students over the
prep's vocabulary at the prep's constant ceiling (the halved-clip rule);
the distilled one's offline phase excluded from every within-run schedule
(SCHEDULE_OFFSET_STEPS). Pins the schedule offset (off: bit for bit), the
halved-clip rule, the arms, student_arm's ceiling, the gates and the resume
recipe -- and, slow, a real prep from a constant-ceiling master and the
distilled student's offline phase under the offset. CPU only."""
import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import distill
import muzero
from tests.test_prioract2 import restore_rngs  # noqa: F401

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
FS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-fresh")
ENV_ID = "MsPacmanNoFrameskip-v4"
MASTER_VOCAB = ablation.ARMS[TK_ARM]["TOKEN_VOCAB"]


@pytest.fixture
def student(monkeypatch):
    """ARMS restored after a test applies --token-vocab."""
    for arm in (DS2_ARM, FS2_ARM, G0_ARM, TK_ARM):
        monkeypatch.setitem(ablation.ARMS, arm, ablation.ARMS[arm])


def _vocab(drop=((0, 8), (0, 32), (5, 16))):
    return tuple(t for t in MASTER_VOCAB if t not in drop)


def _vocab_file(tmp_path, ceiling=7.5, set_path=None, name="vocab.json",
                master_arm=G0_ARM, master_iterations=90, **extra):
    p = tmp_path / name
    doc = {"vocab": [list(t) for t in _vocab()], **extra}
    for k, v in (("master_arm", master_arm),
                 ("master_iterations", master_iterations)):
        if v is not None:
            doc[k] = v
    if ceiling is not None:
        doc["student_ceiling"] = ceiling
    if set_path is not None:
        doc["set_sha256"] = ablation.file_sha256(set_path)
    p.write_text(json.dumps(doc))
    return p


def _header(ceiling=7.5, master_arm=G0_ARM):
    return {"format": distill.FORMAT,
            "vocab": [list(t) for t in _vocab()],
            "master_vocab": [list(t) for t in MASTER_VOCAB],
            "master_snapshot": "m.pth", "master_arm": master_arm,
            "student_ceiling": ceiling}


# --------------------------------------------------------------------------
# the schedule offset
# --------------------------------------------------------------------------
def test_the_offset_is_off_by_default_and_then_bit_for_bit():
    assert muzero.SCHEDULE_OFFSET_STEPS == 0
    for s in (0, 1, 1999, 45000, 180000, 10 ** 7):
        assert muzero.schedule_steps(s) is s


SCHEDULES = ("per_beta", "visit_softmax_temperature",
             "prior_acting_temperature", "token_long_bias", "reward_ceiling")


@pytest.mark.parametrize("name", SCHEDULES)
def test_every_within_run_schedule_reads_the_online_steps(name, monkeypatch):
    """DS2 at online step s (training step s + 20,000) sees what FS2 sees
    at s; the offline steps themselves see step 0."""
    f = getattr(muzero, name)
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(90, 2000):
        fs2 = [f(s) for s in range(0, 180001, 500)]
        monkeypatch.setattr(muzero, "SCHEDULE_OFFSET_STEPS", 20000)
        ds2 = [f(s + 20000) for s in range(0, 180001, 500)]
        offline = [f(s) for s in range(0, 20001, 500)]
        assert ds2 == fs2
        assert offline == [fs2[0]] * len(offline)


def test_the_gen1_distill_arm_lands_on_fs2s_schedule_at_every_iteration():
    at = lambda it: (it - 1) * 2000                  # noqa: E731
    with ablation.arm_config(FS2_ARM), ablation._run_length_globals(90, 2000):
        fs2 = [(muzero.prior_acting_temperature(at(it)),
                muzero.token_long_bias(at(it)), muzero.per_beta(at(it)))
               for it in range(1, 91)]
        assert muzero.SCHEDULE_OFFSET_STEPS == 0
    with ablation.arm_config(DS2_ARM), ablation._run_length_globals(90, 2000):
        assert muzero.SCHEDULE_OFFSET_STEPS == 20000
        ds2 = [(muzero.prior_acting_temperature(20000 + at(it)),
                muzero.token_long_bias(20000 + at(it)),
                muzero.per_beta(20000 + at(it))) for it in range(1, 91)]
    assert ds2 == fs2
    # the stages TK's 90-iteration run has: 0.5 to iteration 45, the bias
    # gone from 24, PER beta 1.0 at the end
    assert fs2[44][0] == 0.5 and fs2[45][0] == 0.25 and fs2[68][0] == 0.125
    assert fs2[22][1] > 0.0 and fs2[23][1] == 0.0
    assert fs2[0][2] == pytest.approx(0.4)
    with ablation.arm_config(DS2_ARM), ablation._run_length_globals(90, 2000):
        assert muzero.per_beta(200000) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the halved-clip rule
# --------------------------------------------------------------------------
def test_clip_share_is_the_share_of_units_removed():
    m = np.array([1.0, 1.0, 5.0, 20.0, 0.0])
    assert distill.clip_share(m, 1.0) == pytest.approx(1 - 4.0 / 27.0)
    assert distill.clip_share(m, 20.0) == 0.0
    assert distill.clip_share(m, 5.0) == pytest.approx(15.0 / 27.0)
    assert distill.clip_share(np.zeros(3), 1.0) == 0.0
    assert distill.clip_share(np.array([]), 1.0) == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_halved_ceiling_is_the_smallest_ceiling_that_halves_the_share(seed):
    rng = np.random.default_rng(seed)
    # Ms. Pac-Man-like: mostly pellets, some power pellets, ghosts, fruit
    m = np.concatenate([np.ones(rng.integers(50, 800)),
                        np.full(rng.integers(0, 20), 5.0),
                        rng.choice([20.0, 40.0, 80.0, 160.0],
                                   size=rng.integers(1, 30)),
                        rng.uniform(10, 500, size=rng.integers(0, 5)),
                        np.zeros(rng.integers(0, 3000))])
    rng.shuffle(m)
    c0 = 1.0
    s0 = distill.clip_share(m, c0)
    c1 = distill.halved_ceiling(m, c0)
    assert c1 > c0 and math.isfinite(c1)
    assert distill.clip_share(m, c1) == pytest.approx(s0 / 2, abs=1e-9)
    assert distill.clip_share(m, c1 * (1 - 1e-6)) > s0 / 2      # smallest
    # it halves again from there (the next swap)
    c2 = distill.halved_ceiling(m, c1)
    assert c2 >= c1
    assert distill.clip_share(m, c2) == pytest.approx(
        distill.clip_share(m, c1) / 2, abs=1e-9)


def test_halved_ceiling_keeps_a_ceiling_that_clips_nothing():
    assert distill.halved_ceiling(np.ones(10), 1.0) == 1.0
    assert distill.halved_ceiling(np.zeros(10), 1.0) == 1.0
    assert distill.halved_ceiling(np.array([]), 3.0) == 3.0
    assert distill.halved_ceiling(np.array([2.0, 4.0]), 4.0) == 4.0
    # one big reward: the ceiling halves the removed part exactly
    m = np.array([1.0] * 10 + [101.0])
    c1 = distill.halved_ceiling(m, 1.0)
    assert distill.clip_share(m, c1) == pytest.approx(
        distill.clip_share(m, 1.0) / 2)
    assert c1 == pytest.approx(51.0)


def test_halved_ceiling_lands_on_a_duplicated_magnitude():
    # audit r1 A-L1: non-integer magnitudes whose exact solution is a
    # duplicated magnitude -- the old window test missed both neighbouring
    # segments and returned the largest magnitude (clip nothing)
    # (c0 puts the exact solution on b: kept(b) = (total + kept(c0)) / 2;
    # the old solver returned d here)
    a, b, d = 1.2534543953681516, 3.2871821987649565, 6.434119732102841
    c0 = 1.242119592045484
    m = np.array([a] * 28 + [b] * 2 + [d] * 4)
    c1 = distill.halved_ceiling(m, c0)
    assert c1 == pytest.approx(b, abs=1e-9)
    assert distill.clip_share(m, c1) <= distill.clip_share(m, c0) / 2 \
        + 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_halved_ceiling_never_falls_through(seed):
    # random duplicated non-integer magnitudes (a game-agnostic reuse):
    # always the smallest ceiling within float noise, never "clip nothing"
    rng = np.random.default_rng(100 + seed)
    for _ in range(300):
        vals = rng.uniform(0.1, 30.0, size=rng.integers(1, 6))
        m = np.repeat(vals, rng.integers(1, 40, size=len(vals)))
        c0 = float(rng.uniform(0.05, m.max()))
        s0 = distill.clip_share(m, c0)
        c1 = distill.halved_ceiling(m, c0)
        if s0 <= 0:
            assert c1 == c0
            continue
        assert c0 <= c1 <= m.max()
        assert distill.clip_share(m, c1) <= s0 / 2 + 1e-9
        if c1 > c0 * (1 + 1e-6):
            assert distill.clip_share(m, c1 * (1 - 1e-6)) > s0 / 2 - 1e-9


def test_reward_magnitudes_count_every_base_step_once():
    lives = [{"traj": [(None, 0, ()), (None, 1, (10.0, 0.0, 200.0)),
                       (None, 2, (-50.0,))]},
             {"traj": [(None, 0, ()), (None, 0, (1600.0,))]}]
    m = distill.reward_magnitudes(lives, 10.0)
    assert m.tolist() == [1.0, 0.0, 20.0, 5.0, 160.0]


# --------------------------------------------------------------------------
# the arms, student_arm's ceiling, the gates, the resume recipe
# --------------------------------------------------------------------------
def test_gen1_arm_deltas_from_g0():
    g0 = ablation.ARMS[G0_ARM]
    ds, fs = ablation.ARMS[DS2_ARM], ablation.ARMS[FS2_ARM]
    d = {k for k in set(g0) | set(ds) if g0.get(k) != ds.get(k)}
    assert d == {"SCHEDULE_OFFSET_STEPS", "_student", "_distill_offline_steps",
                 "_distill_snapshot_every", "_ceiling_from_prep",
                 "_master_arm", "_master_iterations", "_doc",
                 "PREFIX_BN_PER_STEP"}
    assert (ds["SCHEDULE_OFFSET_STEPS"], ds["_distill_offline_steps"],
            ds["_distill_snapshot_every"]) == (20000, 20000, 4000)
    assert ds["_student"] == "distill" and ds["_master_arm"] == G0_ARM
    d = {k for k in set(g0) | set(fs) if g0.get(k) != fs.get(k)}
    assert d == {"_student", "_ceiling_from_prep", "_master_arm",
                 "_master_iterations", "_doc", "PREFIX_BN_PER_STEP"}
    # the rebuild (CURRICULUM Test 2 "REBUILD WITH THE FIX"): PB's fix in
    # both students, so the pair still differs only in the distillation
    # step; the master stays an off arm (its snapshot has no step sets)
    assert ds["PREFIX_BN_PER_STEP"] is True and fs["PREFIX_BN_PER_STEP"] is True
    assert "PREFIX_BN_PER_STEP" not in g0 and not muzero.PREFIX_BN_PER_STEP
    d = {k for k in set(ds) | set(fs) if ds.get(k) != fs.get(k)}
    assert d == {"SCHEDULE_OFFSET_STEPS", "_student", "_distill_offline_steps",
                 "_distill_snapshot_every", "_doc"}
    assert fs["_student"] == "fresh" and fs["_master_arm"] == G0_ARM
    assert ds["_master_iterations"] == fs["_master_iterations"] == 90
    # TK's within-run schedules (not the old DS's end state), TK's tokens
    for spec in (ds, fs):
        assert spec["PRIOR_ACT_STAGED_TEMP"] == ablation.ARMS[TK_ARM][
            "PRIOR_ACT_STAGED_TEMP"] is True
        assert spec["TOKEN_LONG_BIAS"] == 0.5
        assert "PER_BETA" not in spec and "ACTING_TEMPERATURE" not in spec
        assert spec["TOKEN_VOCAB"] == MASTER_VOCAB     # until --token-vocab


@pytest.mark.parametrize("arm", [DS2_ARM, FS2_ARM])
def test_student_arm_applies_the_preps_ceiling(arm, tmp_path, student):
    ablation.student_arm(arm, _vocab_file(tmp_path, ceiling=7.5))
    spec = ablation.ARMS[arm]
    assert (spec["REWARD_GROWTH_START"], spec["REWARD_GROWTH_CEILING"],
            spec["_student_ceiling"]) == (7.5, 7.5, 7.5)
    assert spec["TOKEN_VOCAB"] == _vocab()
    with ablation.arm_config(arm), ablation._run_length_globals(90, 2000):
        assert {muzero.reward_ceiling(s) for s in range(0, 220001, 1000)} \
            == {7.5}
    # the Ray actors' overrides carry it (no underscore)
    over = {k: v for k, v in spec.items() if not k.startswith("_")}
    assert over["REWARD_GROWTH_CEILING"] == 7.5


@pytest.mark.parametrize("bad", [None, True, 0, -1.0, float("nan"),
                                 float("inf"), "8", 1e9])
def test_student_arm_refuses_a_missing_or_bad_ceiling(bad, tmp_path,
                                                      student):
    with pytest.raises(SystemExit, match="student_ceiling"):
        ablation.student_arm(DS2_ARM, _vocab_file(tmp_path, ceiling=bad))


@pytest.mark.parametrize("arm", [DS2_ARM, FS2_ARM])
@pytest.mark.parametrize("key,bad", [
    ("master_arm", TK_ARM), ("master_arm", None),
    ("master_iterations", 40), ("master_iterations", None)])
def test_student_arm_refuses_another_masters_prep(arm, key, bad, tmp_path,
                                                 student):
    # audit r1 A-L3: FS2 has no set header, so vocab.json's own master
    # fields are checked (both students)
    with pytest.raises(SystemExit, match=key):
        ablation.student_arm(arm, _vocab_file(tmp_path, **{key: bad}))
    assert "_student_ceiling" not in ablation.ARMS[arm]


def test_the_old_students_ignore_a_ceiling(tmp_path, student, monkeypatch):
    ds = TK_ARM + "-student-distill"
    monkeypatch.setitem(ablation.ARMS, ds, ablation.ARMS[ds])
    ablation.student_arm(ds, _vocab_file(tmp_path, ceiling=7.5))
    assert ablation.ARMS[ds]["REWARD_GROWTH_START"] == 160.0
    assert "_student_ceiling" not in ablation.ARMS[ds]


def test_gen1_gate(tmp_path, student):
    good = tmp_path / "set.pkl"
    distill.write_set(good, _header(), [])
    tk90 = tmp_path / "tk90.pkl"
    distill.write_set(tk90, _header(master_arm=TK_ARM), [])
    other_c = tmp_path / "other_c.pkl"
    distill.write_set(other_c, _header(ceiling=9.0), [])

    def gate(arm, dset, path_set=None, ceiling=7.5):
        ablation.student_arm(arm, _vocab_file(
            tmp_path, ceiling=ceiling, set_path=path_set or dset or good))
        with ablation.arm_config(arm) as spec:
            ablation.student_gate(spec, dset, True)

    gate(DS2_ARM, good)
    gate(FS2_ARM, None)
    with pytest.raises(SystemExit, match="master arm"):
        gate(DS2_ARM, tk90)
    with pytest.raises(SystemExit, match="student_ceiling"):
        gate(DS2_ARM, other_c)
    with pytest.raises(SystemExit, match="takes no --distill-set"):
        gate(FS2_ARM, good)
    # the arm's ceiling moved away from the prep's: refused (DS2 and FS2)
    for arm, dset in ((DS2_ARM, good), (FS2_ARM, None)):
        ablation.student_arm(arm, _vocab_file(tmp_path, set_path=good))
        with ablation.arm_config(arm) as spec:
            muzero.REWARD_GROWTH_CEILING = 8.0
            with pytest.raises(SystemExit, match="constant student_ceiling"):
                ablation.student_gate(spec, dset, True)


def test_gate_refuses_a_ceiling_never_applied(student):
    for arm in (DS2_ARM, FS2_ARM):
        spec = {**ablation.ARMS[arm], "_token_vocab_sha256": "x"}
        with ablation.arm_config(arm):
            with pytest.raises(SystemExit, match="constant student_ceiling"):
                ablation.student_gate(spec, None, True)


def test_the_offset_belongs_to_the_distilled_student_only(tmp_path, student,
                                                          monkeypatch):
    with ablation.arm_config(G0_ARM) as spec:
        ablation.student_gate(spec, None, True)
        # (G0's arm_config does not own the key: monkeypatch restores it)
        monkeypatch.setattr(muzero, "SCHEDULE_OFFSET_STEPS", 20000)
        with pytest.raises(SystemExit, match="SCHEDULE_OFFSET_STEPS"):
            ablation.student_gate(spec, None, True)
    monkeypatch.setattr(muzero, "SCHEDULE_OFFSET_STEPS", 0)
    good = tmp_path / "set.pkl"
    distill.write_set(good, _header(), [])
    ablation.student_arm(DS2_ARM, _vocab_file(tmp_path, set_path=good))
    applied = ablation.ARMS[DS2_ARM]
    monkeypatch.setitem(ablation.ARMS, DS2_ARM, {
        **applied, "SCHEDULE_OFFSET_STEPS": 4000})
    with ablation.arm_config(DS2_ARM) as spec:
        with pytest.raises(SystemExit, match="SCHEDULE_OFFSET_STEPS"):
            ablation.student_gate(spec, good, True)
    ablation.student_arm(FS2_ARM, _vocab_file(tmp_path))
    monkeypatch.setitem(ablation.ARMS, FS2_ARM, {
        **ablation.ARMS[FS2_ARM], "SCHEDULE_OFFSET_STEPS": 20000})
    with ablation.arm_config(FS2_ARM) as spec:
        with pytest.raises(SystemExit, match="SCHEDULE_OFFSET_STEPS"):
            ablation.student_gate(spec, None, True)


def test_the_resume_recipe_carries_the_ceiling(tmp_path, student):
    dset = tmp_path / "set.pkl"
    distill.write_set(dset, _header(), [])
    ablation.student_arm(DS2_ARM, _vocab_file(tmp_path, set_path=dset))
    with ablation.arm_config(DS2_ARM) as spec:
        extra = ablation.student_expect(spec, dset)
    assert extra == {"token_vocab": [list(t) for t in _vocab()],
                     "reward_ceiling": 7.5,
                     "distill_set_sha256": ablation.file_sha256(dset)}
    ablation.student_arm(FS2_ARM, _vocab_file(tmp_path))
    with ablation.arm_config(FS2_ARM) as spec:
        assert ablation.student_expect(spec, None) == {
            "token_vocab": [list(t) for t in _vocab()],
            "reward_ceiling": 7.5}


@pytest.mark.parametrize("arm", [DS2_ARM, FS2_ARM])
def test_each_student_builds_the_per_step_prefix_sets(arm):
    """The rebuild: a student's value-prefix head carries one set of
    running statistics per LSTM step (PB's K = 5); G0's, the master's, none
    -- so G0's snapshot still loads strictly under its own arm in the prep."""
    with ablation.arm_config(arm) as spec, ablation.token_actions():
        assert muzero.PREFIX_BN_PER_STEP is True
        assert muzero.prefix_bn_steps() == muzero.PREFIX_LSTM_HORIZON == 5
        vp = ablation.build_networks(spec)["value_prefix"]
    assert vp.per_step == 5
    assert vp.norm_out_step_mean.shape[0] == 5
    assert any(k.endswith("_step_mean") for k in vp.state_dict())
    with ablation.arm_config(G0_ARM) as spec, ablation.token_actions():
        assert muzero.PREFIX_BN_PER_STEP is False
        g0 = ablation.build_networks(spec)["value_prefix"]
    assert g0.per_step == 0
    assert not any("_step_" in k for k in g0.state_dict())
    assert muzero.PREFIX_BN_PER_STEP is False      # arm_config restored it


def test_ceiling_fits_supports_is_the_growth_gates_check():
    assert ablation.ceiling_fits_supports(1.0)
    assert ablation.ceiling_fits_supports(160.0)
    assert not ablation.ceiling_fits_supports(1e6)


# --------------------------------------------------------------------------
# real Ms. Pac-Man (CPU): a prep from a constant-ceiling master, then DS2's
# offline phase under the offset
# --------------------------------------------------------------------------
MASTER_ARM = "test-gen1-master"


def _pin(monkeypatch):
    for k in set(ablation.ARMS[DS2_ARM]) | {"TEMPERATURE_TOTAL_STEPS",
                                           "OFFPOLICY_DECAY_STEPS",
                                           "ACT_SEARCH_SIMS",
                                           "TRAINING_BATCHES"}:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP", "DEVICE",
              "RESULTS", "TAG", "INNER_FRAMES"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.DEVICE = torch.device("cpu")
    torch.set_num_threads(2)


def _boosted(record):
    """distill.record with rewards past one unit, as Ms. Pac-Man's power
    pellets, ghosts and fruit pay (random nets eat only pellets, so S(1)
    would be 0 and c1 = c0; audit r1 A-L2): every 4th rewarded base step
    pays 200, every 7th 50."""
    def wrapped(run, base_steps):
        lives, games, total = record(run, base_steps)
        k = 0
        for life in lives:
            recs = []
            for r in life["traj"]:
                rs = []
                for x in r[2]:
                    if x > 0:
                        k += 1
                        x = 200.0 if k % 4 == 0 else (
                            50.0 if k % 7 == 0 else x)
                    rs.append(x)
                recs.append((r[0], r[1], tuple(rs), *r[3:]))
            life["traj"] = recs
        return lives, games, total
    return wrapped


@pytest.fixture
def prepped(monkeypatch, tmp_path, restore_rngs, student):  # noqa: F811
    """A G0-like master (random nets, the constant sign clip, a 4-sim
    search for speed) and its prep over a few recorded games, with rewards
    past one unit (_boosted)."""
    _pin(monkeypatch)
    monkeypatch.setattr(distill, "record", _boosted(distill.record))
    monkeypatch.setitem(ablation.ARMS, MASTER_ARM,
                        {**ablation.ARMS[G0_ARM], "ACT_SEARCH_SIMS": 4})
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with ablation.arm_config(MASTER_ARM) as spec, ablation.token_actions():
        torch.manual_seed(5)
        nets = ablation.build_networks(spec)
    blob = {"arm": MASTER_ARM, "seed": 0, "spec": dict(spec),
            "training_steps": 4000, "iteration": 2,
            **{n: nets[n].state_dict() for n in ablation.NET_NAMES},
            "acting_states": {n: nets[n].state_dict()
                              for n in ("representation", "prediction")}}
    master = tmp_path / "weights-iter002.pth"
    torch.save(blob, master)
    out = tmp_path / "prep"
    logs = []
    header = distill.prep(master, 2, out, base_steps=300, log=logs.append)
    return header, out, logs


@pytest.mark.slow
def test_prep_writes_the_halved_ceiling(prepped):
    header, out, logs = prepped
    h, lives = distill.read_set(out / "set.pkl")
    doc = json.loads((out / "vocab.json").read_text())
    assert doc["student_ceiling"] == h["student_ceiling"]
    assert h["master_ceiling"] == 1.0 and h["reward_unit"] == 10.0
    m = distill.reward_magnitudes(lives, 10.0)
    assert h["reward_units"] == pytest.approx(m.sum(), abs=1e-3)
    assert h["clip_share_master"] == pytest.approx(
        distill.clip_share(m, 1.0), abs=1e-6)
    assert h["student_ceiling"] == distill.halved_ceiling(m, 1.0)
    # the boosted rewards make the sign clip remove units: c1 moves
    assert h["clip_share_master"] > 0
    assert h["student_ceiling"] > 1.0
    assert h["clip_share_student"] == pytest.approx(
        h["clip_share_master"] / 2, abs=1e-6)
    assert h["clipped_base_share_master"] == pytest.approx(
        float(np.mean(m > 1.0)), abs=1e-6)
    assert h["clipped_base_share_student"] == pytest.approx(
        float(np.mean(m > h["student_ceiling"])), abs=1e-6)
    assert 0 < h["clipped_base_share_master"] <= h[
        "clipped_step_share_master"]
    assert any("student clip" in line for line in logs)


@pytest.mark.slow
def test_ds2_trains_offline_at_the_preps_ceiling_under_the_offset(
        prepped, monkeypatch, tmp_path):
    header, out, _ = prepped
    monkeypatch.setitem(ablation.ARMS, DS2_ARM, {
        **ablation.ARMS[DS2_ARM], "_distill_offline_steps": 4,
        "_distill_snapshot_every": 2, "SCHEDULE_OFFSET_STEPS": 4,
        "_master_arm": MASTER_ARM, "_master_iterations": 2})
    ablation.student_arm(DS2_ARM, out / "vocab.json")
    c1 = header["student_ceiling"]
    assert c1 > 1.0                       # the clip is observable (_boosted)
    results = tmp_path / "mspacman-gen1-distill-s0-test"
    results.mkdir()
    ablation.RESULTS, ablation.TAG = results, "-test"
    _, lives = distill.read_set(out / "set.pkl")
    with ablation.arm_config(DS2_ARM) as spec, \
            ablation._run_length_globals(2, 3), ablation.token_actions():
        muzero.TRAINING_BATCHES = 8
        ablation.student_gate(spec, out / "set.pkl", True)
        run = ablation.Run(DS2_ARM, 0, spec)
        try:
            ablation.distill_prefill(run, lives)
            assert run._reward_ceiling == c1
            moved = 0
            for traj, life in zip(run.replay, lives):
                for t, rec in zip(traj, life["traj"]):
                    assert t["reward"] == muzero.grown_reward(rec[2], c1)
                    moved += t["reward"] != muzero.grown_reward(rec[2], 1.0)
                    # audit r1 A-M1: the prefill ages from the first online
                    # step (age 0 through the offline phase)
                    assert t["born_step"] == 4
            assert moved > 0                      # not G0's sign clip
            ablation.distill_offline(run, DS2_ARM, 0, spec, True)
            assert run.trainer.training_steps == 4
            # the schedules restart at the first online step
            assert muzero.prior_acting_temperature(4) == \
                muzero.prior_acting_temperature(0)
            assert muzero.per_beta(4) == pytest.approx(0.4)
        finally:
            run.env.close()
    row0 = torch.load(results / f"snapshots-test-{DS2_ARM}-seed0" /
                      "weights-iter000.pth", weights_only=False)
    assert row0["spec"]["REWARD_GROWTH_CEILING"] == c1
    assert row0["spec"]["SCHEDULE_OFFSET_STEPS"] == 4
    # the rebuild: the offline phase trains the per-step sets (each LSTM
    # step's batch statistics folded into its own set) and row 0 carries
    # them
    assert row0["spec"]["PREFIX_BN_PER_STEP"] is True
    vp = row0["value_prefix"]
    for name in ("norm_in", "norm_out"):
        count = vp[f"{name}_step_count"]
        assert count.shape[0] == 5 and bool((count > 0).all()), name
        assert not torch.equal(vp[f"{name}_step_mean"],
                               torch.zeros_like(vp[f"{name}_step_mean"]))
