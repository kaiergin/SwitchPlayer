"""Generation 2 (Kai, 2026-10-27; CURRICULUM.md "Next experiment
(pre-registered) -- Generation 2: DS2 as the master, one distilled
student"): DS2 -- a P3 student -- is the master. Pins the generation-2 arm,
the prep's student-master spec (master_spec: the snapshot's spec, checked
against the arm's static keys) and its offline steps, student_arm's and
student_gate's student-master vocabulary -- and, slow, a real prep from a
DS2-like master and the generation-2 student's offline phase. CPU only."""
import json
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
from tests.test_gen1 import _boosted, _pin
from tests.test_prioract2 import restore_rngs  # noqa: F401

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
FS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-fresh")
G2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen2-distill")
ENV_ID = "MsPacmanNoFrameskip-v4"
TK_VOCAB = ablation.ARMS[TK_ARM]["TOKEN_VOCAB"]
# DS2-like: a student vocabulary of TK's 36; G2's: one of DS2's
DS2_DROP = ((0, 8), (0, 32), (5, 16), (1, 16))
G2_DROP = DS2_DROP + ((0, 16), (3, 8))


def _drop(drop):
    return tuple(t for t in TK_VOCAB if t not in drop)


DS2_VOCAB, G2_VOCAB = _drop(DS2_DROP), _drop(G2_DROP)


@pytest.fixture
def student(monkeypatch):
    """ARMS restored after a test applies --token-vocab."""
    for arm in (DS2_ARM, FS2_ARM, G2_ARM, G0_ARM, TK_ARM):
        monkeypatch.setitem(ablation.ARMS, arm, ablation.ARMS[arm])


def _student_spec(static, vocab=DS2_VOCAB, c=11.25):
    """`static` as ablation.student_arm leaves it (a DS2-like snapshot's
    spec)."""
    return {**static, "TOKEN_VOCAB": tuple(vocab),
            "_token_vocab_sha256": "a" * 64, "_vocab_set_sha256": "b" * 64,
            "REWARD_GROWTH_START": c, "REWARD_GROWTH_CEILING": c,
            "_student_ceiling": c}


def _doc_file(tmp_path, vocab=G2_VOCAB, master_vocab=DS2_VOCAB, c=21.75,
              set_path=None, master_arm=DS2_ARM, master_iterations=90):
    doc = {"vocab": [list(t) for t in vocab], "student_ceiling": c,
           "master_arm": master_arm, "master_iterations": master_iterations}
    if master_vocab is not None:
        doc["master_vocab"] = [list(t) for t in master_vocab]
    if set_path is not None:
        doc["set_sha256"] = ablation.file_sha256(set_path)
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(doc))
    return p


def _header(master_vocab=DS2_VOCAB, master_arm=DS2_ARM, c=21.75):
    return {"format": distill.FORMAT,
            "vocab": [list(t) for t in G2_VOCAB],
            "master_vocab": [list(t) for t in master_vocab],
            "master_snapshot": "m.pth", "master_arm": master_arm,
            "student_ceiling": c}


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------
def test_gen2_arm_is_ds2s_with_ds2_as_its_master():
    ds2, g2 = ablation.ARMS[DS2_ARM], ablation.ARMS[G2_ARM]
    d = {k for k in set(ds2) | set(g2) if ds2.get(k) != g2.get(k)}
    assert d == {"_master_arm", "_doc"}
    assert g2["_master_arm"] == DS2_ARM == ablation.GEN1_DISTILL_ARM
    assert g2["_master_iterations"] == 90
    assert g2["_student"] == "distill" and g2["_ceiling_from_prep"] is True
    assert (g2["SCHEDULE_OFFSET_STEPS"], g2["_distill_offline_steps"],
            g2["_distill_snapshot_every"]) == (20000, 20000, 4000)
    assert g2["PREFIX_BN_PER_STEP"] is True
    assert g2["TOKEN_VOCAB"] == TK_VOCAB          # until --token-vocab


# --------------------------------------------------------------------------
# student_arm / student_gate with a student master
# --------------------------------------------------------------------------
def test_student_master_vocab_only_for_a_student_master(tmp_path, student):
    doc = json.loads(_doc_file(tmp_path).read_text())
    assert ablation.student_master_vocab(
        ablation.ARMS[G2_ARM], doc, "p") == DS2_VOCAB
    # generation 1: the master G0 is no student -> None, whatever the doc
    for arm in (DS2_ARM, FS2_ARM):
        assert ablation.student_master_vocab(
            ablation.ARMS[arm], doc, "p") is None
    no = {k: v for k, v in doc.items() if k != "master_vocab"}
    with pytest.raises(SystemExit, match="master_vocab"):
        ablation.student_master_vocab(ablation.ARMS[G2_ARM], no, "p")
    # a master vocabulary that is no student vocabulary of TK's 36
    bad = {**doc, "master_vocab": [list(t) for t in TK_VOCAB
                                   if t != (2, 4)]}
    with pytest.raises(SystemExit, match="drops a 4-frame token"):
        ablation.student_master_vocab(ablation.ARMS[G2_ARM], bad, "p")


def test_student_arm_applies_a_student_masters_prep(tmp_path, student):
    ablation.student_arm(G2_ARM, _doc_file(tmp_path, c=21.75))
    spec = ablation.ARMS[G2_ARM]
    assert spec["TOKEN_VOCAB"] == G2_VOCAB
    assert spec["_master_vocab"] == DS2_VOCAB
    assert (spec["REWARD_GROWTH_START"], spec["REWARD_GROWTH_CEILING"],
            spec["_student_ceiling"]) == (21.75, 21.75, 21.75)
    with ablation.arm_config(G2_ARM), ablation.token_actions():
        assert ablation.ACTIONS == len(G2_VOCAB)


def test_student_arm_refuses_a_token_the_master_dropped(tmp_path, student):
    # a subset of TK's 36 that keeps a token DS2 no longer had
    vocab = _drop(G2_DROP[1:])
    assert DS2_DROP[0] in vocab
    with pytest.raises(SystemExit, match="outside the master's"):
        ablation.student_arm(G2_ARM, _doc_file(tmp_path, vocab=vocab))
    assert "_master_vocab" not in ablation.ARMS[G2_ARM]


@pytest.mark.parametrize("key,bad", [
    ("master_arm", G0_ARM), ("master_iterations", 40)])
def test_student_arm_refuses_another_masters_prep(key, bad, tmp_path,
                                                 student):
    with pytest.raises(SystemExit, match=key):
        ablation.student_arm(G2_ARM, _doc_file(tmp_path, **{key: bad}))


def test_generation_1_students_are_unchanged(tmp_path, student):
    # DS2 / FS2 never get a `_master_vocab` (their master played TK's 36)
    for arm in (DS2_ARM, FS2_ARM):
        ablation.student_arm(arm, _doc_file(
            tmp_path, vocab=DS2_VOCAB, master_vocab=None, master_arm=G0_ARM,
            c=11.25))
        assert "_master_vocab" not in ablation.ARMS[arm]
        assert ablation.ARMS[arm]["TOKEN_VOCAB"] == DS2_VOCAB


def test_gen2_gate(tmp_path, student):
    static = ablation.ARMS[G2_ARM]
    good = tmp_path / "set.pkl"
    distill.write_set(good, _header(), [])
    tk = tmp_path / "tk.pkl"
    distill.write_set(tk, _header(master_vocab=TK_VOCAB), [])
    g0 = tmp_path / "g0.pkl"
    distill.write_set(g0, _header(master_arm=G0_ARM), [])

    def gate(dset):
        ablation.ARMS[G2_ARM] = static          # one --token-vocab per run
        ablation.student_arm(G2_ARM, _doc_file(tmp_path, set_path=dset))
        with ablation.arm_config(G2_ARM) as spec:
            ablation.student_gate(spec, dset, True)

    gate(good)
    with pytest.raises(SystemExit, match="student master's"):
        gate(tk)
    with pytest.raises(SystemExit, match="master arm"):
        gate(g0)
    # generation 1's DS2 still requires TK's 36 as its set's master
    # vocabulary
    ds2_set = tmp_path / "ds2.pkl"
    distill.write_set(ds2_set, {**_header(master_vocab=DS2_VOCAB,
                                          master_arm=G0_ARM, c=11.25),
                                "vocab": [list(t) for t in DS2_VOCAB]}, [])
    ablation.student_arm(DS2_ARM, _doc_file(
        tmp_path, vocab=DS2_VOCAB, master_vocab=None, master_arm=G0_ARM,
        c=11.25, set_path=ds2_set))
    with ablation.arm_config(DS2_ARM) as spec:
        with pytest.raises(SystemExit, match="TK arm's"):
            ablation.student_gate(spec, ds2_set, True)


# --------------------------------------------------------------------------
# the prep's master spec
# --------------------------------------------------------------------------
def test_a_generation_0_master_runs_as_its_arm():
    blob = {"spec": {"anything": 1}}
    assert distill.master_spec(G0_ARM, blob, "m") is ablation.ARMS[G0_ARM]
    assert distill.master_offline_steps(ablation.ARMS[G0_ARM]) == 0


def test_a_student_master_runs_as_its_snapshots_spec():
    spec = _student_spec(ablation.ARMS[DS2_ARM])
    got = distill.master_spec(DS2_ARM, {"spec": spec}, "m")
    assert got == spec and got is not spec
    assert got["TOKEN_VOCAB"] == DS2_VOCAB
    assert distill.master_offline_steps(got) == 20000
    assert distill.master_offline_steps(ablation.ARMS[FS2_ARM]) == 0
    # a generation-2 student as a later master: its vocabulary is checked
    # against its own `_master_vocab`
    g2 = {**_student_spec(ablation.ARMS[G2_ARM], G2_VOCAB, 21.75),
          "_master_vocab": DS2_VOCAB}
    assert distill.master_spec(G2_ARM, {"spec": g2}, "m") == g2


@pytest.mark.parametrize("change,match", [
    ({"PREFIX_BN_PER_STEP": False}, "beyond the student keys"),
    ({"SCHEDULE_OFFSET_STEPS": 0}, "beyond the student keys"),
    ({"FOO": 1}, "beyond the student keys"),
    ({"REWARD_GROWTH_START": 1.0}, "constant _student_ceiling"),
    ({"_student_ceiling": None}, "constant _student_ceiling"),
    ({"_student_ceiling": float("nan"), "REWARD_GROWTH_START": float("nan"),
      "REWARD_GROWTH_CEILING": float("nan")}, "constant _student_ceiling"),
    ({"TOKEN_VOCAB": _drop(((2, 4),))}, "drops a 4-frame token"),
])
def test_a_student_masters_spec_is_checked(change, match):
    spec = {**_student_spec(ablation.ARMS[DS2_ARM]), **change}
    with pytest.raises(SystemExit, match=match):
        distill.master_spec(DS2_ARM, {"spec": spec}, "m")


def test_a_student_masters_spec_missing_a_key_is_refused():
    spec = _student_spec(ablation.ARMS[DS2_ARM])
    del spec["TOKEN_LONG_BIAS"]
    with pytest.raises(SystemExit, match="beyond the student keys"):
        distill.master_spec(DS2_ARM, {"spec": spec}, "m")


def test_an_old_student_arm_is_no_master():
    old = TK_ARM + "-student-distill"
    spec = _student_spec(ablation.ARMS[old], c=160.0)
    with pytest.raises(SystemExit, match="constant ceiling"):
        distill.master_spec(old, {"spec": spec}, "m")


def test_installed_restores_the_arm_even_on_failure():
    static = ablation.ARMS[DS2_ARM]
    with pytest.raises(RuntimeError):
        with distill.installed(DS2_ARM, {"x": 1}):
            assert ablation.ARMS[DS2_ARM] == {"x": 1}
            raise RuntimeError
    assert ablation.ARMS[DS2_ARM] is static


@pytest.mark.parametrize("steps", [180000, 220000, 199999])
def test_a_student_masters_final_step_counts_its_offline_steps(
        steps, tmp_path):
    # (20,000 + 90 x 2,000 passes: the slow prep below, at its own scale)
    blob = {"arm": DS2_ARM, "spec": _student_spec(ablation.ARMS[DS2_ARM]),
            "training_steps": steps, "iteration": 90,
            "acting_states": {"representation": {}}}
    m = tmp_path / "m.pth"
    torch.save(blob, m)
    with pytest.raises(SystemExit, match="after 20000 offline steps"):
        distill.prep(m, 90, tmp_path / "out")
    assert ablation.ARMS[DS2_ARM].get("_student_ceiling") is None


# --------------------------------------------------------------------------
# real Ms. Pac-Man (CPU): a prep from a DS2-like master, then G2's offline
# phase
# --------------------------------------------------------------------------
MASTER_ARM = "test-gen2-master"
OFFLINE = 4


@pytest.fixture
def prepped(monkeypatch, tmp_path, restore_rngs, student):  # noqa: F811
    """A DS2-like master (a distilled student at a constant ceiling 3.0
    over DS2_VOCAB, OFFLINE offline steps before 2 iterations; random nets,
    a 4-sim search for speed) and its prep over a few recorded games, with
    rewards past three units (_boosted)."""
    _pin(monkeypatch)
    monkeypatch.setattr(distill, "record", _boosted(distill.record))
    static = {**ablation.ARMS[DS2_ARM], "ACT_SEARCH_SIMS": 4,
              "SCHEDULE_OFFSET_STEPS": OFFLINE,
              "_distill_offline_steps": OFFLINE, "_distill_snapshot_every": 2}
    monkeypatch.setitem(ablation.ARMS, MASTER_ARM, static)
    spec = _student_spec(static, DS2_VOCAB, 3.0)
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with distill.installed(MASTER_ARM, spec), \
            ablation.arm_config(MASTER_ARM) as s, ablation.token_actions():
        torch.manual_seed(5)
        nets = ablation.build_networks(s)
    blob = {"arm": MASTER_ARM, "seed": 0, "spec": dict(spec),
            "training_steps": OFFLINE + 2 * 2000, "iteration": 2,
            **{n: nets[n].state_dict() for n in ablation.NET_NAMES},
            "acting_states": {n: nets[n].state_dict()
                              for n in ("representation", "prediction")}}
    master = tmp_path / "weights-iter002.pth"
    torch.save(blob, master)
    out = tmp_path / "prep"
    logs = []
    header = distill.prep(master, 2, out, base_steps=300, log=logs.append)
    assert ablation.ARMS[MASTER_ARM] is static        # restored
    return header, out, logs


@pytest.mark.slow
def test_prep_from_a_student_master(prepped):
    header, out, logs = prepped
    h, lives = distill.read_set(out / "set.pkl")
    doc = json.loads((out / "vocab.json").read_text())
    assert h["master_arm"] == MASTER_ARM
    assert tuple(map(tuple, h["master_vocab"])) == DS2_VOCAB
    assert h["master_training_steps"] == OFFLINE + 4000
    assert h["master_offline_steps"] == OFFLINE
    # the master's constant ceiling (its spec's, not the static arm's 1.0)
    assert h["master_ceiling"] == 3.0
    m = distill.reward_magnitudes(lives, 10.0)
    assert h["student_ceiling"] == distill.halved_ceiling(m, 3.0) > 3.0
    assert doc["student_ceiling"] == h["student_ceiling"]
    # its final temperature and no long-token bias: the schedules read its
    # online steps (4,000 of 4,000)
    assert h["acting_temperature"] == 0.125
    assert h["long_token_bias"] == 0.0
    vocab = tuple(map(tuple, h["vocab"]))
    assert ablation.validate_student_vocab(vocab, DS2_VOCAB) == vocab
    for life in lives:
        for rec in life["traj"]:
            assert len(rec[3]) == len(vocab)


@pytest.mark.slow
def test_g2_trains_offline_from_a_student_masters_prep(
        prepped, monkeypatch, tmp_path):
    header, out, _ = prepped
    monkeypatch.setitem(ablation.ARMS, G2_ARM, {
        **ablation.ARMS[G2_ARM], "_distill_offline_steps": 4,
        "_distill_snapshot_every": 2, "SCHEDULE_OFFSET_STEPS": 4,
        "_master_arm": MASTER_ARM, "_master_iterations": 2})
    ablation.student_arm(G2_ARM, out / "vocab.json")
    c2 = header["student_ceiling"]
    assert ablation.ARMS[G2_ARM]["_master_vocab"] == DS2_VOCAB
    results = tmp_path / "mspacman-gen2-distill-s0-test"
    results.mkdir()
    ablation.RESULTS, ablation.TAG = results, "-test"
    _, lives = distill.read_set(out / "set.pkl")
    with ablation.arm_config(G2_ARM) as spec, \
            ablation._run_length_globals(2, 3), ablation.token_actions():
        muzero.TRAINING_BATCHES = 8
        ablation.student_gate(spec, out / "set.pkl", True)
        run = ablation.Run(G2_ARM, 0, spec)
        try:
            ablation.distill_prefill(run, lives)
            assert run._reward_ceiling == c2
            for traj, life in zip(run.replay, lives):
                for t, rec in zip(traj, life["traj"]):
                    assert t["reward"] == muzero.grown_reward(rec[2], c2)
            ablation.distill_offline(run, G2_ARM, 0, spec, True)
            assert run.trainer.training_steps == 4
        finally:
            run.env.close()
    row0 = torch.load(results / f"snapshots-test-{G2_ARM}-seed0" /
                      "weights-iter000.pth", weights_only=False)
    assert row0["spec"]["REWARD_GROWTH_CEILING"] == c2
    assert tuple(row0["spec"]["TOKEN_VOCAB"]) == tuple(
        tuple(t) for t in header["vocab"])
    assert row0["spec"]["_master_vocab"] == DS2_VOCAB
    assert row0["spec"]["PREFIX_BN_PER_STEP"] is True
    # a generation-3 prep could take row 0's spec as a student master's
    g2 = distill.master_spec(G2_ARM, row0, "row0")
    assert g2["_student_ceiling"] == c2
    assert np.isfinite(c2)
