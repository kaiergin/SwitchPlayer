"""Test 2 of the per-generation clip (Kai, 2026-10-24; CURRICULUM.md "Next
experiment (pre-registered) -- Test 2: the first swap under the
per-generation clip"): supervise_gen1.py runs DS2 then FS2 after G0, with
the distillation step's machinery. Pins the recipe against the distillation
step's, the constants naming the live G0 run, what ds2 waits for (G0's
marker and evals) and fs2 waits for (DS2's marker, its rows 0 and 5-90, its
offline curve, the prep), the FIXED master and its stops (NEAR RANDOM,
ABANDONED), the prep's once-only run and its retries, the guard / manifest
/ prep-integrity order before each launch, the child's environment, the
stray-state rules (DS2's own pre-checkpoint snapshots excepted), the
per-variant singleton, NO PROGRESS and the row alerts at the prep's
ceiling. CPU only, no launches.

No test reads LIVE state (the distillation step's audit r5 HIGH): the `sup`
fixture points every path into the live G0 run, this tree's results /
probes / prep (REPO) and the process table at empty stand-ins; tests that
need a G0 or run state build it in tmp. Only the constants tests read the
module as deployed (`real_sup`), and only its names."""
import hashlib
import inspect
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402

G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
FS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-fresh")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISORS = ROOT / "benchmarks" / "supervisors"
ENV_ID = "MsPacmanNoFrameskip-v4"
NL = chr(10)


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


def _bounded(effect, n=5):
    """A time.sleep stand-in that runs `effect` and records each call --
    and raises Stop past `n` calls, so a broken wait fails instead of
    looping forever."""
    calls = []

    def sleep(s):
        calls.append(s)
        if len(calls) > n:
            raise Stop("the wait never returned")
        effect(len(calls))

    sleep.calls = calls
    return sleep


@pytest.fixture
def real_sup():
    """The module as deployed: its constants name the LIVE G0 run and this
    tree. Only the constants tests read them -- as names, never the files
    behind them."""
    return _load(SUPERVISORS / "supervise_gen1.py", "supervise_gen1")


@pytest.fixture
def sup(real_sup, monkeypatch, tmp_path):
    """The module with every path into the live G0 run redirected to an
    empty temp folder (G0 "unfinished, not abandoned"), REPO (the run
    folders, probe jsonls and the prep) an empty temp tree, the manifest
    absent, the process table empty, host memory unknown and the real
    preflight / vocabulary check / death snapshot disarmed: no test may
    depend on the real G0's state or on this tree's own run state -- the
    preflights run this suite while G0 trains, after it completed or was
    abandoned, and in the run worktree beside DS2's live files. Tests that
    need a state build it in tmp (fake_g0)."""
    live = tmp_path / "_no_live_runs" / "g0"
    monkeypatch.setattr(real_sup, "G0_TREE", str(live.parent))
    monkeypatch.setattr(real_sup, "G0_BASE", str(live))
    for name, rel in (("G0_MARKER", "resume.ckpt"),
                      ("G0_ABANDONED", "ABANDONED"),
                      ("G0_SUPLOG", "supervisor.log"),
                      ("G0_SNAPSHOT", "weights-iter090.pth")):
        monkeypatch.setattr(real_sup, name, str(live / rel))
    monkeypatch.setattr(real_sup, "G0_PROBES", (
        str(live / "p-acting.jsonl"), str(live / "p.jsonl")))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(real_sup, "REPO", str(repo))
    monkeypatch.setattr(real_sup, "MANIFEST", str(tmp_path / "no-m.json"))
    monkeypatch.setattr(real_sup, "_procs", lambda: iter(()))
    monkeypatch.setattr(real_sup, "commit_free_gb", lambda: None)

    def disarmed(*a, **k):
        raise AssertionError("a test reached the real preflight / vocab "
                             "check without stubbing it")

    monkeypatch.setattr(real_sup, "preflight", disarmed)
    monkeypatch.setattr(real_sup, "vocab_is_ds", disarmed)
    monkeypatch.setattr(real_sup, "snapshot", lambda *a: None)
    return real_sup


@pytest.fixture
def g0sup():
    return _load(SUPERVISORS / "supervise_gen0.py", "supervise_gen0")


@pytest.fixture
def dsup():
    return _load(SUPERVISORS / "supervise_distill.py", "supervise_distill")


def _rows(path, iterations, value=700.0, extra=()):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for i in iterations:
            f.write(json.dumps({"iteration": i, "env": ENV_ID,
                                "exact_mean": value}) + NL)
        for r in extra:
            f.write(json.dumps(r) + NL)


@pytest.fixture
def fake_g0(sup):
    """A G0 state built in tmp: (marker, (acting jsonl, raw jsonl))."""
    marker = pathlib.Path(sup.G0_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker, tuple(pathlib.Path(p) for p in sup.G0_PROBES)


def _complete_g0(sup, fake_g0, value=700.0):
    marker, probes = fake_g0
    marker.write_bytes(b"done")
    _rows(probes[0], sup.G0_ROWS, value)
    _rows(probes[1], sup.G0_ROWS, value)
    pathlib.Path(sup.G0_SNAPSHOT).write_bytes(b"x")


# --------------------------------------------------------------------------
# the recipe, the constants
# --------------------------------------------------------------------------
def test_recipe_is_the_distillation_steps_with_the_gen1_arms(real_sup,
                                                              dsup):
    sup = real_sup
    assert sup.VARIANTS == {"ds2": DS2_ARM, "fs2": FS2_ARM}
    assert sup.RUNS == {"ds2": "mspacman-gen1-distill",
                        "fs2": "mspacman-gen1-fresh"}
    assert sup.SEEDS == {0: "2026-10-24"} and sup.ITERATIONS == 90
    for v, old in (("ds2", "ds"), ("fs2", "fs")):
        cmd = sup.command(v, 0)
        want = [sup.PY] + [
            x.replace(dsup.VARIANTS[old], sup.VARIANTS[v])
             .replace(dsup.PREP_REL, sup.PREP_REL)
            for x in dsup.command(old, 0)[1:]]
        assert cmd == want, v
        assert cmd[cmd.index("--arms") + 1] == sup.VARIANTS[v]
        assert sup.VARIANTS[v] in ablation.ARMS
        assert cmd[cmd.index("--token-vocab") + 1] == (sup.PREP_REL
                                                       + "/vocab.json")
        assert ("--distill-set" in cmd) == (v == "ds2")
        assert "--resume" not in cmd and "--snapshot-weights" in cmd
        for flag, val in (("--iterations", "90"), ("--train-steps", "2000"),
                          ("--env", ENV_ID), ("--frameskip", "4"),
                          ("--episode-cap", "3000"),
                          ("--episodes-per-iter", "3"),
                          ("--eval-every", "5"), ("--checkpoint-every", "2")):
            assert cmd[cmd.index(flag) + 1] == val, flag
        i = cmd.index("--ray")
        assert cmd[i:i + 2] == ["--ray", "--cpu-actor"]
    ds = sup.command("ds2", 0)
    assert ds[ds.index("--distill-set") + 1] == sup.PREP_REL + "/set.pkl"


def test_paths_the_run_folders_and_the_prep(real_sup):
    sup = real_sup
    for v, name in (("ds2", "mspacman-gen1-distill"),
                    ("fs2", "mspacman-gen1-fresh")):
        rel, base, jsonl, ckpt, deaths = sup.paths(v, 0)
        assert rel == f"benchmarks/results/{name}-s0-2026-10-24"
        assert jsonl.endswith(os.path.join(
            rel.split("/")[-1], "ablation-mspacmannoframeskip-skip4.jsonl"))
        assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-"
                             f"{sup.VARIANTS[v]}-seed0.ckpt")
        assert deaths.endswith("deaths.log")
        assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT           # runs from its own tree
    assert pathlib.Path(sup.MANIFEST) == SUPERVISORS / "gen1_manifest.json"
    assert sup.PREP_REL == "benchmarks/results/gen1-prep-s0-2026-10-24"
    assert pathlib.Path(sup.prep_dir()) == (ROOT / "benchmarks" / "results"
                                            / "gen1-prep-s0-2026-10-24")
    assert sup.DS_ROWS == (0,) + tuple(range(5, 95, 5))
    assert sup.DS_CURVE_ROWS == (1, 2, 3, 4)
    assert (sup.MAX_ATTEMPTS, sup.PREP_ATTEMPTS, sup.PREP_RETRY_S) == (40, 3,
                                                                       900)
    assert (sup.NO_PROGRESS_DEATHS, sup.STALL_MIN_S) == (3, 300)
    assert sup.PACE_ALERT_MIN == 25.0 and sup.COMMIT_ALERT_GB == 4.0


def test_g0_constants_name_the_live_g0_run(real_sup, g0sup, monkeypatch):
    """Names only: supervise_gen0.paths with ITS tree, the probe's file
    naming, ablation's snapshot folder."""
    sup = real_sup
    assert sup.G0_TREE == r"Y:\SwitchPlayer-gen0-run"
    assert sup.G0_ARM == G0_ARM == g0sup.VARIANTS["g0"]
    assert ablation.ARMS[DS2_ARM]["_master_arm"] == sup.G0_ARM
    assert ablation.ARMS[FS2_ARM]["_master_arm"] == sup.G0_ARM
    monkeypatch.setattr(g0sup, "REPO", sup.G0_TREE)
    rel, base, _jsonl, ckpt, _d = g0sup.paths("g0", 0)
    assert sup.G0_RUN == rel.split("/")[-1] == (
        "mspacman-tokens-gen0-s0-2026-10-24")
    assert sup.G0_BASE == base
    assert sup.G0_MARKER == ckpt
    assert sup.G0_SUPLOG == os.path.join(base, "supervisor.log")
    assert sup.G0_ABANDONED == os.path.join(base, "ABANDONED")
    assert sup.TAG == g0sup.TAG and sup.ENV_ID == g0sup.ENV_ID
    assert sup.G0_SNAPSHOT == os.path.join(
        base, f"snapshots{sup.TAG}-{G0_ARM}-seed0", "weights-iter090.pth")
    # probe_exhaustive_eval writes HERE/probe_exhaustive_eval-<the
    # snapshot's grandparent folder><suffix>.jsonl; G0's hand-off runs it
    # in G0's tree
    probes = os.path.join(sup.G0_TREE, "benchmarks", "probes")
    assert sup.G0_PROBES == (
        os.path.join(probes, f"probe_exhaustive_eval-{sup.G0_RUN}"
                             f"-acting.jsonl"),
        os.path.join(probes, f"probe_exhaustive_eval-{sup.G0_RUN}.jsonl"))
    assert sup.G0_ROWS == tuple(range(5, 95, 5))
    assert sup.G0_ITERATIONS == g0sup.ITERATIONS == 90


def test_the_alert_bounds(real_sup):
    sup = real_sup
    with ablation.arm_config(DS2_ARM):
        assert max(f for _, f in muzero.TOKEN_VOCAB) // 4 == \
            sup.MAX_TOKEN_STEPS == 8
    assert sup.NEAR_RANDOM == pytest.approx(sup.G_RANDOM + 0.2 * sup.G_HUMAN,
                                            abs=1e-9)
    assert (sup.G_RANDOM, sup.G_HUMAN, sup.NEAR_RANDOM) == (307.3, 973.9,
                                                            502.08)
    assert sup.REWARD_MAX_SLACK < 1e-3
    assert sup.REWARD_SUPPORT_EDGE == muzero.REWARD_CLIP == 400
    # far below the 3-decimal probe means' 1/7000 step in a 7-row mean
    assert sup.NEAR_RANDOM_TOL <= 1e-6 < 0.001 / 7


def _imports(src):
    import re
    return set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", src,
                          re.M))


def test_supervisor_pins_what_the_runs_and_preflight_use(real_sup, g0sup):
    import re
    sup = real_sup
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert set(g0sup.SUITES) <= set(sup.SUITES)
    assert set(g0sup.AUDITED) <= set(sup.AUDITED)
    assert set(sup.SUITES) - set(g0sup.SUITES) == {
        "tests/test_supervise_gen1.py", "tests/test_gen1.py"}
    assert set(sup.AUDITED) - set(g0sup.AUDITED) == {
        "tests/test_supervise_gen1.py", "tests/test_gen1.py",
        "benchmarks/supervisors/supervise_distill.py",
        "benchmarks/supervisors/supervise_gen1.py"}
    for rel in ("experiments/ablation.py", "experiments/distill.py",
                "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py"):
        assert rel in sup.AUDITED
    homes = ("", "experiments/", "benchmarks/probes/", "tests/")
    # the suites: what they import (at any depth) or read
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)
        for name in re.findall(r"(probe_\w+)\.py", src):
            if (ROOT / "benchmarks" / "probes" / f"{name}.py").exists():
                assert f"benchmarks/probes/{name}.py" in sup.AUDITED, \
                    (rel, name)
        for name in _imports(src):
            for home in homes:
                if (ROOT / f"{home}{name}.py").exists():
                    assert f"{home}{name}.py" in sup.AUDITED, (rel, name)
    # the code the runs, the prep and the probe execute: every repo module
    # it imports, transitively
    todo = ["experiments/ablation.py", "experiments/distill.py",
            "benchmarks/probes/probe_exhaustive_eval.py"]
    seen = set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        assert rel in sup.AUDITED, rel
        for name in _imports((ROOT / rel).read_text(encoding="utf-8")):
            for home in homes[:3]:
                if (ROOT / f"{home}{name}.py").exists():
                    todo.append(f"{home}{name}.py")
    assert {"muzero.py", "experiments/ray_actors.py"} <= seen


def test_supervisor_manifest_catches_an_edit(sup, monkeypatch, tmp_path):
    import shutil
    tree = tmp_path / "tree"
    for rel in sup.AUDITED:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    monkeypatch.setattr(sup, "REPO", str(tree))
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tree))))
    assert sup.manifest_check() == (True, "")
    for rel in ("experiments/distill.py", "experiments/ablation.py",
                "benchmarks/supervisors/supervise_gen1.py",
                "tests/test_supervise_gen1.py"):
        p = tree / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# edit\n")
        ok, why = sup.manifest_check()
        assert not ok and rel in why
        # CRLF vs LF: the same. Flip whichever way differs -- the run
        # worktree is CRLF (autocrlf), the dev tree LF (the distillation
        # step's audit r6 HIGH: LF -> CRLF on a CRLF file wrote \r\r\n).
        lf = orig.replace(b"\r\n", b"\n")
        flipped = lf if lf != orig else lf.replace(b"\n", b"\r\n")
        assert flipped != orig
        p.write_bytes(flipped)
        assert sup.manifest_check() == (True, "")
    (tmp_path / "m.json").write_text("[1, 2]")
    assert sup.manifest_check() == (False, "manifest check: not a JSON "
                                           "object")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M experiments/distill.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_refuses_bad_arguments(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a bad argv got past the usage check")

    for name in ("other_supervisors", "run_done", "wait_external",
                 "preflight", "live_scripts"):
        monkeypatch.setattr(sup, name, no)
    for argv in ([], ["ds2"], ["0"], ["ds2", "1"], ["ds", "0"], ["fs", "0"],
                 ["g0", "0"], ["tk", "0"], ["ds2", "x"], ["ds2", "0", "1"],
                 ["0", "ds2"], ["ds2", "0", "--no-wait"],
                 ["fs2", "0", "--no-wait"], ["ds2", "0", "--wait"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


def test_no_wait_is_gone(real_sup):
    """Test 2 has no fallback master: nothing of the --no-wait override
    survives."""
    src = inspect.getsource(real_sup)
    for word in ("no_wait", "NO-WAIT", "sanctioned", "stale_pick",
                 "unsettled", "TKX", "TK-90", "student-distill"):
        assert word not in src, word


@pytest.mark.parametrize("variant", ["ds2", "fs2"])
def test_supervisor_quick_probe_passes(real_sup, variant):
    import subprocess
    r = subprocess.run([sys.executable, "-c", real_sup.quick(variant)],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_quick_probes_tell_the_arms_apart(real_sup):
    import subprocess
    swapped = real_sup.quick("ds2").replace(DS2_ARM, FS2_ARM)
    r = subprocess.run([sys.executable, "-c", swapped], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode != 0 and "AssertionError" in r.stderr


# --------------------------------------------------------------------------
# the predecessors
# --------------------------------------------------------------------------
def test_ds2_waits_for_g0s_marker_and_every_row_of_both_evals(sup, fake_g0):
    marker, probes = fake_g0
    assert any("no checkpoint" in m for m in sup.pred_missing("ds2", 0))
    marker.write_bytes(b"x" * 1_000_000)
    assert any("full size" in m for m in sup.pred_missing("ds2", 0))
    marker.write_bytes(b"done")
    _rows(probes[0], range(5, 95, 5))
    _rows(probes[1], range(5, 90, 5), extra=[
        {"iteration": 90, "env": ENV_ID, "max_noop": 2},    # a smoke row
        {"iteration": 90, "env": "Pong-v4"}])
    assert sup.pred_missing("ds2", 0) == ["p.jsonl rows [90]"]
    _rows(probes[1], [90])
    assert sup.pred_missing("ds2", 0) == []
    # row 0 is no G0 row: never waited for
    assert 0 not in sup.G0_ROWS


def test_fs2_waits_for_ds2_its_rows_its_curve_and_the_prep(sup):
    ck = pathlib.Path(sup.paths("ds2", 0)[3])
    ck.parent.mkdir(parents=True)
    run = sup.run_name("ds2", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    _rows(probes / f"probe_exhaustive_eval-{run}-acting.jsonl",
          range(5, 95, 5))
    _rows(probes / f"probe_exhaustive_eval-{run}.jsonl", (0,) + tuple(
        range(5, 95, 5)))
    _rows(probes / f"probe_exhaustive_eval-{run}-offline-acting.jsonl",
          [1, 2, 3, 4])
    _rows(probes / f"probe_exhaustive_eval-{run}-offline.jsonl", [1, 2, 4])
    miss = sup.pred_missing("fs2", 0)
    assert any("no checkpoint" in m for m in miss)
    assert any("-acting.jsonl rows [0]" in m for m in miss)
    assert any("-offline.jsonl rows [3]" in m for m in miss)
    assert any("vocab.json" in m for m in miss)
    ck.write_bytes(b"done")
    _rows(probes / f"probe_exhaustive_eval-{run}-acting.jsonl", [0])
    _rows(probes / f"probe_exhaustive_eval-{run}-offline.jsonl", [3])
    os.makedirs(sup.prep_dir())
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text("{}")
    assert sup.pred_missing("fs2", 0) == []
    # fs2's predecessor is DS2 in THIS tree, driven by supervise_gen1.py ds2
    name, marker, _p, suplog, script, pvar, flag = sup.predecessor("fs2", 0)
    assert (name, marker, script, pvar) == ("DS2", str(ck),
                                            "supervise_gen1.py", "ds2")
    assert flag == os.path.join(sup.paths("ds2", 0)[1], "ABANDONED")
    name, marker, _p, suplog, script, pvar, flag = sup.predecessor("ds2", 0)
    assert (name, marker, suplog, script, pvar, flag) == (
        "G0", sup.G0_MARKER, sup.G0_SUPLOG, "supervise_gen0.py", "g0",
        sup.G0_ABANDONED)


def test_wait_external_blocks_and_alerts(sup, fake_g0, monkeypatch, capsys):
    marker, probes = fake_g0
    marker.write_bytes(b"done")
    os.utime(marker, (1, 1))                           # done long ago
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [])

    def rows_land(k):
        if k == 2:
            _rows(probes[0], range(5, 95, 5))
            _rows(probes[1], range(5, 95, 5))

    sleep = _bounded(rows_land)
    sleeps = sleep.calls
    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("ds2", 0) == "ready"
    out = capsys.readouterr().out
    assert "WAITING for G0 before ds2 s0" in out
    assert "ALERT: G0 completed" in out and "evals still lack rows" in out
    assert "G0 complete and evaluated; ds2 s0 may start" in out
    # the master's snapshot is checked while waiting
    assert "master snapshot missing: " + sup.G0_SNAPSHOT in out
    assert sleeps == [60, 60]
    # unfinished and undriven
    marker.write_bytes(b"x" * 1_000_000)
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("ds2", 0)
    out = capsys.readouterr().out
    assert "nothing drives it: no live supervise_gen0.py g0" in out
    assert "master snapshot missing" not in out        # no marker yet
    # driven, but its log silent for hours
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [5])
    pathlib.Path(sup.G0_SUPLOG).write_text("x")
    os.utime(sup.G0_SUPLOG, (1, 1))
    with pytest.raises(Stop):
        sup.wait_external("ds2", 0)
    assert "is alive but its log is silent" in capsys.readouterr().out
    # its supervisor gave up
    pathlib.Path(sup.G0_SUPLOG).write_text("LAUNCHED g0 s0 attempt 3" + NL
                                           + "NO PROGRESS: 3 x" + NL)
    with pytest.raises(Stop):
        sup.wait_external("ds2", 0)
    assert "G0's supervisor gave up -- ds2 keeps waiting" in \
        capsys.readouterr().out


def test_wait_external_stops_when_this_run_is_abandoned(sup, fake_g0,
                                                        monkeypatch):
    base = pathlib.Path(sup.paths("ds2", 0)[1])
    base.mkdir(parents=True)
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: (base / "ABANDONED").write_text("")))
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [1])
    assert sup.wait_external("ds2", 0) == "abandoned"


@pytest.mark.parametrize("g0_complete", [False, True])
def test_an_abandoned_g0_is_master_abandoned(sup, fake_g0, monkeypatch,
                                             capsys, g0_complete):
    """No fallback: ds2 stops (PREP FAILED, the hand-off's failed state)
    whether G0 was abandoned before or after completing."""
    if g0_complete:
        _complete_g0(sup, fake_g0)
    sleep = _bounded(lambda k: pathlib.Path(sup.G0_ABANDONED).write_text(""))
    sleeps = sleep.calls
    monkeypatch.setattr(sup.time, "sleep", sleep)
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [])
    if g0_complete:
        pathlib.Path(sup.G0_ABANDONED).write_text("")
    assert sup.wait_external("ds2", 0) == "failed"
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "MASTER ABANDONED" in l]
    assert line and line[0].startswith("PREP FAILED: MASTER ABANDONED")
    assert len(sleeps) == (0 if g0_complete else 1)
    # main stops there: no preflight, no prep, no launch
    monkeypatch.setattr(sup, "preflight", _stop)
    monkeypatch.setattr(sup, "run_prep", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["ds2", "0"]) == 1
    assert "PREP FAILED: MASTER ABANDONED" in capsys.readouterr().out


def test_fs2_keeps_waiting_for_an_abandoned_ds2(sup, monkeypatch, capsys):
    base = pathlib.Path(sup.paths("ds2", 0)[1])
    base.mkdir(parents=True)
    (base / "ABANDONED").write_text("")
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("fs2", 0)
    out = capsys.readouterr().out
    assert "DS2 is ABANDONED" in out and "Kai's call" in out
    assert "PREP FAILED" not in out
    # and an undriven DS2 is named
    (base / "ABANDONED").unlink()
    with pytest.raises(Stop):
        sup.wait_external("fs2", 0)
    assert "no live supervise_gen1.py ds2" in capsys.readouterr().out
    # audit r1 B-NIT2: DS2's supervisor stopped at the master gate -> Kai's
    # call, not "rerun its launcher"
    for stop in ("PREP FAILED: MASTER NEAR RANDOM: G0's deployed mean ...",
                 "PREP FAILED: MASTER ABANDONED: G0's run folder ..."):
        (base / "supervisor.log").write_text(
            "preflight ok (manifest + quick probe + 19 suites)\n"
            + stop + "\n")
        with pytest.raises(Stop):
            sup.wait_external("fs2", 0)
        out = capsys.readouterr().out
        assert "Kai's call" in out and stop[:30] in out
        assert "-- rerun its launcher" not in out
    # a later preflight (a relaunch after the fix) clears it
    (base / "supervisor.log").write_text(
        "PREP FAILED: MASTER NEAR RANDOM: ...\npreflight ok (...)\n")
    with pytest.raises(Stop):
        sup.wait_external("fs2", 0)
    assert "no live supervise_gen1.py ds2" in capsys.readouterr().out
    # another give-up keeps the rerun advice
    (base / "supervisor.log").write_text(
        "preflight ok (...)\nPREP FAILED after 3 attempts -- see ...\n")
    with pytest.raises(Stop):
        sup.wait_external("fs2", 0)
    assert "no live supervise_gen1.py ds2" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the master: fixed to G0, after the NEAR RANDOM rule
# --------------------------------------------------------------------------
def test_choose_master_is_g0s_final_snapshot(sup, fake_g0):
    marker, probes = fake_g0
    with pytest.raises(SystemExit, match="no completion marker"):
        sup.choose_master()
    marker.write_bytes(b"done")
    with pytest.raises(SystemExit, match="rows 60-90 are incomplete"):
        sup.choose_master()
    _rows(probes[0], range(5, 90, 5), 900.0)
    _rows(probes[1], range(5, 95, 5), 900.0)       # raw rows do not count
    with pytest.raises(SystemExit, match="rows 60-90 are incomplete"):
        sup.choose_master()
    _rows(probes[0], [90], 900.0)
    with pytest.raises(SystemExit, match="final snapshot is missing"):
        sup.choose_master()
    pathlib.Path(sup.G0_SNAPSHOT).write_bytes(b"x")
    assert sup.choose_master() == {
        "which": "G0", "snapshot": sup.G0_SNAPSHOT, "iterations": 90,
        "g0_mean_60_90": pytest.approx(900.0)}


@pytest.mark.parametrize("deployed, raw, ok", [
    ((502.08,) * 7, 502.08, True),      # exactly g = 0.2: passes
    # a decimal 502.08 mean whose float sum lands a hair below
    ((502.08,) * 6 + (502.08 - 1e-12,), 0.0, True),
    ((502.08,) * 6 + (502.079,), 9999.0, False),   # 1/7000 below: fails
    ((502.07,) * 7, 9999.0, False),     # the DEPLOYED mean decides
    ((9999.0,) * 7, 100.0, True),
    ((307.3,) * 7, 307.3, False)])
def test_the_near_random_rule(sup, fake_g0, deployed, raw, ok):
    marker, probes = fake_g0
    marker.write_bytes(b"done")
    pathlib.Path(sup.G0_SNAPSHOT).write_bytes(b"x")
    _rows(probes[0], range(5, 60, 5), 9999.0)      # rows before 60: unread
    for i, v in zip(range(60, 95, 5), deployed):
        _rows(probes[0], [i], v)
    _rows(probes[1], range(5, 95, 5), raw)
    mean = sum(deployed) / 7
    if ok:
        assert sup.choose_master()["g0_mean_60_90"] == pytest.approx(mean)
    else:
        with pytest.raises(SystemExit) as e:
            sup.choose_master()
        assert str(e.value).startswith("MASTER NEAR RANDOM")
        assert f"{mean:.2f}" in str(e.value) and "502.08" in str(e.value)


def test_the_window_is_rows_60_to_90(sup, fake_g0):
    marker, probes = fake_g0
    marker.write_bytes(b"done")
    pathlib.Path(sup.G0_SNAPSHOT).write_bytes(b"x")
    for i, v in zip(range(60, 95, 5), (800, 900, 1000, 1100, 1200, 700,
                                       1300)):
        _rows(probes[0], [i], float(v))
    assert sup.choose_master()["g0_mean_60_90"] == pytest.approx(1000.0)
    # a smoke row (max_noop) or another env's never stands in
    probes[0].write_text("")
    _rows(probes[0], range(60, 95, 5), 900.0, extra=[
        {"iteration": 60, "env": ENV_ID, "max_noop": 2, "exact_mean": 1.0}])
    assert sup.choose_master()["g0_mean_60_90"] == pytest.approx(900.0)


def test_an_abandoned_g0_is_never_the_master(sup, fake_g0):
    _complete_g0(sup, fake_g0, 2000.0)
    pathlib.Path(sup.G0_ABANDONED).write_text("")
    with pytest.raises(SystemExit) as e:
        sup.choose_master()
    assert str(e.value).startswith("MASTER ABANDONED")


def test_master_files_missing(sup, fake_g0):
    marker, _ = fake_g0
    assert sup.master_files_missing() == []          # G0 unfinished
    marker.write_bytes(b"done")
    assert sup.master_files_missing() == [sup.G0_SNAPSHOT]
    pathlib.Path(sup.G0_SNAPSHOT).write_bytes(b"x")
    assert sup.master_files_missing() == []


# --------------------------------------------------------------------------
# the prep
# --------------------------------------------------------------------------
PICK = {"which": "G0", "snapshot": "m.pth", "iterations": 90,
        "g0_mean_60_90": 700.0}


@pytest.fixture
def prep_env(sup, monkeypatch):
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(sup, "choose_master",
                        lambda: calls.append("choose") or dict(PICK))
    return calls


def _write_prep(sup, good=True, **doc):
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True, exist_ok=True)
    (d / "set.pkl").write_bytes(b"set")
    sha = hashlib.sha256(b"set" if good else b"other").hexdigest()
    (d / "vocab.json").write_text(json.dumps({
        "set_sha256": sha, "master_arm": G0_ARM, "master_iterations": 90,
        "student_ceiling": 7.5, "vocab": [[0, 4], [1, 4]], **doc}))


class FakeProc:
    """A finished Popen: runs `effect` (writing the prep's files) at start."""

    def __init__(self, rc, effect=None):
        self.returncode, self.pid = rc, 4321
        if effect:
            effect()

    def poll(self):
        return self.returncode


def _popen(monkeypatch, sup, plan, seen=None):
    """subprocess.Popen -> FakeProc per call, from `plan`: a list of (rc,
    effect); `seen` collects (args, kwargs)."""
    def popen(args, **k):
        if seen is not None:
            seen.append((args, k))
        rc, effect = plan.pop(0)
        return FakeProc(rc, effect)

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup.subprocess, "run", _stop)


def test_the_prep_runs_once_and_logs_its_master(sup, prep_env, monkeypatch,
                                                capsys):
    calls = prep_env
    runs = []
    _popen(monkeypatch, sup, [(1, None), (0, lambda: _write_prep(sup))], runs)
    env = {"X": "1"}
    assert sup.run_prep(env) is True
    out = capsys.readouterr().out
    assert "MASTER G0: m.pth (deployed 60-90 700.00, g 0.403)" in out
    assert "PREP attempt 1 failed" in out and "PREP done (attempt 2)" in out
    assert calls == ["choose"] and len(runs) == 2
    args, k = runs[0]
    assert args == [sup.PY, "experiments/distill.py", "--master", "m.pth",
                    "--master-iterations", "90", "--out", sup.PREP_REL,
                    "--seed", "0"]
    assert k["cwd"] == sup.REPO and k["env"] is env
    master = json.loads(pathlib.Path(sup.prep_dir(), "master.json")
                        .read_text())
    assert master == PICK
    assert not list(pathlib.Path(sup.prep_dir()).glob("*.tmp"))
    assert sup.run_prep({}) is True                  # done: nothing runs
    assert len(runs) == 2 and calls == ["choose"]


def test_a_restarted_prep_runs_the_master_gate_again(sup, prep_env,
                                                     monkeypatch):
    """The master is fixed: a prep (re)started by a new supervisor runs the
    gate again and rewrites master.json; a torn one is no obstacle."""
    calls = prep_env
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "master.json").write_text('{"which": "G0", "snap')    # torn
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))], seen)
    assert sup.run_prep({}) is True and calls == ["choose"]
    assert sup.read_pick(str(d / "master.json")) == PICK
    assert seen[0][0][3] == "m.pth" and seen[0][0][5] == "90"


@pytest.mark.parametrize("why, start", [
    ("MASTER NEAR RANDOM: G0's deployed mean ...", "PREP FAILED: MASTER NEAR "
                                                   "RANDOM"),
    ("MASTER ABANDONED: x", "PREP FAILED: MASTER ABANDONED"),
    ("G0's deployed rows 60-90 are incomplete", "PREP FAILED: G0's")])
def test_a_master_gate_failure_is_prep_failed(sup, prep_env, monkeypatch,
                                              capsys, why, start):
    monkeypatch.setattr(sup, "choose_master", lambda: (_ for _ in ()).throw(
        SystemExit(why)))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert any(l.startswith(start) for l in out.splitlines()), out
    assert not pathlib.Path(sup.prep_dir(), "master.json").exists()


def test_near_random_stops_the_real_gate(sup, fake_g0, monkeypatch, capsys):
    """The real choose_master inside run_prep: the line the hand-off reads
    as a failed state, with the mean; nothing starts."""
    _complete_g0(sup, fake_g0, 480.5)
    monkeypatch.setattr(sup, "wait_no_trainer", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("PREP FAILED")]
    assert line and line[0].startswith("PREP FAILED: MASTER NEAR RANDOM")
    assert "480.50" in line[0] and "Kai's call" in line[0]
    pathlib.Path(sup.G0_ABANDONED).write_text("")
    assert sup.run_prep({}) is False
    assert "PREP FAILED: MASTER ABANDONED" in capsys.readouterr().out


def test_the_prep_gives_up_after_its_attempts(sup, prep_env, monkeypatch,
                                              capsys):
    _popen(monkeypatch, sup, [(1, None)] * 3)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert out.count("failed (rc=1") == sup.PREP_ATTEMPTS == 3
    assert "PREP FAILED after 3 attempts" in out


def test_a_failed_attempt_names_its_reason(sup, prep_env, monkeypatch,
                                         capsys):
    # audit r1 B-L1: the reason in the supervisor's log, not only prep.log
    def exits():
        d = pathlib.Path(sup.prep_dir())
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "prep.log", "a", encoding="utf-8") as f:
            f.write("recorded 3 games\nSystemExit: the halved-clip ceiling "
                    "999.0 lies past the reward supports\n\n")
    _popen(monkeypatch, sup, [(1, exits)] * 3)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert out.count("its last line: SystemExit: the halved-clip ceiling "
                     "999.0 lies past the reward supports") == 3
    assert "PREP FAILED after 3 attempts" in out
    assert sup.prep_log_tail(os.path.join(sup.prep_dir(), "nope")) == ""


def test_the_prep_retries_15_min_apart(sup, prep_env, monkeypatch):
    sleeps = []
    monkeypatch.setattr(sup.time, "sleep", sleeps.append)
    _popen(monkeypatch, sup, [(1, None)] * 3)
    assert sup.run_prep({}) is False
    assert sleeps.count(sup.PREP_RETRY_S) == 2 and sup.PREP_RETRY_S == 900


def test_a_silent_prep_is_an_alert(sup, prep_env, monkeypatch, capsys):
    class Slow(FakeProc):
        polls = 0

        def poll(self):
            Slow.polls += 1
            return None if Slow.polls < 3 else self.returncode

    import time as real_time
    clock = [real_time.time()]      # prep.log's real mtime is "now"

    def sleep(s):
        clock[0] += 3600

    monkeypatch.setattr(sup, "time", SimpleNamespace(
        time=lambda: clock[0], sleep=sleep, strftime=real_time.strftime))
    monkeypatch.setattr(sup.subprocess, "Popen", lambda args, **k: Slow(
        0, lambda: _write_prep(sup)))
    assert sup.run_prep({}) is True
    assert "ALERT: the prep (pid 4321) has been silent" in \
        capsys.readouterr().out


def test_a_damaged_prep_is_moved_aside_and_redone(sup, prep_env, monkeypatch,
                                                  capsys):
    calls = prep_env
    _write_prep(sup, good=False)
    sup.write_pick(os.path.join(sup.prep_dir(), "master.json"), PICK)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep({}) is True
    out = capsys.readouterr().out
    assert "PREP damaged" in out and "moved aside" in out
    d = pathlib.Path(sup.prep_dir())
    assert len(list(d.glob("set.pkl.damaged-*"))) == 1
    assert len(list(d.glob("vocab.json.damaged-*"))) == 1
    assert sup.set_intact() == (True, "") and calls == ["choose"]


def test_a_prep_that_never_checks_out_fails(sup, prep_env, monkeypatch,
                                            capsys):
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup, good=False))] * 3)
    assert sup.run_prep({}) is False
    assert "does not match" in capsys.readouterr().out


@pytest.mark.parametrize("doc, why", [
    ({}, ""),
    ({"master_arm": "effzero-x-tokens-rgrow"}, "not G0's final snapshot"),
    ({"master_iterations": 270}, "not G0's final snapshot"),
    ({"student_ceiling": None}, "no valid student_ceiling"),
    ({"student_ceiling": 0.0}, "no valid student_ceiling"),
    ({"student_ceiling": True}, "no valid student_ceiling"),
    ({"student_ceiling": "7.5"}, "no valid student_ceiling")])
def test_set_intact_checks_the_master_and_the_ceiling(sup, doc, why):
    _write_prep(sup, **doc)
    ok, got = sup.set_intact()
    assert ok == (not why) and why in got
    pathlib.Path(sup.prep_dir(), "set.pkl").unlink()
    assert sup.set_intact()[1].startswith("prep files:")


def test_a_launched_run_never_redoes_its_prep(sup, prep_env, monkeypatch,
                                             capsys):
    """A launched DS2 is bound to the set; a transient lock or real damage
    blocks the launch (the integrity gate), never a redo."""
    calls = prep_env
    _write_prep(sup, good=False)
    _popen(monkeypatch, sup, [])                     # any launch fails
    assert sup.run_prep({}, redo_ok=False) is True
    out = capsys.readouterr().out
    assert "never redone" in out and calls == []
    assert not list(pathlib.Path(sup.prep_dir()).glob("*.damaged-*"))
    # a transient read failure that clears is not damage
    _write_prep(sup)
    sup.write_pick(os.path.join(sup.prep_dir(), "master.json"), PICK)
    real = sup.set_intact
    flaky = [(False, "prep files: [Errno 13] locked"), None]

    def once():
        r = flaky.pop(0)
        return r if r else real()

    monkeypatch.setattr(sup, "set_intact", once)
    assert sup.run_prep({}) is True
    assert "damaged" not in capsys.readouterr().out and calls == []


def test_a_launched_run_never_starts_a_prep(sup, prep_env, monkeypatch,
                                            capsys):
    """vocab.json missing after DS2 was launched: no prep at all."""
    calls = prep_env
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "set.pkl").write_bytes(b"bound")
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}, redo_ok=False) is True
    assert "never redone" in capsys.readouterr().out and calls == []
    assert (d / "set.pkl").read_bytes() == b"bound"


def test_the_prep_needs_the_audited_tree(sup, prep_env, monkeypatch, capsys):
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: (False, "not the audited revision: x"))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    assert "PREP FAILED: not the audited revision" in capsys.readouterr().out


def test_an_orphaned_prep_that_finished_counts(sup, prep_env, monkeypatch,
                                               capsys):
    """The next attempt's guard waited out an orphaned prep that wrote the
    prep: no second prep."""
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: _write_prep(sup))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is True
    assert "PREP done (by an earlier prep)" in capsys.readouterr().out


def test_stale_prep_tmp_files_are_swept(sup, prep_env, monkeypatch):
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "set.pkl.4242.tmp").write_bytes(b"x")
    (d / "keep.txt").write_bytes(b"x")
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep({}) is True
    assert not list(d.glob("*.tmp")) and (d / "keep.txt").exists()


def test_a_finished_prep_without_its_master_json_is_refused(
        sup, prep_env, monkeypatch, capsys):
    _write_prep(sup)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    assert "PREP FAILED: a finished prep without a readable master.json" in \
        capsys.readouterr().out
    assert sup.run_prep({}, redo_ok=False) is True   # launched: bound


def test_fs2_launches_only_on_ds2s_vocabulary_and_ceiling(sup, real_sup):
    """The marker is built the way run_one builds it (mark_run_completed of
    run_expect with student_expect's keys: the vocabulary AND the
    ceiling), and the check runs in a child process. (`sup` disarms
    vocab_is_ds: a fresh copy of the module, pointed at the same temp
    REPO, runs the real one.)"""
    fresh = _load(SUPERVISORS / "supervise_gen1.py", "supervise_gen1_vocab")
    fresh.REPO, fresh.PY = sup.REPO, sys.executable
    check = fresh.vocab_is_ds
    assert fresh.prep_dir() == sup.prep_dir()
    os.makedirs(sup.prep_dir())
    vocab = [[0, 4], [1, 4], [1, 16]]
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text(
        json.dumps({"vocab": vocab, "student_ceiling": 7.5}))
    ck = pathlib.Path(sup.paths("ds2", 0)[3])
    ck.parent.mkdir(parents=True)
    ok, why = check(0)
    assert not ok and "vocab check" in why            # no marker yet
    expect = ablation.run_expect(sup.VARIANTS["ds2"], 0, 90, 2000, 3, 5,
                                 student={"token_vocab": vocab,
                                          "reward_ceiling": 7.5,
                                          "distill_set_sha256": "x"})
    ablation.mark_run_completed(ck, expect)
    assert check(0) == (True, "")
    ablation.mark_run_completed(ck, {**expect, "token_vocab": vocab[:2]})
    ok, why = check(0)
    assert not ok and "not the vocabulary DS2 trained with" in why
    ablation.mark_run_completed(ck, {**expect, "reward_ceiling": 8.0})
    ok, why = check(0)
    assert not ok and "not the ceiling DS2 trained with" in why
    no_c = dict(expect)
    del no_c["reward_ceiling"]
    ablation.mark_run_completed(ck, no_c)
    assert not check(0)[0]
    assert not hasattr(real_sup, "torch")     # no torch in the supervisor


# --------------------------------------------------------------------------
# the trainer guard, one supervisor per variant
# --------------------------------------------------------------------------
def _psutil(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


GUARD_PROCS = [
    (1, "python.exe", ["py", "Y:" + chr(92) + "run" + chr(92) + "experiments"
                       + chr(92) + "ablation.py", "--arms", DS2_ARM]),
    (2, "python.exe", ["py", "experiments/ablation.py", "--arms", G0_ARM,
                       "--seed", "1"]),
    (3, "Python3.12", ["py", "experiments/ablation.py", "--arms", "pong-x"]),
    (4, "python.exe", ["py", "experiments" + chr(92) + "distill.py",
                       "--master", "m.pth"]),                    # the prep
    (5, "python.exe", ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                       "x.pth"]),
    (6, "python.exe", ["py", "-m", "pytest", "tests/test_gen1.py"]),
    (7, "python.exe", None),                # unreadable: never counted
    (8, "Code.exe", ["code", "experiments/ablation.py"]),
    (9, "git.exe", ["git", "diff", "--", "experiments/distill.py"]),
    (10, None, ["experiments/ablation.py"]),
    (11, "python.exe", ["py", "tests/test_distill.py"]),
    (12, "python.exe", ["py", "ablation.py", "--arms", FS2_ARM]),
    (13, "python.exe", ["py", "-m", "experiments.distill", "--master", "m"]),
    (14, "python.exe", ["py", "-m", "ablation"]),
    (15, "python.exe", ["py", "-m", "pytest", "ablation"]),
    (16, "python.exe", ["py", "experiments/ablation.pyc"]),
    (17, "python.exe", ["py", "benchmarks/supervisors/supervise_gen1.py",
                        "ds2", "0"]),
    (18, "python.exe", ["py", "-m", "distill"])]


def test_live_trainers_finds_any_run_or_prep(sup, real_sup, g0sup,
                                             monkeypatch):
    """is_trainer ported from supervise_gen0.py: the same verdict on every
    command line; the prep counts; python only; unreadable not counted."""
    fresh = _load(SUPERVISORS / "supervise_gen1.py", "supervise_gen1_procs")
    monkeypatch.setattr(sup, "_procs", fresh._procs)
    _psutil(monkeypatch, [{"pid": p, "name": n, "cmdline": c}
                          for p, n, c in GUARD_PROCS])
    assert sup.live_trainers() == [1, 2, 3, 4, 12, 13, 14, 18]
    for _p, _n, cmd in GUARD_PROCS:
        if cmd is not None:
            assert sup.is_trainer(cmd) == g0sup.is_trainer(cmd), cmd
    assert sup.TRAINER_SCRIPTS == g0sup.TRAINER_SCRIPTS
    assert sup.TRAINER_MODULES == g0sup.TRAINER_MODULES


def _procs(monkeypatch, sup, procs):
    monkeypatch.setattr(sup, "_procs", lambda: iter(procs))


def test_live_scripts_match_the_exact_name_and_variant(sup, monkeypatch):
    procs = [(1, ["py", r"C:\x\supervise_gen1.py", "ds2", "0"]),
             (2, ["py", "benchmarks/supervisors/supervise_gen1.py", "fs2",
                  "0"]),
             (3, ["py", "supervise_gen1.py.bak", "ds2", "0"]),
             (4, ["py", "benchmarks/supervisors/supervise_gen0.py", "g0",
                  "0"]),
             (5, ["py", "supervise_gen1.py", "ds", "0"]),
             (6, ["py", "experiments/ablation.py", "--arms", "x"])]
    _procs(monkeypatch, sup, procs)
    assert sup.live_scripts("supervise_gen1.py") == [1, 2, 5]
    assert sup.live_scripts("supervise_gen1.py", "ds2") == [1]
    assert sup.live_scripts("supervise_gen1.py", "fs2") == [2]
    assert sup.live_scripts("supervise_gen0.py", "g0") == [4]
    assert sup.live_trainers() == [6]
    monkeypatch.setattr(sup, "_mine", lambda: {1})
    assert sup.other_supervisors("ds2") == []
    assert sup.other_supervisors("fs2") == [2]


def test_one_supervisor_per_variant(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    _procs(monkeypatch, sup, [
        (20, ["py", "benchmarks/supervisors/supervise_gen1.py", "ds2", "0"])])
    monkeypatch.setattr(sup, "_mine", lambda: {99})
    for name in ("run_done", "wait_external", "preflight"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["ds2", "0"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_gen1.py ds2 is alive (pids [20])" in \
        out
    # the other variant's supervisor is no obstacle
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["fs2", "0"]) == 0
    assert "RUN COMPLETE fs2 s0" in capsys.readouterr().out


def test_wait_no_trainer_waits_out_any_trainer_and_streams_rows(
        sup, monkeypatch, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "ds2", 0)
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        jsonl.write_text(json.dumps({"iteration": 40, "reward_ceiling": 99.0,
                                     "elapsed_s": 1.0}) + NL)

    monkeypatch.setattr(sup.time, "sleep", sleep)
    sup.wait_no_trainer(rows)
    out = capsys.readouterr().out
    assert out.count("ALERT: an ablation.py or distill.py trainer is "
                     "already running (pids [7])") == 1     # told once
    assert sleeps == [60, 60]
    assert "EVAL ds2 s0 iter 40" in out           # an orphan stays watched


# --------------------------------------------------------------------------
# main: the order, the launch gate, the child
# --------------------------------------------------------------------------
def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys):
    monkeypatch.setattr(sup, "wait_external", _stop)
    for v in ("ds2", "fs2"):
        ck = pathlib.Path(sup.paths(v, 0)[3])
        ck.parent.mkdir(parents=True)
        ck.write_bytes(b"done")
        assert sup.main([v, "0"]) == 0
        assert f"RUN COMPLETE {v} s0" in capsys.readouterr().out


def test_a_resume_skips_the_wait(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "wait_external",
                        lambda *a: (_ for _ in ()).throw(
                            AssertionError("a resume must not wait")))
    monkeypatch.setattr(sup, "preflight", _stop)
    for v in ("ds2", "fs2"):
        ck = pathlib.Path(sup.paths(v, 0)[3])
        ck.parent.mkdir(parents=True)
        ck.write_bytes(b"x" * 1_000_001)
        with pytest.raises(Stop):
            sup.main([v, "0"])
        assert f"RESUMING {v} s0" in capsys.readouterr().out
        ck.unlink()
        # a LAUNCHED line in its own log is a resume too
        (ck.parent / "supervisor.log").write_text(
            f"LAUNCHED {v} s0 attempt 1 (fresh), pid 1" + NL)
        with pytest.raises(Stop):
            sup.main([v, "0"])
        assert f"RESUMING {v} s0" in capsys.readouterr().out
        (ck.parent / "supervisor.log").unlink()
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["ds2", "0"])


def test_launched(sup, tmp_path):
    base, ck = tmp_path / "b", tmp_path / "b" / "c.ckpt"
    base.mkdir()
    assert not sup.launched(str(base), str(ck))
    # a NO-WAIT line (the distillation step's override) is nothing here
    (base / "supervisor.log").write_text("NO-WAIT override: ds2 s0 x" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("x LAUNCHED ds2" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("LAUNCHED ds2 s0 attempt 1" + NL)
    assert sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("")
    ck.write_bytes(b"x")
    assert sup.launched(str(base), str(ck))


class Dead:
    pid, returncode = 99, 1

    def poll(self):
        return 1


def _loop(sup, monkeypatch, events, preps=None):
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: events.append("wait") or "ready")
    monkeypatch.setattr(sup, "preflight",
                        lambda v: events.append("preflight") or (True, ""))

    def run_prep(env, redo_ok):
        events.append(("prep", redo_ok))
        if preps is not None:
            preps.append(env)
        return True

    monkeypatch.setattr(sup, "run_prep", run_prep)
    monkeypatch.setattr(sup, "vocab_is_ds",
                        lambda seed: events.append("vocab") or (True, ""))
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or (True, ""))
    monkeypatch.setattr(sup, "set_intact",
                        lambda: events.append("intact") or (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)


def test_ds2_runs_wait_preflight_prep_then_each_launch_guarded(
        sup, monkeypatch, capsys):
    events, preps, launched = [], [], []
    _loop(sup, monkeypatch, events, preps)
    # the preflight's CPU-only env must never reach the trainer or the prep
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    ck = pathlib.Path(sup.paths("ds2", 0)[3])

    def popen(args, **k):
        events.append("popen")
        launched.append((list(args), k))
        if len(launched) == 1:          # the first attempt leaves a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["ds2", "0"]) == 1                 # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == (["wait", "preflight", ("prep", True)]
                      + ["guard", "manifest", "intact", "popen"] * 5)
    rel = sup.paths("ds2", 0)[0]
    for i, (args, k) in enumerate(launched):
        assert args == sup.command("ds2", 0) + (["--resume"] if i else [])
        assert k["cwd"] == sup.REPO
        assert k["env"]["ABLATION_RESULTS"] == rel
        assert k["env"]["PYTHONFAULTHANDLER"] == "1"
        assert "CUDA_VISIBLE_DEVICES" not in k["env"]
    # the prep's env: neither the run folder nor the CPU-only flag
    assert "ABLATION_RESULTS" not in preps[0]
    assert "CUDA_VISIBLE_DEVICES" not in preps[0]
    # --token-vocab / --distill-set name the prep's files, from the child's
    # cwd
    args, k = launched[0]
    for flag, name in (("--token-vocab", "vocab.json"),
                       ("--distill-set", "set.pkl")):
        assert os.path.normpath(os.path.join(
            k["cwd"], args[args.index(flag) + 1])) == os.path.normpath(
            os.path.join(sup.prep_dir(), name))


def test_fs2_has_no_prep_and_checks_ds2s_vocabulary(sup, monkeypatch):
    events, launched = [], []
    _loop(sup, monkeypatch, events)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    def popen(args, **k):
        events.append("popen")
        launched.append((list(args), k))
        raise Stop

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        sup.main(["fs2", "0"])
    assert events == ["wait", "preflight", "guard", "manifest", "vocab",
                      "popen"]
    args, k = launched[0]
    assert args == sup.command("fs2", 0) and "--distill-set" not in args
    assert os.path.normpath(os.path.join(
        k["cwd"], args[args.index("--token-vocab") + 1])) == \
        os.path.normpath(os.path.join(sup.prep_dir(), "vocab.json"))
    assert k["env"]["ABLATION_RESULTS"] == sup.paths("fs2", 0)[0]
    assert "CUDA_VISIBLE_DEVICES" not in k["env"]


def test_a_failed_prep_launches_nothing(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup, "run_prep", lambda env, redo_ok: False)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["ds2", "0"]) == 1


@pytest.mark.parametrize("variant, check", [("ds2", "set_intact"),
                                            ("fs2", "vocab_is_ds")])
def test_a_broken_prep_blocks_the_launch_until_intact(sup, monkeypatch,
                                                      capsys, variant,
                                                      check):
    events = []
    _loop(sup, monkeypatch, events)
    checks = [(False, "set.pkl does not match vocab.json's set_sha256"),
              (False, "set.pkl does not match vocab.json's set_sha256"),
              (True, "")]
    monkeypatch.setattr(sup, check,
                        lambda *a: events.append("check") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    with pytest.raises(Stop):
        sup.main([variant, "0"])
    gate = events[events.index("guard"):]
    assert gate == ["guard", "manifest", "check"] * 3
    out = capsys.readouterr().out
    assert out.count("not relaunching until") == 1          # told once


def test_a_failing_manifest_blocks_the_launch(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events)
    checks = [(False, "not the audited revision: x"), (True, "")]
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda *a, **k: events.append("popen") or _stop())
    with pytest.raises(Stop):
        sup.main(["ds2", "0"])
    assert events[3:] == ["guard", "manifest", "guard", "manifest",
                          "intact", "popen"]


def test_stray_state(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "wait_external", _stop)
    rel, base, _j, ckpt, _d = sup.paths("ds2", 0)
    base = pathlib.Path(base)
    off = base / f"{base.name}-offline" / "snapshots-x"
    off.mkdir(parents=True)
    (off / "weights-iter001.pth").write_bytes(b"x")
    own = base / "snapshots-x"
    own.mkdir()
    (own / "weights-iter000.pth").write_bytes(b"x")
    assert len(sup.stray_snapshots(str(base), ckpt)) == 2
    assert sup.main(["ds2", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    # DS2's own death before its first checkpoint: launched, and exactly
    # row 0 + the offline curve excused
    (base / "supervisor.log").write_text("LAUNCHED ds2 s0 attempt 1 (fresh)"
                                         + NL)
    assert sup.stray_snapshots(str(base), ckpt, own_ok=True) == []
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["ds2", "0"])
    (own / "weights-iter005.pth").write_bytes(b"x")  # a later row: stray
    assert sup.main(["ds2", "0"]) == 2
    (own / "weights-iter005.pth").unlink()
    # fs2 never gets the excuse
    _r, fbase, _fj, fck, _fd = sup.paths("fs2", 0)
    fsnap = pathlib.Path(fbase) / "snapshots-x"
    fsnap.mkdir(parents=True)
    (fsnap / "weights-iter000.pth").write_bytes(b"x")
    (pathlib.Path(fbase) / "supervisor.log").write_text("LAUNCHED fs2" + NL)
    assert sup.main(["fs2", "0"]) == 2
    capsys.readouterr()
    # probe rows without a checkpoint: refused even when launched
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    _rows(probes / f"probe_exhaustive_eval-{base.name}-offline-acting.jsonl",
          [1])
    assert sup.main(["ds2", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    assert sup.stray_probe_rows("ds2", 0, ckpt) == []


def test_the_attempt_loop_refuses_stray_probe_rows(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    run = sup.run_name("fs2", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"

    def rows_appear(*a):
        _rows(probes / f"probe_exhaustive_eval-{run}.jsonl", [5])

    monkeypatch.setattr(sup, "wait_no_trainer", rows_appear)
    assert sup.main(["fs2", "0"]) == 2
    assert "stray snapshots / probe jsonls" in capsys.readouterr().out


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("fs2", 0)[1])
    ckpt = pathlib.Path(sup.paths("fs2", 0)[3])
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["fs2", "0"]) == 0
    assert "ABANDONED fs2 s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["fs2", "0"]) == 0
    assert "RUN COMPLETE fs2 s0" in capsys.readouterr().out


def test_a_resumed_ds2_never_redoes_its_prep(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    ck = pathlib.Path(sup.paths("ds2", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)                  # a resume
    with pytest.raises(Stop):
        sup.main(["ds2", "0"])
    assert events[0] == "preflight" and ("prep", False) in events


def test_an_abandoned_ds2_runs_no_prep(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    base = pathlib.Path(sup.paths("ds2", 0)[1])

    def preflight(v):
        base.mkdir(parents=True, exist_ok=True)
        (base / "ABANDONED").write_text("")
        return True, ""

    monkeypatch.setattr(sup, "preflight", preflight)
    assert sup.main(["ds2", "0"]) == 0
    assert not any(isinstance(e, tuple) for e in events)
    assert "no prep" in capsys.readouterr().out


def test_main_returns_when_abandoned_while_waiting(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "abandoned")
    monkeypatch.setattr(sup, "preflight", _stop)
    assert sup.main(["ds2", "0"]) == 0
    assert "ABANDONED ds2 s0" in capsys.readouterr().out
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "failed")
    assert sup.main(["ds2", "0"]) == 1


def test_main_source_order(sup):
    src = inspect.getsource(sup.main)
    assert src.index("other_supervisors(") < src.index("run_done(") < \
        src.index("abandoned(base)") < src.index("stray_snapshots(") < \
        src.index("wait_external(")
    assert src.index("rows.prime()") < src.index("preflight(variant)")
    assert src.index("preflight(variant)") < src.index("run_prep(")
    assert src.index("run_prep(") < src.index("for attempt in range(")
    loop = src[src.index("for attempt in range("):]
    assert loop.index("wait_no_trainer(rows)") < loop.index(
        "manifest_check()") < loop.index("set_intact()") < loop.index(
        "vocab_is_ds(seed)") < loop.index("subprocess.Popen(")
    assert src.count("wait_no_trainer(") == 1
    assert "redo_ok=not launched(base, ckpt)" in src


# --------------------------------------------------------------------------
# CRASH LOOP / NO PROGRESS (supervise_gen0.py's rule)
# --------------------------------------------------------------------------
def _clock(sup, monkeypatch, t0=1_000_000.0):
    import time as real_time
    clock = [t0]

    def sleep(s):
        clock[0] += s                       # backoffs take clock time

    monkeypatch.setattr(sup, "time", SimpleNamespace(
        time=lambda: clock[0], sleep=sleep, strftime=real_time.strftime))
    return clock


@pytest.mark.parametrize("variant", ["ds2", "fs2"])
@pytest.mark.parametrize("life, stop, n", [
    (240, "CRASH LOOP", 5),          # quick deaths: CRASH LOOP's
    (480, "NO PROGRESS", 3),         # Test 1 audit r2 LOW-1: the old gap
    (1200, "NO PROGRESS", 3),
    (3 * 3600, "NO PROGRESS", 3)])
def test_every_repeating_death_without_a_checkpoint_stops(
        sup, monkeypatch, capsys, variant, life, stop, n):
    events = []
    _loop(sup, monkeypatch, events)
    clock = _clock(sup, monkeypatch)
    launches = []

    class Dies:
        pid, returncode = 99, 1

        def poll(self):
            clock[0] += life
            return 1

    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda args, **k: launches.append(args) or Dies())
    assert sup.main([variant, "0"]) == 1
    out = capsys.readouterr().out
    assert stop in out and "MAX ATTEMPTS" not in out
    assert len(launches) == n
    assert sup.STALL_MIN_S == 300 < 465


def test_a_checkpoint_resets_the_no_progress_count(sup, monkeypatch,
                                                   capsys):
    events = []
    _loop(sup, monkeypatch, events)
    clock = _clock(sup, monkeypatch)
    ck = pathlib.Path(sup.paths("ds2", 0)[3])
    launches = []

    class Slow:
        pid, returncode = 99, 1

        def poll(self):
            clock[0] += 1200                # each attempt lives 20 min
            return 1

    def popen(args, **k):
        launches.append(args)
        if len(launches) == 2:              # attempt 2 writes a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
            os.utime(ck, (clock[0] + 600, clock[0] + 600))
        return Slow()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["ds2", "0"]) == 1
    out = capsys.readouterr().out
    assert "NO PROGRESS: 3 attempts in a row" in out
    assert "CRASH LOOP" not in out
    assert len(launches) == 5                # 1 stall, reset, then 3 stalls
    assert launches[2][-1] == "--resume" and "--resume" not in launches[0]


# --------------------------------------------------------------------------
# rows: the prep's constant ceiling
# --------------------------------------------------------------------------
C = 7.5


@pytest.mark.parametrize("row, per, want", [
    ({"iteration": 1, "reward_ceiling": C, "reward_max": C}, "n/a", []),
    ({"iteration": 90, "reward_ceiling": C, "reward_max": 8 * C}, "24.9", []),
    ({"iteration": 9, "reward_ceiling": 7.4999}, "n/a", []),   # 4 decimals
    ({"iteration": 9, "reward_ceiling": 7.4}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5, "reward_ceiling": 1.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 50, "reward_ceiling": 160.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5}, "n/a", ["reward_ceiling"]),
    ({"iteration": 0, "reward_ceiling": None}, "n/a", []),
    ({"iteration": 40, "reward_ceiling": C}, "25.5", ["min/iteration"]),
    ({"iteration": 40, "reward_ceiling": C, "reward_max": 8 * C + 0.01},
     "n/a", ["reward_max"]),
    ({"iteration": 40, "reward_ceiling": 160.0, "reward_max": 400.0}, "30",
     ["reward_ceiling", "min/iteration", "reward_max"]),
    # past the support's edge (both reward_max alerts)
    ({"iteration": 40, "reward_ceiling": C, "reward_max": 400.5}, "n/a",
     ["8 x the ceiling", "support's edge"]),
])
def test_row_alerts(sup, row, per, want):
    got = sup.row_alerts(row, per, C)
    assert len(got) == len(want), got
    for g, w in zip(got, want):
        assert w in g


def test_row_alerts_at_the_rounded_bound(sup):
    c = 7.123457
    bound = round(8 * c, 4)              # what the row logs at the bound
    assert bound > 8 * c
    assert sup.row_alerts({"iteration": 3, "reward_ceiling": round(c, 4),
                           "reward_max": bound}, "n/a", c) == []


def test_row_alerts_without_a_ceiling(sup):
    got = sup.row_alerts({"iteration": 5, "reward_ceiling": 7.5,
                          "reward_max": 1e6}, "n/a", None)
    assert len(got) == 2 and "no valid student_ceiling" in got[0]
    assert "support's edge" in got[1]          # checked without a ceiling


def test_rows_check_against_the_preps_ceiling(sup, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "fs2", 0)

    def row(**r):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"elapsed_s": 1.0, **r}) + NL)

    row(iteration=1, reward_ceiling=7.5)
    rows.emit()                                  # no prep yet
    assert "no valid student_ceiling" in capsys.readouterr().out
    os.makedirs(sup.prep_dir())
    vocab = pathlib.Path(sup.prep_dir(), "vocab.json")
    vocab.write_text(json.dumps({"student_ceiling": 7.5}))
    row(iteration=2, reward_ceiling=7.5, reward_max=60.0, policy_maxp=0.4,
        reward_kept=0.9, tok_frames_per_decision=12.0)
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL fs2 s0 iter 2") and "ALERT" not in out
    assert "maxp 0.4" in out and "ceiling 7.5  kept 0.9  max 60.0" in out
    assert "frames/decision 12.0" in out
    row(iteration=3, reward_ceiling=1.0, reward_max=60.1)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: fs2 s0 row 3: reward_ceiling 1.0" in out
    assert "ALERT: fs2 s0 row 3: reward_max 60.1 > 8 x the ceiling" in out
    # read once: constant for the whole run
    vocab.write_text(json.dumps({"student_ceiling": 1.0}))
    row(iteration=4, reward_ceiling=7.5)
    rows.emit()
    assert "ALERT" not in capsys.readouterr().out


def test_rows_alert_on_low_host_commit(sup, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.2)
    os.makedirs(sup.prep_dir())
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text(
        json.dumps({"student_ceiling": 7.5}))
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "ds2", 0)
    jsonl.write_text(json.dumps({"iteration": 40, "reward_ceiling": 7.5,
                                 "elapsed_s": 1.0}) + NL)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: ds2 s0 row 40: host commit free 3.2 GB < 4 GB" in out


@pytest.mark.parametrize("text, want", [
    ('{"student_ceiling": 7.5}', 7.5),
    ('{"student_ceiling": 8}', 8.0),
    ('{"student_ceiling": 0}', None),
    ('{"student_ceiling": -1.0}', None),
    ('{"student_ceiling": true}', None),
    ('{"student_ceiling": "7.5"}', None),
    ('{"student_ceiling": NaN}', None),
    ('{"student_ceiling": Infinity}', None),
    ('{}', None), ('[7.5]', None), ('{"student_ceil', None), (None, None)])
def test_prep_ceiling(sup, text, want):
    if text is not None:
        os.makedirs(sup.prep_dir())
        pathlib.Path(sup.prep_dir(), "vocab.json").write_text(text)
    assert sup.prep_ceiling() == want
