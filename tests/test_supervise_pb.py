"""PB (Kai, 2026-10-24; CURRICULUM.md "Next experiment (pre-registered) --
PB: per-step prefix BatchNorm statistics on the token arm"):
supervise_pb.py runs TK's recipe on the PB arm (TK + PREFIX_BN_PER_STEP)
once G0 is out of the GPU's way. Pins the recipe against TK's (the arm the
one change on the command line; the arm's own delta is tests/test_pbn.py's),
the paths and the constants naming the live G0 run, TK's ceiling schedule
in the row alerts, the G0 wait (its completion marker or its ABANDONED file
lets PB start; an unfinished G0 nothing drives, or whose supervisor gave
up, is an hourly ALERT and the wait goes on; a resume skips it), the
manifest, the quick probe, the guard before each launch, the child's
environment, the refusals, NO PROGRESS and the AUDITED coverage. CPU only,
no launches.

No test reads LIVE state (the distillation step's audit r5 HIGH): the `sup`
fixture points every path into the live G0 run, this tree's results /
probes (REPO) and the process table at empty stand-ins; tests that need a
G0 or run state build it in tmp. Only the constants tests read the module
as deployed (`real_sup`), and only its names."""
import inspect
import json
import os
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
PB_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow-pbn")
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
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


def _bounded(effect=lambda k: None, n=5):
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
    return _load(SUPERVISORS / "supervise_pb.py", "supervise_pb")


@pytest.fixture
def sup(real_sup, monkeypatch, tmp_path):
    """The module with every path into the live G0 run redirected to an
    empty temp folder (G0 "unfinished, not abandoned, undriven"), REPO (the
    run folder and probe jsonls) an empty temp tree, the manifest absent,
    the process table empty, host memory unknown and the real preflight /
    death snapshot disarmed: no test may depend on the real G0's state or
    on this tree's own run state -- the preflight runs this suite while G0
    trains, after it completed, and in the run worktree beside PB's live
    files. Tests that need a state build it in tmp (fake_g0)."""
    live = tmp_path / "_no_live_runs" / "g0"
    monkeypatch.setattr(real_sup, "G0_TREE", str(live.parent))
    monkeypatch.setattr(real_sup, "G0_BASE", str(live))
    for name, rel in (("G0_MARKER", "resume.ckpt"),
                      ("G0_ABANDONED", "ABANDONED"),
                      ("G0_SUPLOG", "supervisor.log")):
        monkeypatch.setattr(real_sup, name, str(live / rel))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(real_sup, "REPO", str(repo))
    monkeypatch.setattr(real_sup, "MANIFEST", str(tmp_path / "no-m.json"))
    monkeypatch.setattr(real_sup, "_procs", lambda: iter(()))
    monkeypatch.setattr(real_sup, "commit_free_gb", lambda: None)

    def disarmed(*a, **k):
        raise AssertionError("a test reached the real preflight without "
                             "stubbing it")

    monkeypatch.setattr(real_sup, "preflight", disarmed)
    monkeypatch.setattr(real_sup, "snapshot", lambda *a: None)
    return real_sup


@pytest.fixture
def fake_g0(sup):
    """G0's run folder built in tmp: its (not yet existing) marker."""
    marker = pathlib.Path(sup.G0_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker


@pytest.fixture
def g0sup():
    return _load(SUPERVISORS / "supervise_gen0.py", "supervise_gen0")


@pytest.fixture
def g1sup():
    return _load(SUPERVISORS / "supervise_gen1.py", "supervise_gen1")


@pytest.fixture
def tk90():
    return _load(SUPERVISORS / "supervise_rgrow.py", "supervise_rgrow")


@pytest.fixture
def dsup():
    return _load(SUPERVISORS / "supervise_distill.py", "supervise_distill")


def _procs(monkeypatch, sup, procs):
    monkeypatch.setattr(sup, "_procs", lambda: iter(procs))


G0_SUPERVISOR = ["py", "-u", "benchmarks" + chr(92) + "supervisors"
                 + chr(92) + "supervise_gen0.py", "g0", "0"]


# --------------------------------------------------------------------------
# the recipe, the constants
# --------------------------------------------------------------------------
def test_recipe_is_tks_with_the_pb_arm(real_sup, tk90, g0sup):
    """PB = TK + the flag: on the command line the arm is the only change
    against TK-90's own recipe (and Test 1's, which is TK's)."""
    sup = real_sup
    assert sup.VARIANTS == {"pb": PB_ARM} and PB_ARM == TK_ARM + "-pbn"
    assert tk90.VARIANTS["tk"] == TK_ARM == sup.TK_ARM
    assert sup.RUNS == {"pb": "mspacman-tokens-rgrow-pbn"}
    assert sup.SEEDS == {0: "2026-10-25"} and sup.ITERATIONS == 90
    p, t, g = (sup.command("pb", 0), tk90.command("tk", 0),
               g0sup.command("g0", 0))
    i = p.index("--arms")
    assert p[i + 1] == PB_ARM and t[i + 1] == TK_ARM and g[i + 1] == G0_ARM
    assert p[:i + 1] + p[i + 2:] == t[:i + 1] + t[i + 2:] == \
        g[:i + 1] + g[i + 2:]
    assert p[0] == sup.PY and p[1] == "experiments/ablation.py"
    assert "--resume" not in p and "--snapshot-weights" in p
    for flag, val in (("--seed", "0"), ("--iterations", "90"),
                      ("--train-steps", "2000"), ("--env", ENV_ID),
                      ("--frameskip", "4"), ("--episode-cap", "3000"),
                      ("--episodes-per-iter", "3"), ("--eval-every", "5"),
                      ("--checkpoint-every", "2")):
        assert p[p.index(flag) + 1] == val, flag
    assert p[-2:] == ["--ray", "--cpu-actor"]
    assert len(p) == 25                   # nothing beyond TK's flags


def test_paths_and_the_run_folder(real_sup):
    sup = real_sup
    rel, base, jsonl, ckpt, deaths = sup.paths("pb", 0)
    assert rel == "benchmarks/results/mspacman-tokens-rgrow-pbn-s0-2026-10-25"
    assert sup.run_name("pb", 0) == rel.split("/")[-1]
    assert jsonl.endswith(os.path.join(
        rel.split("/")[-1], "ablation-mspacmannoframeskip-skip4.jsonl"))
    assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{PB_ARM}"
                         f"-seed0.ckpt")
    assert deaths.endswith("deaths.log")
    assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT           # runs from its own tree
    assert pathlib.Path(sup.MANIFEST) == SUPERVISORS / "pb_manifest.json"
    assert sup.MAX_ATTEMPTS == 40
    assert (sup.NO_PROGRESS_DEATHS, sup.STALL_MIN_S) == (3, 300)
    assert sup.PACE_ALERT_MIN == 25.0 and sup.COMMIT_ALERT_GB == 4.0
    assert sup.PRED_SILENT_ALERT_S == 4 * 3600
    assert (sup.ENV_ID, sup.TAG) == (ENV_ID, "-mspacmannoframeskip-skip4")
    assert not hasattr(sup, "torch")          # no torch in the supervisor


def test_g0_constants_name_the_live_g0_run(real_sup, g0sup, g1sup,
                                           monkeypatch):
    """Names only: supervise_gen0.paths with ITS tree, as supervise_gen1.py
    (Test 2's audited wait) names them."""
    sup = real_sup
    assert sup.G0_TREE == r"Y:\SwitchPlayer-gen0-run" == g1sup.G0_TREE
    assert sup.G0_ARM == G0_ARM == g0sup.VARIANTS["g0"]
    monkeypatch.setattr(g0sup, "REPO", sup.G0_TREE)
    rel, base, _jsonl, ckpt, _d = g0sup.paths("g0", 0)
    assert sup.G0_RUN == rel.split("/")[-1] == (
        "mspacman-tokens-gen0-s0-2026-10-24")
    assert sup.G0_BASE == base == g1sup.G0_BASE
    assert sup.G0_MARKER == ckpt == g1sup.G0_MARKER
    assert sup.G0_SUPLOG == os.path.join(base, "supervisor.log") == \
        g1sup.G0_SUPLOG
    assert sup.G0_ABANDONED == os.path.join(base, "ABANDONED") == \
        g1sup.G0_ABANDONED


# --------------------------------------------------------------------------
# rows: TK's within-run ceiling schedule
# --------------------------------------------------------------------------
def _tk(it):
    """TK's ceiling at row `it`, as a row logs it (4 decimals)."""
    return round(160.0 ** min(1.0, (it - 1) * 2000 / 90_000), 4)


def test_the_ceiling_schedule_is_tks(real_sup, dsup, tk90):
    """supervise_distill.py's fs rule and TK-90's own supervisor's, and
    the arm machinery's (muzero.reward_ceiling at the iteration's starting
    step) under TK's arm."""
    sup = real_sup
    assert (sup.CEILING, sup.CEILING_END_STEPS) == (160.0, 90_000)
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(90, 2000):
        for it in range(1, 91):
            want = muzero.reward_ceiling((it - 1) * 2000)
            assert sup.expected_ceiling(it) == pytest.approx(want, rel=1e-9)
            assert sup.expected_ceiling(it) == dsup.expected_ceiling(
                "fs", it) == tk90.expected_ceiling(it), it
    assert sup.expected_ceiling(1) == 1.0
    assert sup.expected_ceiling(10) == pytest.approx(160 ** 0.2)
    assert sup.expected_ceiling(23) == pytest.approx(160 ** (44 / 90))
    assert sup.expected_ceiling(45) == pytest.approx(160 ** (88 / 90))
    assert sup.expected_ceiling(45) < 160.0 == sup.expected_ceiling(46) == \
        sup.expected_ceiling(90)


def test_the_pb_arm_runs_tks_ceiling_schedule(real_sup):
    """What the row alert checks is what the PB arm does: TK's schedule."""
    with ablation.arm_config(PB_ARM), ablation._run_length_globals(90, 2000):
        for it in range(1, 91):
            assert real_sup.expected_ceiling(it) == pytest.approx(
                muzero.reward_ceiling((it - 1) * 2000), rel=1e-9), it


def test_the_alert_bounds(real_sup):
    assert real_sup.REWARD_SUPPORT_EDGE == muzero.REWARD_CLIP == 400


@pytest.mark.parametrize("row, per, want", [
    ({"iteration": 1, "reward_ceiling": 1.0, "reward_max": 1.0}, "n/a", []),
    ({"iteration": 2, "reward_ceiling": _tk(2)}, "n/a", []),
    ({"iteration": 10, "reward_ceiling": _tk(10), "reward_max": 8.0},
     "14.0", []),
    ({"iteration": 23, "reward_ceiling": _tk(23)}, "n/a", []),
    ({"iteration": 45, "reward_ceiling": _tk(45)}, "n/a", []),
    ({"iteration": 46, "reward_ceiling": 160.0, "reward_max": 400.0},
     "24.9", []),
    ({"iteration": 90, "reward_ceiling": 160.0, "reward_max": 80.0}, "n/a",
     []),
    ({"iteration": 0, "reward_ceiling": None}, "n/a", []),
    # G0's constant sign clip, a row off the schedule, the end state early
    ({"iteration": 10, "reward_ceiling": 1.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 50, "reward_ceiling": 1.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 10, "reward_ceiling": _tk(11)}, "n/a",
     ["reward_ceiling"]),
    ({"iteration": 45, "reward_ceiling": 160.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5, "reward_ceiling": "1.2"}, "n/a", ["reward_ceiling"]),
    ({"iteration": 40, "reward_ceiling": _tk(40)}, "25.5",
     ["min/iteration"]),
    ({"iteration": 40, "reward_ceiling": _tk(40), "reward_max": 400.5},
     "n/a", ["support's edge"]),
    ({"iteration": 40, "reward_ceiling": 1.0, "reward_max": 1e4}, "30",
     ["reward_ceiling", "min/iteration", "support's edge"]),
])
def test_row_alerts(sup, row, per, want):
    got = sup.row_alerts(row, per)
    assert len(got) == len(want), got
    for g, w in zip(got, want):
        assert w in g


def test_rows_print_the_clip_prior_and_token_fields(sup, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "pb", 0)

    def row(**r):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"elapsed_s": 1.0, **r}) + NL)

    row(iteration=5, return_mean=500.0, reward_ceiling=_tk(5),
        reward_kept=0.97, reward_max=3.0, policy_maxp=0.61,
        tok_frames_per_decision=9.5, tok_decisions_per_game=40.0)
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL pb s0 iter 5")
    assert "maxp 0.61" in out
    assert f"ceiling {_tk(5)}  kept 0.97  max 3.0" in out
    assert "frames/decision 9.5  decisions/life 40.0" in out
    assert "ALERT" not in out
    rows.emit()                                  # a row is printed once
    assert capsys.readouterr().out == ""
    row(iteration=6, reward_ceiling=1.0)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: pb s0 row 6: reward_ceiling 1.0, the schedule says" in out


def test_rows_alert_on_low_host_commit(sup, tmp_path, monkeypatch, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "pb", 0)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.2)
    jsonl.write_text(json.dumps({"iteration": 40,
                                 "reward_ceiling": _tk(40),
                                 "elapsed_s": 1.0}) + NL)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: pb s0 row 40: host commit free 3.2 GB < 4 GB" in out


# --------------------------------------------------------------------------
# what the preflight pins: the manifest, the quick probe, the coverage
# --------------------------------------------------------------------------
def _imports(src):
    return set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", src,
                          re.M))


def test_supervisor_pins_what_the_run_and_preflight_use(real_sup, g0sup):
    sup = real_sup
    for rel in sup.AUDITED:                      # every suite among them
        assert (ROOT / rel).is_file(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert len(set(sup.SUITES)) == len(sup.SUITES)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert set(g0sup.SUITES) <= set(sup.SUITES)
    assert set(g0sup.AUDITED) <= set(sup.AUDITED)
    assert set(sup.SUITES) - set(g0sup.SUITES) == {
        "tests/test_supervise_pb.py", "tests/test_pbn.py",
        "tests/test_supervise_gen1.py", "tests/test_gen1.py"}
    assert set(sup.AUDITED) - set(g0sup.AUDITED) == {
        "tests/test_supervise_pb.py", "tests/test_pbn.py",
        "tests/test_supervise_gen1.py", "tests/test_gen1.py",
        "benchmarks/probes/probe_prefix_bn.py",
        "benchmarks/supervisors/supervise_distill.py",
        "benchmarks/supervisors/supervise_gen1.py",
        "benchmarks/supervisors/supervise_pb.py"}
    for rel in ("networks.py", "muzero.py", "muzero_vtree.py",
                "experiments/ablation.py", "experiments/ray_actors.py",
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
    # the quick probe's imports
    for name in re.findall(r"import ([\w, ]+);", sup.quick("pb"))[-1].split(
            ","):
        name = name.strip()
        if name != "sys":
            assert any(f"{h}{name}.py" in sup.AUDITED for h in homes), name
    # the code the run and the readout's probes execute: every repo module
    # it imports, transitively
    todo = ["experiments/ablation.py", "experiments/distill.py",
            "benchmarks/probes/probe_exhaustive_eval.py",
            "benchmarks/probes/probe_prefix_bn.py"]
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
    assert {"muzero.py", "networks.py", "experiments/ray_actors.py"} <= seen


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
    for rel in ("experiments/ablation.py", "networks.py",
                "benchmarks/probes/probe_prefix_bn.py",
                "benchmarks/supervisors/supervise_pb.py",
                "tests/test_supervise_pb.py"):
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
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tree))))
    (tree / "networks.py").unlink()               # a missing AUDITED file
    ok, why = sup.manifest_check()
    assert not ok and why.startswith("manifest check:")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M networks.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_write_manifest_from_a_clean_tree(sup, monkeypatch,
                                                     tmp_path, capsys):
    """A clean tree: the AUDITED hashes and the commit, nothing else."""
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup, "REPO", str(ROOT))
    calls = []

    def run(args, **k):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=(
            "abc123\n" if args[1] == "rev-parse" else ""), stderr="")

    monkeypatch.setattr(sup.subprocess, "run", run)
    assert sup.main(["--write-manifest"]) == 0
    doc = json.loads((tmp_path / "m.json").read_text())
    assert doc.pop("_commit") == "abc123"
    assert doc == sup.file_hashes(str(ROOT))
    assert calls[0][:4] == ["git", "status", "--porcelain", "--"]
    assert calls[0][4:] == sup.AUDITED


def test_supervisor_refuses_bad_arguments(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a bad argv got past the usage check")

    for name in ("other_supervisors", "run_done", "wait_external",
                 "preflight", "live_scripts"):
        monkeypatch.setattr(sup, name, no)
    for argv in ([], ["pb"], ["0"], ["pb", "1"], ["tk", "0"], ["g0", "0"],
                 ["pb", "x"], ["pb", "0", "1"], ["0", "pb"],
                 ["pb", "0", "--no-wait"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


def test_supervisor_quick_probe_passes(real_sup):
    import subprocess
    r = subprocess.run([sys.executable, "-c", real_sup.quick("pb")],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_quick_probe_tells_the_arms_apart(real_sup):
    """TK itself (the flag off) and G0 fail it."""
    import subprocess
    probe = real_sup.quick("pb")
    assert "spec['PREFIX_BN_PER_STEP'] is True" in probe
    for arm in (TK_ARM, G0_ARM):
        r = subprocess.run([sys.executable, "-c",
                            probe.replace(repr(PB_ARM), repr(arm))],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        assert r.returncode != 0 and ("AssertionError" in r.stderr
                                      or "KeyError" in r.stderr), arm


# --------------------------------------------------------------------------
# the G0 wait
# --------------------------------------------------------------------------
def test_pred_missing_is_g0s_marker_or_its_abandoned_file(sup, fake_g0):
    marker = fake_g0
    assert len(sup.pred_missing()) == 1
    assert "no checkpoint" in sup.pred_missing()[0]
    marker.write_bytes(b"x" * 1_000_000)          # a live checkpoint
    assert "full size" in sup.pred_missing()[0]
    assert sup.marker_missing(str(marker)) == [
        "completion marker (checkpoint still full size)"]
    marker.write_bytes(b"done")                   # the completion marker
    assert sup.pred_missing() == [] == sup.marker_missing(str(marker))
    marker.write_bytes(b"x" * 1_000_000)
    pathlib.Path(sup.G0_ABANDONED).write_text("")  # abandoned: PB may go
    assert sup.pred_missing() == []
    assert sup.pred_unattended() is None


def test_g0s_marker_lets_pb_start_at_once(sup, fake_g0, monkeypatch,
                                          capsys):
    fake_g0.write_bytes(b"done")
    monkeypatch.setattr(sup.time, "sleep", _stop)
    assert sup.wait_external("pb", 0) == "ready"
    out = capsys.readouterr().out
    assert "G0 complete; pb s0 may start" in out
    assert "WAITING" not in out and "ALERT" not in out


def test_g0s_abandoned_file_lets_pb_start(sup, fake_g0, monkeypatch,
                                          capsys):
    """Abandoned while PB waits: PB goes (it needs no G0 result)."""
    fake_g0.write_bytes(b"x" * 1_000_000)
    _procs(monkeypatch, sup, [(5, G0_SUPERVISOR)])
    sleep = _bounded(lambda k: k == 3 and pathlib.Path(
        sup.G0_ABANDONED).write_text(""))
    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("pb", 0) == "ready"
    out = capsys.readouterr().out
    assert sleep.calls == [60, 60, 60]
    assert out.count("WAITING for G0 before pb s0 [0 h]: missing G0's "
                     "completion marker") == 1
    assert "G0 ABANDONED; pb s0 may start" in out
    assert "ALERT" not in out                   # driven, its log fresh


def test_an_undriven_g0_alerts_hourly_and_the_wait_goes_on(
        sup, fake_g0, monkeypatch, capsys):
    fake_g0.write_bytes(b"x" * 1_000_000)
    sleep = _bounded(lambda k: k == 125 and fake_g0.write_bytes(b"done"),
                     n=200)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("pb", 0) == "ready"
    out = capsys.readouterr().out
    assert len(sleep.calls) == 125 and set(sleep.calls) == {60}
    for h in (0, 1, 2):
        assert f"WAITING for G0 before pb s0 [{h} h]" in out
    assert out.count("WAITING") == 3
    assert out.count("ALERT: G0 is unfinished and nothing drives it: no "
                     "live supervise_gen0.py g0 -- rerun its launcher") == 3
    assert "gave up" not in out
    assert out.rstrip().endswith("G0 complete; pb s0 may start (G0's evals "
                                 "and result are not waited for)")


def test_a_driven_g0_is_no_alert_until_its_log_goes_silent(
        sup, fake_g0, monkeypatch, capsys):
    fake_g0.write_bytes(b"x" * 1_000_000)
    _procs(monkeypatch, sup, [
        (5, G0_SUPERVISOR),
        (6, ["py", "supervise_gen0.py", "g1", "0"]),        # not G0's
        (7, ["py", "benchmarks/supervisors/supervise_pb.py", "pb", "0"])])
    assert sup.live_scripts("supervise_gen0.py", "g0") == [5]
    pathlib.Path(sup.G0_SUPLOG).write_text("LAUNCHED g0 s0 attempt 1" + NL)
    assert sup.pred_unattended() is None
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("pb", 0)
    out = capsys.readouterr().out
    assert "WAITING for G0" in out and "ALERT" not in out
    os.utime(sup.G0_SUPLOG, (1, 1))              # silent for decades
    with pytest.raises(Stop):
        sup.wait_external("pb", 0)
    out = capsys.readouterr().out
    assert ("ALERT: G0 is unfinished and nothing drives it: G0's supervisor "
            "(pids [5]) is alive but its log is silent") in out
    assert "do NOT rerun its launcher beside it" in out
    # undriven: G0's supervisor for another variant does not count
    _procs(monkeypatch, sup, [(6, ["py", "supervise_gen0.py", "g1", "0"])])
    assert "no live supervise_gen0.py g0" in sup.pred_unattended()


@pytest.mark.parametrize("stop", ["CRASH LOOP: 5 deaths within 60 min",
                                  "MAX ATTEMPTS REACHED",
                                  "NO PROGRESS: 3 attempts in a row"])
def test_a_g0_supervisor_that_gave_up_alerts_and_the_wait_goes_on(
        sup, fake_g0, monkeypatch, capsys, stop):
    fake_g0.write_bytes(b"x" * 1_000_000)
    log = pathlib.Path(sup.G0_SUPLOG)
    log.write_text("preflight ok (x)" + NL + "LAUNCHED g0 s0 attempt 3"
                   + NL + stop + NL)
    assert sup.pred_gave_up()
    sleep = _bounded(n=61)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(Stop):
        sup.wait_external("pb", 0)
    out = capsys.readouterr().out
    assert len(sleep.calls) == 62                 # kept waiting
    assert out.count("ALERT: G0's supervisor gave up -- pb keeps waiting "
                     "(Kai's call)") == 2         # hourly
    assert out.count("nothing drives it: no live supervise_gen0.py g0") == 2
    # a later launch or preflight line clears it
    for live in ("LAUNCHED g0 s0 attempt 4", "preflight ok (y)",
                 "PREFLIGHT not ready (0 min): x"):
        log.write_text(stop + NL + live + NL)
        assert not sup.pred_gave_up(), live
    log.write_text("x " + stop + NL)               # not at a line start
    assert not sup.pred_gave_up()
    log.unlink()
    assert not sup.pred_gave_up()


def test_the_wait_stops_when_this_run_is_abandoned(sup, fake_g0,
                                                   monkeypatch, capsys):
    base = pathlib.Path(sup.paths("pb", 0)[1])
    base.mkdir(parents=True)
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: k == 2 and (base / "ABANDONED").write_text("")))
    assert sup.wait_external("pb", 0) == "abandoned"
    # main returns there: no preflight, no launch
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: (base / "ABANDONED").write_text("")))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    capsys.readouterr()
    assert sup.main(["pb", "0"]) == 0
    out = capsys.readouterr().out
    assert "WAITING for G0" in out
    assert "ABANDONED pb s0: the run folder holds an ABANDONED file" in out


def test_main_waits_for_g0_before_the_preflight(sup, fake_g0, monkeypatch,
                                                capsys):
    """A fresh start: G0 unfinished -> no preflight; G0 done -> on to it."""
    fake_g0.write_bytes(b"x" * 1_000_000)
    seen = []
    monkeypatch.setattr(sup, "preflight", lambda v: seen.append(v) or _stop())
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    monkeypatch.setattr(sup.time, "sleep", _bounded(n=3))
    with pytest.raises(Stop, match="the wait never returned"):
        sup.main(["pb", "0"])
    assert seen == []
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: fake_g0.write_bytes(b"done")))
    with pytest.raises(Stop):
        sup.main(["pb", "0"])
    assert seen == ["pb"]
    out = capsys.readouterr().out
    assert "G0 complete; pb s0 may start" in out and "RESUMING" not in out


def test_a_resume_skips_the_wait(sup, fake_g0, monkeypatch, capsys):
    fake_g0.write_bytes(b"x" * 1_000_000)            # G0 still training
    monkeypatch.setattr(sup, "wait_external",
                        lambda *a: (_ for _ in ()).throw(
                            AssertionError("a resume must not wait")))
    monkeypatch.setattr(sup, "preflight", _stop)
    ck = pathlib.Path(sup.paths("pb", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)
    with pytest.raises(Stop):
        sup.main(["pb", "0"])
    assert "RESUMING pb s0" in capsys.readouterr().out
    ck.unlink()
    # a LAUNCHED line in its own log is a resume too
    (ck.parent / "supervisor.log").write_text(
        "LAUNCHED pb s0 attempt 1 (fresh), pid 1" + NL)
    with pytest.raises(Stop):
        sup.main(["pb", "0"])
    assert "RESUMING pb s0" in capsys.readouterr().out
    # anything else is a fresh start: it waits
    (ck.parent / "supervisor.log").write_text(
        "WAITING for G0 before pb s0 [3 h]: missing x" + NL
        + "x LAUNCHED pb" + NL)
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["pb", "0"])


def test_launched(sup, tmp_path):
    base, ck = tmp_path / "b", tmp_path / "b" / "c.ckpt"
    base.mkdir()
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("x LAUNCHED pb" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("LAUNCHED pb s0 attempt 1" + NL)
    assert sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("")
    ck.write_bytes(b"x")
    assert sup.launched(str(base), str(ck))


# --------------------------------------------------------------------------
# main: the order, the guard before each launch, the child
# --------------------------------------------------------------------------
def test_main_source_order(sup):
    src = inspect.getsource(sup.main)
    assert src.index("other_supervisors(") < src.index("run_done(") < \
        src.index("abandoned(base)") < src.index("stray_snapshots(") < \
        src.index("stray_probe_rows(") < src.index("launched(base, ckpt)") \
        < src.index("wait_external(") < src.index("rows.prime()") < \
        src.index("preflight(variant)") < src.index("for attempt in range(")
    loop = src[src.index("for attempt in range("):]
    assert loop.index("wait_no_trainer(rows)") < loop.index(
        "manifest_check()") < loop.index("subprocess.Popen(")
    assert src.count("wait_no_trainer(") == 1
    assert src.count("wait_external(") == 1


class Dead:
    pid, returncode = 99, 1

    def poll(self):
        return 1


def _loop(sup, monkeypatch, events):
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: events.append("wait") or "ready")
    monkeypatch.setattr(sup, "preflight",
                        lambda v: events.append("preflight") or (True, ""))
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)


def test_every_launch_attempt_waits_for_no_trainer(sup, monkeypatch,
                                                   capsys):
    """Behavioural: the wait, the preflight, then the guard and the
    manifest before EVERY Popen; the child's recipe, cwd and env."""
    events, seen = [], []
    _loop(sup, monkeypatch, events)
    # the preflight's CPU-only env must never reach the trainer
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    ck = pathlib.Path(sup.paths("pb", 0)[3])

    def popen(args, **k):
        events.append("popen")
        seen.append((list(args), k))
        if len(seen) == 1:          # the first attempt leaves a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["pb", "0"]) == 1              # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == (["wait", "preflight"]
                      + ["guard", "manifest", "popen"] * 5)
    rel = sup.paths("pb", 0)[0]
    for i, (args, k) in enumerate(seen):
        assert args == sup.command("pb", 0) + (["--resume"] if i else [])
        assert k["cwd"] == sup.REPO
        assert k["env"]["ABLATION_RESULTS"] == rel
        assert k["env"]["PYTHONFAULTHANDLER"] == "1"
        assert "CUDA_VISIBLE_DEVICES" not in k["env"]
        assert k["creationflags"] == 0x08000000


def test_a_failing_manifest_blocks_the_launch(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    checks = [(False, "not the audited revision: x"),
              (False, "not the audited revision: x"), (True, "")]
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda *a, **k: events.append("popen") or _stop())
    with pytest.raises(Stop):
        sup.main(["pb", "0"])
    assert events[2:] == ["guard", "manifest"] * 3 + ["popen"]
    assert capsys.readouterr().out.count("not relaunching until") == 1


def test_wait_no_trainer_waits_out_any_trainer_and_streams_rows(
        sup, monkeypatch, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "pb", 0)
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
    assert "EVAL pb s0 iter 40" in out            # an orphan stays watched
    assert "ALERT: pb s0 row 40: reward_ceiling 99.0" in out


def _psutil(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


BS = chr(92)
GUARD_PROCS = [
    (1, "python.exe", ["py", "Y:" + BS + "run" + BS + "experiments" + BS
                       + "ablation.py", "--arms", PB_ARM]),
    (2, "python.exe", ["py", "experiments/ablation.py", "--arms", G0_ARM,
                       "--seed", "0"]),                       # G0 itself
    (3, "Python3.12", ["py", "experiments/ablation.py", "--arms", "pong-x"]),
    (4, "python.exe", ["py", "experiments" + BS + "distill.py",
                       "--master", "m.pth"]),                 # a prep
    (5, "python.exe", ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                       "x.pth"]),
    (6, "python.exe", ["py", "-m", "pytest", "tests/test_pbn.py"]),
    (7, "python.exe", None),                # unreadable: never counted
    (8, "Code.exe", ["code", "experiments/ablation.py"]),
    (9, "git.exe", ["git", "diff", "--", "experiments/distill.py"]),
    (10, None, ["experiments/ablation.py"]),
    (11, "python.exe", ["py", "tests/test_distill.py"]),
    (12, "python.exe", ["py", "ablation.py", "--arms", TK_ARM]),
    (13, "python.exe", ["py", "-m", "experiments.distill", "--master", "m"]),
    (14, "python.exe", ["py", "-m", "ablation"]),
    (15, "python.exe", ["py", "-m", "pytest", "ablation"]),
    (16, "python.exe", ["py", "experiments/ablation.pyc"]),
    (17, "python.exe", ["py", "benchmarks/supervisors/supervise_pb.py",
                        "pb", "0"]),
    (18, "python.exe", ["py", "-m", "distill"]),
    (19, "python.exe", ["py", "benchmarks/probes/probe_prefix_bn.py",
                        "x.pth"])]


def test_live_trainers_finds_any_run_or_prep(sup, g0sup, monkeypatch):
    """is_trainer as supervise_gen0.py's: the same verdict on every
    command line; G0's trainer and a prep count; python only; unreadable
    not counted."""
    fresh = _load(SUPERVISORS / "supervise_pb.py", "supervise_pb_procs")
    monkeypatch.setattr(sup, "_procs", fresh._procs)
    _psutil(monkeypatch, [{"pid": p, "name": n, "cmdline": c}
                          for p, n, c in GUARD_PROCS])
    assert sup.live_trainers() == [1, 2, 3, 4, 12, 13, 14, 18]
    for _p, _n, cmd in GUARD_PROCS:
        if cmd is not None:
            assert sup.is_trainer(cmd) == g0sup.is_trainer(cmd), cmd
    assert sup.TRAINER_SCRIPTS == g0sup.TRAINER_SCRIPTS
    assert sup.TRAINER_MODULES == g0sup.TRAINER_MODULES


def test_live_scripts_match_the_exact_name_and_variant(sup, monkeypatch):
    _procs(monkeypatch, sup, [
        (1, ["py", "C:" + BS + "x" + BS + "supervise_pb.py", "pb", "0"]),
        (2, ["py", "supervise_pb.py.bak", "pb", "0"]),
        (3, G0_SUPERVISOR),
        (4, ["py", "supervise_pb.py", "--write-manifest"]),
        (5, ["py", "experiments/ablation.py", "--arms", PB_ARM])])
    assert sup.live_scripts("supervise_pb.py") == [1, 4]
    assert sup.live_scripts("supervise_pb.py", "pb") == [1]
    assert sup.live_scripts("supervise_gen0.py", "g0") == [3]
    assert sup.live_trainers() == [5]
    monkeypatch.setattr(sup, "_mine", lambda: {1})
    assert sup.other_supervisors("pb") == []
    monkeypatch.setattr(sup, "_mine", lambda: {99})
    assert sup.other_supervisors("pb") == [1]


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    _procs(monkeypatch, sup, [
        (20, ["py", "benchmarks/supervisors/supervise_pb.py", "pb", "0"])])
    monkeypatch.setattr(sup, "_mine", lambda: {99})
    for name in ("run_done", "wait_external", "preflight"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["pb", "0"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_pb.py pb is alive (pids [20])" in out


def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys):
    def no(*a, **k):
        raise AssertionError("no wait and no preflight for a finished run")

    monkeypatch.setattr(sup, "wait_external", no)
    monkeypatch.setattr(sup, "preflight", no)
    ck = pathlib.Path(sup.paths("pb", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    assert sup.main(["pb", "0"]) == 0
    assert "RUN COMPLETE pb s0" in capsys.readouterr().out


def test_abandoned_and_stray_snapshots(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("must stop before any wait, preflight or "
                             "launch")

    rel, base, _j, ckpt, _d = sup.paths("pb", 0)
    base = pathlib.Path(base)
    snaps = base / "snapshots-x"
    snaps.mkdir(parents=True)
    (snaps / "weights-iter005.pth").write_bytes(b"x")
    for name in ("wait_external", "preflight", "wait_no_trainer"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["pb", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    # ABANDONED stops every launch, before any wait or preflight
    (base / "ABANDONED").write_text("")
    assert sup.main(["pb", "0"]) == 0
    assert "ABANDONED pb s0" in capsys.readouterr().out


def test_probe_rows_without_a_checkpoint_are_refused(sup, monkeypatch,
                                                     capsys):
    def no(*a, **k):
        raise AssertionError("must stop before any wait, preflight or "
                             "launch")

    for name in ("wait_external", "preflight", "wait_no_trainer"):
        monkeypatch.setattr(sup, name, no)
    rel, base, _j, ckpt, _d = sup.paths("pb", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    probes.mkdir(parents=True)
    f = probes / ("probe_exhaustive_eval-mspacman-tokens-rgrow-pbn-s0-"
                  "2026-10-25-acting.jsonl")
    f.write_text(json.dumps({"iteration": 5}) + NL)
    assert sup.stray_probe_rows("pb", 0, ckpt) == [f.name]
    assert sup.main(["pb", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    # TK-90's own probe jsonl (another run name) is no obstacle
    f.rename(probes / ("probe_exhaustive_eval-mspacman-tokens-rgrow-s0-"
                       "2026-10-19-acting.jsonl"))
    assert sup.stray_probe_rows("pb", 0, ckpt) == []
    pathlib.Path(base).mkdir(parents=True)
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)  # a legit resume
    f.write_text(json.dumps({"iteration": 5}) + NL)
    assert sup.stray_probe_rows("pb", 0, ckpt) == []
    f.write_text("")                                  # empty: not a record
    pathlib.Path(ckpt).unlink()
    assert sup.stray_probe_rows("pb", 0, ckpt) == []


def test_stray_state_rechecked_before_a_launch_without_checkpoint(
        sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"

    def old_rows_appear(*a):
        probes.mkdir(parents=True, exist_ok=True)
        (probes / ("probe_exhaustive_eval-mspacman-tokens-rgrow-pbn-s0-"
                   "2026-10-25.jsonl")).write_text("{}" + NL)

    monkeypatch.setattr(sup, "wait_no_trainer", old_rows_appear)
    assert sup.main(["pb", "0"]) == 2
    assert "stray snapshots / probe jsonls but no checkpoint" in \
        capsys.readouterr().out


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("pb", 0)[1])
    ckpt = pathlib.Path(sup.paths("pb", 0)[3])
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["pb", "0"]) == 0
    assert "ABANDONED pb s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["pb", "0"]) == 0
    assert "RUN COMPLETE pb s0" in capsys.readouterr().out


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


@pytest.mark.parametrize("life, stop, n", [
    (240, "CRASH LOOP", 5),          # quick deaths: CRASH LOOP's
    (480, "NO PROGRESS", 3),         # Test 1 audit r2 LOW-1: the old gap
    (1200, "NO PROGRESS", 3),
    (3 * 3600, "NO PROGRESS", 3)])
def test_every_repeating_death_without_a_checkpoint_stops(
        sup, monkeypatch, capsys, life, stop, n):
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
    assert sup.main(["pb", "0"]) == 1
    out = capsys.readouterr().out
    assert stop in out and "MAX ATTEMPTS" not in out
    assert len(launches) == n
    assert sup.STALL_MIN_S == 300 < 465


def test_a_checkpoint_resets_the_no_progress_count(sup, monkeypatch,
                                                   capsys):
    events = []
    _loop(sup, monkeypatch, events)
    clock = _clock(sup, monkeypatch)
    ck = pathlib.Path(sup.paths("pb", 0)[3])
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
    assert sup.main(["pb", "0"]) == 1
    out = capsys.readouterr().out
    assert "NO PROGRESS: 3 attempts in a row" in out
    assert "CRASH LOOP" not in out
    assert len(launches) == 5                # 1 stall, reset, then 3 stalls
    assert launches[2][-1] == "--resume" and "--resume" not in launches[0]
