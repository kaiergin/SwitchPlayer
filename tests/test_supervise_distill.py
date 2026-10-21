"""Curriculum P3's distillation step (Kai, 2026-10-21; CURRICULUM.md "P3's
distillation step"): supervise_distill.py runs DS then FS, after TKX. Pins
the recipe against TK's, each variant's ceiling, what ds waits for (TKX's
marker and evals) and fs waits for (DS's marker, its rows 0 and 5-90, the
prep), the master choice, the prep's once-only run and its retries, the
--no-wait refusal, the guard / manifest / prep-integrity order before each
launch, the stray-state rules (DS's own pre-checkpoint snapshots excepted),
the per-variant singleton and the row alerts. CPU only, no launches."""
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

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISORS = ROOT / "benchmarks" / "supervisors"
ENV_ID = "MsPacmanNoFrameskip-v4"


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sup():
    return _load(SUPERVISORS / "supervise_distill.py", "supervise_distill")


@pytest.fixture
def tk90():
    return _load(SUPERVISORS / "supervise_rgrow.py", "supervise_rgrow")


@pytest.fixture
def tkx():
    return _load(SUPERVISORS / "supervise_rgrowx.py", "supervise_rgrowx")


class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


# --------------------------------------------------------------------------
# the recipe
# --------------------------------------------------------------------------
def test_recipe_is_tks_with_the_student_inputs(sup, tk90):
    assert sup.VARIANTS == {"ds": TK_ARM + "-student-distill",
                            "fs": TK_ARM + "-student-fresh"}
    assert sup.SEEDS == {0: "2026-10-21"} and sup.ITERATIONS == 90
    base = tk90.command("tk", 0)[1:]
    vocab = ["--token-vocab", sup.PREP_REL + "/vocab.json"]
    dset = ["--distill-set", sup.PREP_REL + "/set.pkl"]
    for v, extra in (("ds", vocab + dset), ("fs", vocab)):
        cmd = sup.command(v, 0)
        assert cmd[0] == sup.PY
        arm_i = cmd.index("--arms") + 1
        assert cmd[arm_i] == sup.VARIANTS[v]
        want = list(base)
        want[want.index("--arms") + 1] = sup.VARIANTS[v]
        assert cmd[1:] == want + extra
    for v in ("ds", "fs"):
        assert sup.VARIANTS[v] in ablation.ARMS


def test_paths_and_the_run_folders(sup):
    for v, name in (("ds", "mspacman-student-distill"),
                    ("fs", "mspacman-student-fresh")):
        rel, base, jsonl, ckpt, deaths = sup.paths(v, 0)
        assert rel == f"benchmarks/results/{name}-s0-2026-10-21"
        assert jsonl.endswith(os.path.join(rel.split("/")[-1],
                                           f"ablation{sup.TAG}.jsonl"))
        assert ckpt.endswith(f"resume{sup.TAG}-{sup.VARIANTS[v]}-seed0.ckpt")
    assert sup.prep_dir() == os.path.join(sup.REPO,
                                          *sup.PREP_REL.split("/"))


def test_each_variants_ceiling_is_its_arms(sup):
    for v in ("ds", "fs"):
        with ablation.arm_config(sup.VARIANTS[v]), \
                ablation._run_length_globals(90, 2000):
            for it in range(1, 91):
                assert sup.expected_ceiling(v, it) == pytest.approx(
                    muzero.reward_ceiling((it - 1) * 2000), rel=1e-9)
    assert sup.expected_ceiling("ds", 1) == 160.0
    assert sup.expected_ceiling("fs", 1) == 1.0


def test_supervisor_pins_what_the_runs_and_preflight_use(sup, tkx):
    import re
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert set(tkx.SUITES) <= set(sup.SUITES)
    assert set(tkx.AUDITED) <= set(sup.AUDITED)
    assert set(sup.SUITES) - set(tkx.SUITES) == {
        "tests/test_supervise_distill.py", "tests/test_distill.py"}
    assert set(sup.AUDITED) - set(tkx.AUDITED) == {
        "tests/test_supervise_distill.py", "tests/test_distill.py",
        "experiments/distill.py",
        "benchmarks/supervisors/supervise_distill.py"}
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)


def test_supervisor_manifest_catches_an_edit(sup, monkeypatch, tmp_path):
    import shutil
    for rel in sup.AUDITED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tmp_path))))
    assert sup.manifest_check() == (True, "")
    for rel in ("experiments/distill.py",
                "benchmarks/supervisors/supervise_distill.py"):
        p = tmp_path / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# edit\n")
        ok, why = sup.manifest_check()
        assert not ok and rel in why
        p.write_bytes(orig.replace(b"\n", b"\r\n"))    # CRLF vs LF: same
        assert sup.manifest_check() == (True, "")


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
    for argv in ([], ["ds"], ["0"], ["ds", "1"], ["tk", "0"], ["tkx", "0"],
                 ["ds", "x"], ["ds", "0", "1"], ["0", "ds"],
                 ["ds", "0", "--wait"], ["ds", "0", "--no-wait", "x"],
                 ["fs", "0", "--no-wait"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


@pytest.mark.parametrize("variant", ["ds", "fs"])
def test_supervisor_quick_probe_passes(sup, variant):
    import subprocess
    r = subprocess.run([sys.executable, "-c", sup.quick(variant)], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------
# the predecessors, the master
# --------------------------------------------------------------------------
def test_tkx_constants_name_the_live_tkx_run(sup, tkx, monkeypatch):
    monkeypatch.setattr(tkx, "REPO", sup.TKX_TREE)
    rel, base, _jsonl, ckpt, _d = tkx.paths("tkx", 0)
    assert sup.TKX_RUN == rel.split("/")[-1]
    assert sup.TKX_MARKER == ckpt
    assert sup.TKX_SUPLOG == os.path.join(base, "supervisor.log")
    assert sup.TKX_SNAPSHOT == os.path.join(
        base, f"snapshots{sup.TAG}-{TK_ARM}-seed0", "weights-iter270.pth")
    assert sup.TKX_ROWS == tuple(range(5, 275, 5))
    assert sup.TK90_SNAPSHOT.endswith(os.path.join(
        "mspacman-tokens-rgrow-s0-2026-10-19",
        f"snapshots{sup.TAG}-{TK_ARM}-seed0", "weights-iter090.pth"))
    assert pathlib.Path(sup.tk90_probe()).exists()   # committed on the branch
    assert sup.DS_ROWS == (0,) + tuple(range(5, 95, 5))


def _rows(path, iterations, value=500.0, extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for i in iterations:
            f.write(json.dumps({"iteration": i, "env": ENV_ID,
                                "exact_mean": value}) + chr(10))
        for r in extra:
            f.write(json.dumps(r) + chr(10))


@pytest.fixture
def fake_tkx(sup, monkeypatch, tmp_path):
    marker = tmp_path / "tkx" / "resume.ckpt"
    marker.parent.mkdir()
    probes = (tmp_path / "tkx" / "p-acting.jsonl",
              tmp_path / "tkx" / "p.jsonl")
    monkeypatch.setattr(sup, "TKX_MARKER", str(marker))
    monkeypatch.setattr(sup, "TKX_PROBES", tuple(map(str, probes)))
    monkeypatch.setattr(sup, "TKX_SUPLOG", str(tmp_path / "tkx" / "sup.log"))
    monkeypatch.setattr(sup, "REPO", str(tmp_path / "repo"))
    return marker, probes


def test_ds_waits_for_tkxs_marker_and_every_row_of_both_evals(sup, fake_tkx):
    marker, probes = fake_tkx
    assert any("no checkpoint" in m for m in sup.pred_missing("ds", 0))
    marker.write_bytes(b"x" * 1_000_000)
    assert any("full size" in m for m in sup.pred_missing("ds", 0))
    marker.write_bytes(b"done")
    _rows(probes[0], range(5, 275, 5))
    _rows(probes[1], range(5, 270, 5), extra=[
        {"iteration": 270, "env": ENV_ID, "max_noop": 2},   # a smoke row
        {"iteration": 270, "env": "Pong-v4"}])
    miss = sup.pred_missing("ds", 0)
    assert miss == ["p.jsonl rows [270]"]
    _rows(probes[1], [270])
    assert sup.pred_missing("ds", 0) == []


def test_fs_waits_for_ds_its_rows_its_curve_and_the_prep(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    ck = pathlib.Path(sup.paths("ds", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    run = sup.run_name("ds", 0)
    probes = tmp_path / "benchmarks" / "probes"
    _rows(probes / f"probe_exhaustive_eval-{run}-acting.jsonl",
          range(5, 95, 5))
    _rows(probes / f"probe_exhaustive_eval-{run}.jsonl", (0,) + tuple(
        range(5, 95, 5)))
    _rows(probes / f"probe_exhaustive_eval-{run}-offline-acting.jsonl",
          [1, 2, 3, 4])
    _rows(probes / f"probe_exhaustive_eval-{run}-offline.jsonl", [1, 2, 4])
    miss = sup.pred_missing("fs", 0)
    assert any("-acting.jsonl rows [0]" in m for m in miss)
    assert any("-offline.jsonl rows [3]" in m for m in miss)   # r1 B-L2
    assert any("vocab.json" in m for m in miss)
    _rows(probes / f"probe_exhaustive_eval-{run}-acting.jsonl", [0])
    _rows(probes / f"probe_exhaustive_eval-{run}-offline.jsonl", [3])
    os.makedirs(sup.prep_dir())
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text("{}")
    assert sup.pred_missing("fs", 0) == []


def test_wait_external_blocks_and_alerts(sup, fake_tkx, monkeypatch, capsys,
                                         tmp_path):
    marker, probes = fake_tkx
    marker.write_bytes(b"done")
    os.utime(marker, (1, 1))                           # done long ago
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [])
    tk90 = tmp_path / "tk90.pth"
    monkeypatch.setattr(sup, "TK90_SNAPSHOT", str(tk90))
    monkeypatch.setattr(sup, "TKX_SNAPSHOT", str(tmp_path / "tkx.pth"))
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            _rows(probes[0], range(5, 275, 5))
            _rows(probes[1], range(5, 275, 5))

    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("ds", 0) is True
    out = capsys.readouterr().out
    assert "WAITING for TKX before ds s0" in out
    assert "ALERT: TKX completed" in out and "evals still lack rows" in out
    assert "TKX complete and evaluated; ds s0 may start" in out
    # the masters' snapshots are checked while waiting (r1 B-L10)
    assert "master snapshot(s) missing" in out and str(tk90) in out
    assert str(tmp_path / "tkx.pth") in out
    marker.write_bytes(b"x" * 1_000_000)               # unfinished, undriven
    probes[0].unlink()
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("ds", 0)
    out = capsys.readouterr().out
    assert "nothing drives it: no live supervise_rgrowx.py" in out
    assert "master snapshot(s) missing: " + str(tk90) in out
    # an ABANDONED predecessor is named (r1 B-L8)
    abandoned = pathlib.Path(sup.TKX_MARKER).parent / "ABANDONED"
    monkeypatch.setattr(sup, "TKX_ABANDONED", str(abandoned))
    abandoned.write_text("")
    with pytest.raises(Stop):
        sup.wait_external("ds", 0)
    assert "TKX is ABANDONED" in capsys.readouterr().out


def test_wait_external_stops_when_this_run_is_abandoned(sup, fake_tkx,
                                                        monkeypatch):
    base = pathlib.Path(sup.paths("ds", 0)[1])
    base.mkdir(parents=True)

    def sleep(s):
        (base / "ABANDONED").write_text("")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [1])
    assert sup.wait_external("ds", 0) is False      # r1 B-L9


def _probe_file(path, rows, value):
    _rows(path, rows, value=value)


def test_choose_master_picks_the_better_7_row_window(sup, fake_tkx,
                                                     monkeypatch, tmp_path):
    marker, probes = fake_tkx
    tk90p = tmp_path / "tk90-acting.jsonl"
    monkeypatch.setattr(sup, "tk90_probe", lambda: str(tk90p))
    with pytest.raises(SystemExit, match="TK-90"):
        sup.choose_master()
    _probe_file(tk90p, range(5, 95, 5), 856.62)
    # TKX without its marker (abandoned / --no-wait): TK-90, whatever its rows
    _probe_file(probes[0], range(5, 275, 5), 1000.0)
    pick = sup.choose_master()
    assert pick["which"] == "TK-90" and pick["iterations"] == 90
    assert pick["snapshot"] == sup.TK90_SNAPSHOT
    assert pick["tkx_mean_240_270"] is None
    marker.write_bytes(b"done")
    pick = sup.choose_master()
    assert pick["which"] == "TKX" and pick["iterations"] == 270
    assert pick["snapshot"] == sup.TKX_SNAPSHOT
    assert pick["tkx_mean_240_270"] == 1000.0
    assert pick["tk90_mean_60_90"] == pytest.approx(856.62)
    probes[0].unlink()
    _probe_file(probes[0], range(5, 275, 5), 800.0)    # TKX worse
    assert sup.choose_master()["which"] == "TK-90"
    probes[0].unlink()
    tk90_mean = sup.window_mean(sup.probe_rows(str(tk90p)), 60, 90)
    _probe_file(probes[0], range(5, 275, 5), tk90_mean)   # a tie: TK-90
    assert sup.choose_master()["which"] == "TK-90"
    probes[0].unlink()
    # completed but rows 240-270 incomplete: refused, never TK-90 (r1 B-L7)
    _probe_file(probes[0], range(5, 270, 5), 2000.0)
    with pytest.raises(SystemExit, match="rows 240-270 are incomplete"):
        sup.choose_master()


def test_master_files_missing(sup, fake_tkx, monkeypatch, tmp_path):
    marker, _ = fake_tkx
    a, b = tmp_path / "tk90.pth", tmp_path / "tkx.pth"
    monkeypatch.setattr(sup, "TK90_SNAPSHOT", str(a))
    monkeypatch.setattr(sup, "TKX_SNAPSHOT", str(b))
    assert sup.master_files_missing() == [str(a)]     # TKX unfinished
    marker.write_bytes(b"done")
    assert sup.master_files_missing() == [str(a), str(b)]
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    assert sup.master_files_missing() == []


def test_tk90s_committed_window_is_the_preregistered_856(sup):
    rows = sup.probe_rows(sup.tk90_probe())
    assert sup.window_mean(rows, 60, 90) == pytest.approx(856.62, abs=0.01)


# --------------------------------------------------------------------------
# the prep
# --------------------------------------------------------------------------
@pytest.fixture
def prep_env(sup, monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    pick = {"which": "TK-90", "snapshot": "m.pth", "iterations": 90,
            "tkx_mean_240_270": None, "tk90_mean_60_90": 856.62}
    calls = []
    monkeypatch.setattr(sup, "choose_master",
                        lambda: calls.append("choose") or dict(pick))
    return calls


def _write_prep(sup, good=True):
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True, exist_ok=True)
    (d / "set.pkl").write_bytes(b"set")
    sha = hashlib.sha256(b"set" if good else b"other").hexdigest()
    (d / "vocab.json").write_text(json.dumps({"set_sha256": sha}))


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
    effect)."""
    def popen(args, **k):
        if seen is not None:
            seen.append(args)
        rc, effect = plan.pop(0)
        return FakeProc(rc, effect)

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup.subprocess, "run", _stop)


def test_the_prep_runs_once_and_keeps_its_first_master(sup, prep_env,
                                                      monkeypatch, capsys):
    calls = prep_env
    runs = []
    _popen(monkeypatch, sup, [(1, None), (0, lambda: _write_prep(sup))], runs)
    assert sup.run_prep({}) is True
    out = capsys.readouterr().out
    assert "MASTER TK-90" in out and "PREP attempt 1 failed" in out
    assert "PREP done (attempt 2)" in out
    assert calls == ["choose"] and len(runs) == 2
    assert runs[0][1:] == ["experiments/distill.py", "--master", "m.pth",
                           "--master-iterations", "90", "--out",
                           sup.PREP_REL, "--seed", "0"]
    master = json.loads(pathlib.Path(sup.prep_dir(), "master.json")
                        .read_text())
    assert master["which"] == "TK-90"
    assert not list(pathlib.Path(sup.prep_dir()).glob("*.tmp"))
    assert sup.run_prep({}) is True                  # done: nothing runs
    assert len(runs) == 2 and calls == ["choose"]


def test_a_restarted_prep_reuses_the_logged_master(sup, prep_env,
                                                   monkeypatch):
    calls = prep_env
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "master.json").write_text(json.dumps({
        "which": "TKX", "snapshot": "x.pth", "iterations": 270,
        "tkx_mean_240_270": 1.0, "tk90_mean_60_90": 0.0}))
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))], seen)
    assert sup.run_prep({}) is True
    assert calls == [] and seen[0][3] == "x.pth" and seen[0][5] == "270"


def test_an_unreadable_master_json_is_chosen_again(sup, prep_env,
                                                   monkeypatch):
    calls = prep_env
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "master.json").write_text('{"which": "TKX", "snap')   # torn (r1 L6)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep({}) is True and calls == ["choose"]
    assert sup.read_pick(str(d / "master.json"))["which"] == "TK-90"


def test_a_master_rule_failure_is_prep_failed(sup, prep_env, monkeypatch,
                                              capsys):
    monkeypatch.setattr(sup, "choose_master", lambda: (_ for _ in ()).throw(
        SystemExit("TKX completed but its deployed rows 240-270 are "
                   "incomplete")))
    _popen(monkeypatch, sup, [])
    assert sup.run_prep({}) is False
    assert "PREP FAILED: TKX completed" in capsys.readouterr().out


def test_the_prep_gives_up_after_its_attempts(sup, prep_env, monkeypatch,
                                              capsys):
    _popen(monkeypatch, sup, [(1, None)] * 3)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert out.count("failed (rc=1") == sup.PREP_ATTEMPTS == 3
    assert "PREP FAILED after 3 attempts" in out


def test_a_silent_prep_is_an_alert(sup, prep_env, monkeypatch, capsys):
    class Slow(FakeProc):
        polls = 0

        def poll(self):
            Slow.polls += 1
            return None if Slow.polls < 3 else self.returncode

    import time as real_time
    clock = [real_time.time()]      # prep.log's real mtime is "now"
    monkeypatch.setattr(sup, "time", SimpleNamespace(
        time=lambda: clock[0], sleep=None, strftime=real_time.strftime))

    def sleep(s):
        clock[0] += 3600

    sup.time.sleep = sleep
    monkeypatch.setattr(sup.subprocess, "Popen", lambda args, **k: Slow(
        0, lambda: _write_prep(sup)))
    assert sup.run_prep({}) is True
    assert "ALERT: the prep (pid 4321) has been silent" in \
        capsys.readouterr().out


def test_a_damaged_prep_is_moved_aside_and_redone(sup, prep_env, monkeypatch,
                                                  capsys):
    _write_prep(sup, good=False)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep({}) is True
    out = capsys.readouterr().out
    assert "PREP damaged" in out and "moved aside" in out
    d = pathlib.Path(sup.prep_dir())
    assert len(list(d.glob("set.pkl.damaged-*"))) == 1
    assert len(list(d.glob("vocab.json.damaged-*"))) == 1
    assert sup.set_intact() == (True, "")


def test_a_prep_that_never_checks_out_fails(sup, prep_env, monkeypatch,
                                            capsys):
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup, good=False))] * 3)
    assert sup.run_prep({}) is False
    assert "does not match" in capsys.readouterr().out


def test_a_sanctioned_run_never_redoes_its_prep(sup, prep_env, monkeypatch,
                                                capsys):
    """r1 B-M2: a launched DS is bound to the set; a transient lock or real
    damage blocks the launch (the integrity gate), never a redo."""
    calls = prep_env
    _write_prep(sup, good=False)
    _popen(monkeypatch, sup, [])                     # any launch fails
    assert sup.run_prep({}, redo_ok=False) is True
    out = capsys.readouterr().out
    assert "never redone" in out and calls == []
    assert not list(pathlib.Path(sup.prep_dir()).glob("*.damaged-*"))
    # a transient read failure that clears is not damage
    _write_prep(sup)
    real = sup.set_intact
    flaky = [(False, "prep files: [Errno 13] locked"), None]

    def once():
        r = flaky.pop(0)
        return r if r else real()

    monkeypatch.setattr(sup, "set_intact", once)
    assert sup.run_prep({}) is True
    assert "damaged" not in capsys.readouterr().out


def test_the_prep_needs_the_audited_tree(sup, prep_env, monkeypatch, capsys):
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: (False, "not the audited revision: x"))
    _popen(monkeypatch, sup, [])
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    assert "PREP FAILED: not the audited revision" in capsys.readouterr().out


def test_fs_launches_only_on_dss_vocabulary(sup, monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    os.makedirs(sup.prep_dir())
    vocab = [[0, 4], [1, 4], [1, 16]]
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text(
        json.dumps({"vocab": vocab}))
    ck = pathlib.Path(sup.paths("ds", 0)[3])
    ck.parent.mkdir(parents=True)
    ok, why = sup.vocab_is_ds(0)
    assert not ok and "vocab check" in why            # no marker yet
    torch.save({"completed": True, "token_vocab": vocab}, ck)
    assert sup.vocab_is_ds(0) == (True, "")
    torch.save({"completed": True, "token_vocab": vocab[:2]}, ck)
    ok, why = sup.vocab_is_ds(0)
    assert not ok and "not the vocabulary DS trained with" in why


def test_the_trainer_guard_counts_the_prep(sup, monkeypatch):
    _procs(monkeypatch, sup, [
        (1, ["py", r"Y:\x\experiments\distill.py", "--master", "m"]),
        (2, ["py", "-m", "pytest", "tests/test_distill.py"]),
        (3, ["py", "experiments/ablation.py", "--arms", "a"])])
    assert sup.live_trainers() == [1, 3]               # r1 B-M1


def _procs(monkeypatch, sup, procs):
    monkeypatch.setattr(sup, "_procs", lambda: iter(procs))


def test_live_scripts_match_the_exact_name_and_variant(sup, monkeypatch):
    procs = [(1, ["py", r"C:\x\supervise_distill.py", "ds", "0"]),
             (2, ["py", "benchmarks/supervisors/supervise_distill.py", "fs",
                  "0"]),
             (3, ["py", "supervise_distill.py.bak", "ds", "0"]),
             (4, ["py", "post_rgrowx.py"]),
             (5, ["py", "post_rgrow.py"]),
             (6, ["py", "experiments/ablation.py", "--arms", "x"])]
    _procs(monkeypatch, sup, procs)
    assert sup.live_scripts("supervise_distill.py") == [1, 2]
    assert sup.live_scripts("supervise_distill.py", "ds") == [1]
    assert sup.live_scripts("supervise_distill.py", "fs") == [2]
    assert sup.live_scripts("post_rgrowx.py") == [4]
    assert sup.live_trainers() == [6]
    monkeypatch.setattr(sup, "_mine", lambda: {1})
    assert sup.other_supervisors("ds") == []
    assert sup.other_supervisors("fs") == [2]


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [77])
    monkeypatch.setattr(sup, "run_done", _stop)
    assert sup.main(["ds", "0"]) == 2
    assert "another supervise_distill.py ds is alive" in \
        capsys.readouterr().out


def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           tmp_path, capsys):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "wait_external", _stop)
    ck = pathlib.Path(sup.paths("fs", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    assert sup.main(["fs", "0"]) == 0
    assert "RUN COMPLETE fs s0" in capsys.readouterr().out


def test_no_wait_is_logged_and_refused_while_tkx_is_driven(sup, monkeypatch,
                                                            tmp_path, capsys):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "wait_external", _stop)
    marker = tmp_path / "tkx" / "resume.ckpt"
    marker.parent.mkdir()
    monkeypatch.setattr(sup, "TKX_MARKER", str(marker))
    monkeypatch.setattr(sup, "TKX_ABANDONED", str(tmp_path / "tkx" / "AB"))
    alive = {"supervise_rgrowx.py": [5], "post_rgrowx.py": []}
    monkeypatch.setattr(sup, "live_scripts", lambda s, v=None: alive[s])
    assert sup.main(["ds", "0", "--no-wait"]) == 2
    assert "supervise_rgrowx.py is alive" in capsys.readouterr().out
    alive.update({"supervise_rgrowx.py": [], "post_rgrowx.py": [6]})
    assert sup.main(["ds", "0", "--no-wait"]) == 2
    assert "post_rgrowx.py is alive" in capsys.readouterr().out
    alive["post_rgrowx.py"] = []
    # neither TKX's marker nor its ABANDONED file (r1 B-L11)
    assert sup.main(["ds", "0", "--no-wait"]) == 2
    assert "neither its completion marker nor an ABANDONED" in \
        capsys.readouterr().out
    (tmp_path / "tkx" / "AB").write_text("")
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["ds", "0", "--no-wait"])
    assert "NO-WAIT override: ds s0" in capsys.readouterr().out


def test_a_resume_skips_the_wait(sup, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "wait_external",
                        lambda *a: (_ for _ in ()).throw(
                            AssertionError("a resume must not wait")))
    monkeypatch.setattr(sup, "preflight", _stop)
    ck = pathlib.Path(sup.paths("ds", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)
    with pytest.raises(Stop):
        sup.main(["ds", "0"])
    assert "RESUMING ds s0" in capsys.readouterr().out
    ck.unlink()
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["ds", "0"])


def _loop(sup, monkeypatch, tmp_path, events):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: events.append("wait") or True)
    monkeypatch.setattr(sup, "preflight",
                        lambda v: events.append("preflight") or (True, ""))
    monkeypatch.setattr(sup, "run_prep",
                        lambda env, redo_ok: events.append(("prep", redo_ok))
                        and False or True)
    monkeypatch.setattr(sup, "vocab_is_ds",
                        lambda seed: events.append("vocab") or (True, ""))
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or (True, ""))
    monkeypatch.setattr(sup, "set_intact",
                        lambda: events.append("intact") or (True, ""))
    monkeypatch.setattr(sup, "snapshot", lambda *a: None)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)


def test_ds_runs_wait_preflight_prep_then_each_launch_guarded(
        sup, monkeypatch, tmp_path, capsys):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)

    class Dead:
        pid, returncode = 99, 1

        def poll(self):
            return 1

    launched = []

    def popen(args, **k):
        events.append("popen")
        launched.append((args, k["env"]))
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["ds", "0"]) == 1                 # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == (["wait", "preflight", ("prep", True)]
                      + ["guard", "manifest", "intact", "popen"] * 5)
    args, env = launched[0]
    assert args == sup.command("ds", 0)
    assert env["ABLATION_RESULTS"] == sup.paths("ds", 0)[0]
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_fs_has_no_prep_and_no_set_check(sup, monkeypatch, tmp_path):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)

    def popen(args, **k):
        events.append("popen")
        raise Stop

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        sup.main(["fs", "0"])
    assert events == ["wait", "preflight", "guard", "manifest", "vocab",
                      "popen"]


def test_a_failed_prep_launches_nothing(sup, monkeypatch, tmp_path):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    monkeypatch.setattr(sup, "run_prep", lambda env, redo_ok: False)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["ds", "0"]) == 1


def test_a_broken_prep_blocks_the_launch_until_intact(sup, monkeypatch,
                                                      tmp_path, capsys):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    checks = [(False, "set.pkl does not match vocab.json's set_sha256"),
              (True, "")]
    monkeypatch.setattr(sup, "set_intact",
                        lambda: events.append("intact") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    with pytest.raises(Stop):
        sup.main(["ds", "0"])
    assert events[3:] == ["guard", "manifest", "intact"] * 2
    assert "not relaunching until" in capsys.readouterr().out


def test_stray_state(sup, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "wait_external", _stop)
    rel, base, _j, ckpt, _d = sup.paths("ds", 0)
    base = pathlib.Path(base)
    off = base / f"{base.name}-offline" / "snapshots-x"
    off.mkdir(parents=True)
    (off / "weights-iter001.pth").write_bytes(b"x")
    own = base / "snapshots-x"
    own.mkdir()
    (own / "weights-iter000.pth").write_bytes(b"x")
    assert len(sup.stray_snapshots(str(base), ckpt)) == 2
    assert sup.main(["ds", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    # DS's own death before its first checkpoint: sanctioned, and exactly
    # row 0 + the offline curve excused (r1 B-L4)
    (base / "supervisor.log").write_text("LAUNCHED ds s0 attempt 1 (fresh)\n")
    assert sup.stray_snapshots(str(base), ckpt, own_ok=True) == []
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["ds", "0"])
    (own / "weights-iter005.pth").write_bytes(b"x")  # a later row: stray
    assert sup.main(["ds", "0"]) == 2
    (own / "weights-iter005.pth").unlink()
    # fs never gets the excuse
    _r, fbase, _fj, fck, _fd = sup.paths("fs", 0)
    fsnap = pathlib.Path(fbase) / "snapshots-x"
    fsnap.mkdir(parents=True)
    (fsnap / "weights-iter000.pth").write_bytes(b"x")
    (pathlib.Path(fbase) / "supervisor.log").write_text("LAUNCHED fs\n")
    assert sup.main(["fs", "0"]) == 2
    capsys.readouterr()
    # probe rows without a checkpoint: refused even when sanctioned
    probes = tmp_path / "benchmarks" / "probes"
    _rows(probes / f"probe_exhaustive_eval-{base.name}-offline-acting.jsonl",
          [1])
    assert sup.main(["ds", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    assert sup.stray_probe_rows("ds", 0, ckpt) == []


def test_a_resumed_ds_never_redoes_its_prep(sup, monkeypatch, tmp_path):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    ck = pathlib.Path(sup.paths("ds", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)                  # a resume
    with pytest.raises(Stop):
        sup.main(["ds", "0"])
    assert ("prep", False) in events


def test_an_abandoned_ds_runs_no_prep(sup, monkeypatch, tmp_path, capsys):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    base = pathlib.Path(sup.paths("ds", 0)[1])

    def preflight(v):
        base.mkdir(parents=True, exist_ok=True)
        (base / "ABANDONED").write_text("")
        return True, ""

    monkeypatch.setattr(sup, "preflight", preflight)
    assert sup.main(["ds", "0"]) == 0
    assert not any(isinstance(e, tuple) for e in events)
    assert "no prep" in capsys.readouterr().out


def test_main_returns_when_abandoned_while_waiting(sup, monkeypatch,
                                                   tmp_path, capsys):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    monkeypatch.setattr(sup, "wait_external", lambda v, s: False)
    monkeypatch.setattr(sup, "preflight", _stop)
    assert sup.main(["ds", "0"]) == 0
    assert "ABANDONED ds s0" in capsys.readouterr().out


def test_the_attempt_loop_refuses_stray_probe_rows(sup, monkeypatch,
                                                   tmp_path, capsys):
    events = []
    _loop(sup, monkeypatch, tmp_path, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    run = sup.run_name("fs", 0)
    probes = tmp_path / "benchmarks" / "probes"

    def rows_appear(*a):
        _rows(probes / f"probe_exhaustive_eval-{run}.jsonl", [5])

    monkeypatch.setattr(sup, "wait_no_trainer", rows_appear)
    assert sup.main(["fs", "0"]) == 2
    assert "stray snapshots / probe jsonls" in capsys.readouterr().out


def test_main_source_order(sup):
    src = inspect.getsource(sup.main)
    assert src.index("rows.prime()") < src.index("preflight(variant)")
    assert src.index("preflight(variant)") < src.index("run_prep(")
    assert src.index("run_prep(") < src.index("for attempt in range(")
    loop = src[src.index("for attempt in range("):]
    assert loop.index("wait_no_trainer(rows)") < loop.index(
        "manifest_check()") < loop.index("set_intact()") < loop.index(
        "vocab_is_ds(seed)") < loop.index("subprocess.Popen(")
    assert "redo_ok=not sanctioned(base, ckpt)" in src


@pytest.mark.parametrize("variant, row, per, want", [
    ("ds", {"iteration": 5, "reward_ceiling": 160.0}, "10.0", []),
    ("ds", {"iteration": 5, "reward_ceiling": 5.0}, "10.0", ["reward_ceiling"]),
    ("fs", {"iteration": 1, "reward_ceiling": 1.0}, "n/a", []),
    ("fs", {"iteration": 46, "reward_ceiling": 160.0}, "30.0", ["min/iteration"]),
    ("fs", {"iteration": 50, "reward_ceiling": 160.0, "reward_max": 401},
     "10", ["reward_max"]),
])
def test_row_alerts(sup, variant, row, per, want):
    got = sup.row_alerts(variant, row, per)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert w in g


def test_rows_print_and_alert(sup, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.0)
    j = tmp_path / "a.jsonl"
    rows = sup.Rows(str(j), "ds", 0)
    rows.prime()
    j.write_text(json.dumps({"iteration": 5, "reward_ceiling": 160.0,
                             "elapsed_s": 600, "return_mean": 700,
                             "tok_frames_per_decision": 12}) + chr(10))
    rows.emit()
    out = capsys.readouterr().out
    assert "EVAL ds s0 iter 5" in out and "frames/decision 12" in out
    assert "host commit free 3.0 GB" in out
