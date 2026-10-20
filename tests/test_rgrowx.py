"""The token arm on Ms. Pac-Man, 3x longer (Kai, 2026-10-20; CURRICULUM.md
"Next experiment (pre-registered) -- the token arm on Ms. Pac-Man, 3x
longer"): supervise_rgrowx.py runs TK unchanged for 270 iterations, its
schedules stretched by --iterations alone. Pins the recipe against TK's,
the stretched schedules against the pre-registration, the wait for RG (its
marker and its evals), the --no-wait refusal, the any-trainer guard, the
manifest and the row alerts. CPU only, no launches."""
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
RG_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISORS = ROOT / "benchmarks" / "supervisors"


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sup():
    return _load(SUPERVISORS / "supervise_rgrowx.py", "supervise_rgrowx")


@pytest.fixture
def tk90():
    return _load(SUPERVISORS / "supervise_rgrow.py", "supervise_rgrow")


@pytest.fixture
def stretched():
    """The arm's config with the run-length globals of 270 x 2000."""
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(270, 2000):
        yield


# --------------------------------------------------------------------------
# the recipe and the schedules
# --------------------------------------------------------------------------
def test_recipe_is_tks_with_270_iterations(sup, tk90):
    assert sup.VARIANTS == {"tkx": TK_ARM} and tk90.VARIANTS["tk"] == TK_ARM
    assert sup.SEEDS == {0: "2026-10-20"}
    x, t = sup.command("tkx", 0), tk90.command("tk", 0)
    i = x.index("--iterations")
    assert t.index("--iterations") == i
    assert x[i + 1] == "270" and t[i + 1] == "90"
    assert x[1:i + 1] + x[i + 2:] == t[1:i + 1] + t[i + 2:]
    assert x[0] == t[0]                              # the same interpreter
    assert x[1:5] == ["experiments/ablation.py", "--arms", TK_ARM, "--seed"]
    assert "--resume" not in x


def test_paths_and_the_run_folder(sup):
    rel, base, jsonl, ckpt, deaths = sup.paths("tkx", 0)
    assert rel == "benchmarks/results/mspacman-tokens-rgrow-x270-s0-2026-10-20"
    assert jsonl.endswith("ablation-mspacmannoframeskip-skip4.jsonl")
    assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{TK_ARM}-seed0.ckpt")
    assert deaths.endswith("deaths.log")
    assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT          # runs from its own tree
    assert pathlib.Path(sup.MANIFEST).name == "rgrowx_manifest.json"
    assert sup.MAX_ATTEMPTS == 80


def test_the_supervisors_ceiling_is_the_arms_at_270(sup, stretched):
    assert muzero.TEMPERATURE_TOTAL_STEPS == 540_000
    for it in range(1, 271):
        want = muzero.reward_ceiling((it - 1) * 2000)
        assert sup.expected_ceiling(it) == pytest.approx(want, rel=1e-12), it


def test_the_stretched_schedules_match_the_preregistration(stretched):
    at = lambda it: (it - 1) * 2000                  # noqa: E731
    # the ceiling: 1 -> 160 log-linear, 160 from iteration 136
    assert muzero.reward_ceiling(at(1)) == 1.0
    assert muzero.reward_ceiling(at(45)) == pytest.approx(5.229, abs=1e-3)
    assert muzero.reward_ceiling(at(68)) == pytest.approx(12.414, abs=1e-3)
    assert muzero.reward_ceiling(at(90)) == pytest.approx(28.385, abs=1e-3)
    assert muzero.reward_ceiling(at(135)) == pytest.approx(154.097, abs=1e-3)
    assert muzero.reward_ceiling(at(135)) < 160.0
    assert muzero.reward_ceiling(at(136)) == 160.0
    assert muzero.reward_ceiling(at(270)) == 160.0
    # the long-token bias: 0.5 -> 0, zero from iteration 69
    assert muzero.token_long_bias(at(1)) == 0.5
    assert muzero.token_long_bias(at(68)) > 0.0
    assert muzero.token_long_bias(at(69)) == 0.0
    # prior-acting temperature: 0.5 x the staged 1 / 0.5 / 0.25
    assert muzero.prior_acting_temperature(at(1)) == 0.5
    assert muzero.prior_acting_temperature(at(135)) == 0.5
    assert muzero.prior_acting_temperature(at(136)) == 0.25
    assert muzero.prior_acting_temperature(at(203)) == 0.25
    assert muzero.prior_acting_temperature(at(204)) == 0.125
    # PER beta over the whole run; the off-policy horizon decay
    assert muzero.per_beta(0) == pytest.approx(0.4)
    assert muzero.per_beta(540_000) == pytest.approx(1.0)
    assert muzero.OFFPOLICY_DECAY_STEPS == 162_000


def test_what_the_stretch_leaves_unchanged(stretched):
    assert muzero.MAX_REPLAY_TRANSITIONS == 31_250
    assert muzero.LR_DECAY_AT is None              # no decay, as in TK-90
    assert muzero.LR_WARMUP_STEPS == 1000
    assert muzero.LEARNING_RATE == 0.2
    assert muzero.VTREE_REFRESH_BATCH == 64
    assert muzero.VTREE_COMMUNAL_TRAJS == 1
    assert len(muzero.TOKEN_VOCAB) == 36
    assert muzero.REWARD_GROWTH is True


def test_tk90s_landmarks_for_the_comparison(tk90):
    """The pre-registration's TK-90 comparison numbers: 160 at iteration
    46, the bias zero from 24, the temperature stages at 46 and 69."""
    at = lambda it: (it - 1) * 2000                  # noqa: E731
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(90, 2000):
        assert muzero.reward_ceiling(at(45)) < 160.0
        assert muzero.reward_ceiling(at(46)) == 160.0
        assert muzero.token_long_bias(at(23)) > 0.0
        assert muzero.token_long_bias(at(24)) == 0.0
        assert muzero.prior_acting_temperature(at(45)) == 0.5
        assert muzero.prior_acting_temperature(at(46)) == 0.25
        assert muzero.prior_acting_temperature(at(68)) == 0.25
        assert muzero.prior_acting_temperature(at(69)) == 0.125
        for it in range(1, 91):
            assert tk90.expected_ceiling(it) == pytest.approx(
                muzero.reward_ceiling(at(it)), rel=1e-12)


# --------------------------------------------------------------------------
# what the supervisor pins
# --------------------------------------------------------------------------
def test_supervisor_pins_what_the_run_and_preflight_use(sup, tk90):
    import re
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert set(tk90.SUITES) <= set(sup.SUITES)
    assert set(tk90.AUDITED) <= set(sup.AUDITED)
    assert set(sup.SUITES) - set(tk90.SUITES) == {"tests/test_rgrowx.py"}
    assert set(sup.AUDITED) - set(tk90.AUDITED) == {
        "tests/test_rgrowx.py", "benchmarks/supervisors/supervise_rgrowx.py"}
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
    for rel in ("experiments/ablation.py",
                "benchmarks/supervisors/supervise_rgrowx.py"):
        p = tmp_path / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# edit\n")
        ok, why = sup.manifest_check()
        assert not ok and rel in why
        p.write_bytes(orig.replace(b"\r\n", b"\n"))   # CRLF vs LF: same
        assert sup.manifest_check() == (True, "")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M muzero.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_refuses_bad_arguments(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a bad argv got past the usage check")

    # a wrongly accepted argv fails here instead of waiting (audit r1 A-L7)
    for name in ("other_supervisors", "run_done", "wait_external",
                 "preflight", "live_rgrow_supervisors"):
        monkeypatch.setattr(sup, name, no)
    for argv in ([], ["tkx"], ["0"], ["tkx", "1"], ["tk", "0"], ["rg", "0"],
                 ["tkx", "x"], ["tkx", "0", "1"], ["0", "tkx"],
                 ["tkx", "0", "--wait"], ["tkx", "0", "--no-wait", "x"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


def test_supervisor_quick_probe_passes(sup):
    import subprocess
    r = subprocess.run([sys.executable, "-c", sup.quick("tkx")], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------
# the wait for RG (its marker AND its evals), --no-wait, the trainer guard
# --------------------------------------------------------------------------
def test_rg_constants_name_the_live_rg_run(sup, tk90, monkeypatch):
    monkeypatch.setattr(tk90, "REPO", sup.RG_TREE)
    rel, base, _jsonl, ckpt, _d = tk90.paths("rg", 0)
    assert sup.RG_ARM == RG_ARM == tk90.VARIANTS["rg"]
    assert sup.RG_MARKER == ckpt
    assert sup.RG_SUPLOG == os.path.join(base, "supervisor.log")
    assert rel.endswith("/" + sup.RG_RUN)
    probes = os.path.join(sup.RG_TREE, "benchmarks", "probes")
    assert sup.RG_PROBES == (
        os.path.join(probes, f"probe_exhaustive_eval-{sup.RG_RUN}-acting.jsonl"),
        os.path.join(probes, f"probe_exhaustive_eval-{sup.RG_RUN}.jsonl"))
    assert sup.RG_ROWS == tuple(range(5, 95, 5))
    # the literal live paths (audit r1 A-L7: not only self-consistent)
    bs = chr(92)
    assert sup.RG_TREE == "Y:" + bs + "SwitchPlayer-rgrow-run"
    assert sup.RG_MARKER == bs.join([
        "Y:", "SwitchPlayer-rgrow-run", "benchmarks", "results",
        "mspacman-rgrow-s0-2026-10-19",
        f"resume-mspacmannoframeskip-skip4-{RG_ARM}-seed0.ckpt"])
    assert sup.RG_PROBES[0] == bs.join([
        "Y:", "SwitchPlayer-rgrow-run", "benchmarks", "probes",
        "probe_exhaustive_eval-mspacman-rgrow-s0-2026-10-19-acting.jsonl"])


@pytest.fixture
def rg_files(sup, monkeypatch, tmp_path):
    marker = tmp_path / "rg.ckpt"
    probes = (tmp_path / "p-acting.jsonl", tmp_path / "p.jsonl")
    suplog = tmp_path / "supervisor.log"
    monkeypatch.setattr(sup, "RG_MARKER", str(marker))
    monkeypatch.setattr(sup, "RG_PROBES", tuple(str(p) for p in probes))
    monkeypatch.setattr(sup, "RG_SUPLOG", str(suplog))
    # RG is driven (a live supervisor): never the real process table here
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [1])
    return SimpleNamespace(marker=marker, probes=probes, suplog=suplog)


def _rows(path, iterations, extra=()):
    lines = [json.dumps({"iteration": i, "exact_mean": 500.0,
                         "env": "MsPacmanNoFrameskip-v4"})
             for i in iterations]
    lines += list(extra)
    path.write_text("".join(l + chr(10) for l in lines))


def test_rg_missing_needs_the_marker_and_every_row_of_both_evals(sup,
                                                                 rg_files):
    got = sup.rg_missing()
    assert len(got) == 3 and "no checkpoint" in got[0]
    rg_files.marker.write_bytes(b"x" * 1_000_000)       # still a full ckpt
    assert "still full size" in sup.rg_missing()[0]
    rg_files.marker.write_bytes(b"x" * 100)             # the marker
    _rows(rg_files.probes[0], range(5, 95, 5))
    _rows(rg_files.probes[1], range(5, 90, 5), extra=[
        json.dumps({"iteration": 90, "max_noop": 2,       # a smoke record
                    "env": "MsPacmanNoFrameskip-v4"}),
        json.dumps({"iteration": 90, "exact_mean": 1.0}),  # Pong default
        "not json", json.dumps([90])])
    got = sup.rg_missing()
    assert got == ["p.jsonl rows [90]"]
    _rows(rg_files.probes[1], range(5, 95, 5))
    assert sup.rg_missing() == []


def test_wait_external_blocks_until_rg_is_evaluated(sup, rg_files,
                                                    monkeypatch, capsys):
    rg_files.suplog.write_text("LAUNCHED rg s0 attempt 1" + chr(10)
                               + "CRASH LOOP: 5 deaths" + chr(10))
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 200:                # fail, never hang (r2 A-L6)
            raise AssertionError("wait_external never returned")
        if len(sleeps) == 3:                 # RG finishes and its evals land
            rg_files.marker.write_bytes(b"x")
            for p in rg_files.probes:
                _rows(p, range(5, 95, 5))

    monkeypatch.setattr(sup.time, "sleep", fake_sleep)
    sup.wait_external("tkx", 0)
    out = capsys.readouterr().out
    assert sleeps == [60, 60, 60]
    assert "WAITING for RG" in out and "no checkpoint" in out
    assert "ALERT: RG's supervisor gave up" in out
    assert "may start" in out
    sleeps.clear()
    sup.wait_external("tkx", 0)              # done: no wait, no line
    assert sleeps == [] and capsys.readouterr().out == ""


def test_wait_external_alerts_when_rg_is_done_but_unevaluated(
        sup, rg_files, monkeypatch, capsys):
    rg_files.marker.write_bytes(b"x")
    _rows(rg_files.probes[0], range(5, 95, 5))
    _rows(rg_files.probes[1], range(5, 90, 5))         # row 90 never lands
    old = os.path.getmtime(rg_files.marker) - 3 * 3600
    os.utime(rg_files.marker, (old, old))
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 200:                # fail, never hang (r2 A-L6)
            raise AssertionError("wait_external never returned")
        if len(sleeps) == 61:
            _rows(rg_files.probes[1], range(5, 95, 5))

    monkeypatch.setattr(sup.time, "sleep", fake_sleep)
    sup.wait_external("tkx", 0)
    out = capsys.readouterr().out
    assert out.count("ALERT: RG completed 3.0 h ago but its evals still "
                     "lack rows") == 2                  # at once, then hourly
    assert "post_rgrow.py" in out and "may start" in out
    # a fresh marker (RG just finished): no ALERT yet
    rg_files.marker.write_bytes(b"x")
    _rows(rg_files.probes[1], range(5, 90, 5))
    sleeps.clear()

    def land(s):
        sleeps.append(s)
        if len(sleeps) > 200:
            raise AssertionError("wait_external never returned")
        _rows(rg_files.probes[1], range(5, 95, 5))

    monkeypatch.setattr(sup.time, "sleep", land)
    sup.wait_external("tkx", 0)
    assert "ALERT" not in capsys.readouterr().out


def test_rg_gave_up_reads_the_last_lines(sup, rg_files):
    assert not sup.rg_gave_up()                         # no log
    rg_files.suplog.write_text("CRASH LOOP: x" + chr(10) + "LAUNCHED rg"
                               + chr(10))
    assert not sup.rg_gave_up()                         # relaunched since
    rg_files.suplog.write_text("LAUNCHED rg" + chr(10) + "MAX ATTEMPTS"
                               + chr(10))
    assert sup.rg_gave_up()


class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


def test_no_wait_is_logged_and_refused_while_an_rgrow_supervisor_lives(
        sup, monkeypatch, tmp_path, capsys):
    def no(*a, **k):
        raise AssertionError("wait_external must not run under --no-wait")

    # the preflight runs this suite: pin the host probes, never read the
    # real process table or the real run folder (whose supervisor.log may
    # hold LAUNCHED lines on a relaunch) here, and never launch (r2 A-L7)
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup.subprocess, "Popen", no)
    monkeypatch.setattr(sup, "wait_no_trainer", no)
    monkeypatch.setattr(sup, "run_done", lambda v, s: False)
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [])
    monkeypatch.setattr(sup, "live_rgrow_handoffs", lambda: [])
    monkeypatch.setattr(sup, "wait_external", no)
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["tkx", "0", "--no-wait"])
    assert "NO-WAIT override: tkx s0" in capsys.readouterr().out
    monkeypatch.setattr(sup, "wait_external", _stop)  # without it: waits
    with pytest.raises(Stop):
        sup.main(["tkx", "0"])
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [42])
    assert sup.main(["tkx", "0", "--no-wait"]) == 2
    out = capsys.readouterr().out
    assert "refusing --no-wait" in out and "[42]" in out
    # RG's hand-off alive: refused too (audit r3 A-L5)
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [])
    monkeypatch.setattr(sup, "live_rgrow_handoffs", lambda: [43])
    assert sup.main(["tkx", "0", "--no-wait"]) == 2
    out = capsys.readouterr().out
    assert "post_rgrow.py is alive" in out and "[43]" in out


def test_live_rgrow_handoffs_exact_name(sup, monkeypatch):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 1, "name": "python.exe",
         "cmdline": ["py", "-u", "C:" + bs + "s" + bs + "post_rgrow.py",
                     "Y:" + bs + "w"]},
        {"pid": 2, "name": "python.exe",
         "cmdline": ["py", "-u", "C:/s/post_rgrowx.py", "Y:/w"]},
        {"pid": 3, "name": "python.exe",
         "cmdline": ["py", "check_post_rgrow.py"]},
        {"pid": 4, "name": "cmd.exe",
         "cmdline": ["cmd", "/c", "post_rgrow.py"]}])
    assert sup.live_rgrow_handoffs() == [1]


def test_wait_external_alerts_when_nothing_drives_rg(sup, rg_files,
                                                    monkeypatch, capsys):
    """A reboot that relaunched only TKX (audit r3 A-L1)."""
    rg_files.marker.write_bytes(b"x" * 1_000_000)      # RG unfinished
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [])
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 200:
            raise AssertionError("wait_external never returned")
        if len(sleeps) == 2:
            rg_files.marker.write_bytes(b"x")
            for p in rg_files.probes:
                _rows(p, range(5, 95, 5))

    monkeypatch.setattr(sup.time, "sleep", fake_sleep)
    sup.wait_external("tkx", 0)
    out = capsys.readouterr().out
    assert "ALERT: RG is unfinished and nothing drives it" in out
    # a live supervisor with a fresh log: no ALERT; a silent log: ALERT
    monkeypatch.setattr(sup, "live_rgrow_supervisors", lambda: [1])
    rg_files.marker.write_bytes(b"x" * 1_000_000)
    rg_files.suplog.write_text("EVAL rg" + chr(10))
    assert not sup.rg_unattended()
    old = os.path.getmtime(rg_files.suplog) - 5 * 3600
    os.utime(rg_files.suplog, (old, old))
    assert sup.rg_unattended()
    rg_files.marker.write_bytes(b"x")                  # finished: never
    assert not sup.rg_unattended()


def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys):
    def no(*a, **k):
        raise AssertionError("no wait or preflight for a finished run")

    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "wait_external", no)
    monkeypatch.setattr(sup, "live_rgrow_supervisors", no)
    monkeypatch.setattr(sup, "preflight", no)
    assert sup.main(["tkx", "0"]) == 0
    assert sup.main(["tkx", "0", "--no-wait"]) == 0
    assert "RUN COMPLETE tkx s0" in capsys.readouterr().out


def _procs(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


def test_live_rgrow_supervisors_finds_either_variant_only(sup, monkeypatch):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 1, "cmdline": ["py", "Y:" + bs + "run" + bs + "benchmarks"
                               + bs + "supervisors" + bs
                               + "supervise_rgrow.py", "tk", "0"]},
        {"pid": 2, "cmdline": ["py", "supervise_rgrow.py", "rg", "0",
                               "--no-wait"]},
        {"pid": 3, "cmdline": ["py", "benchmarks/supervisors/"
                               "supervise_rgrowx.py", "tkx", "0"]},
        {"pid": 4, "cmdline": ["py", "supervise_tokens.py", "tk", "0"]},
        {"pid": 5, "cmdline": ["py", "notsupervise_rgrow.py", "rg", "0"]},
        {"pid": 6, "cmdline": None}])
    assert sup.live_rgrow_supervisors() == [1, 2]


def test_live_trainers_finds_any_ablation_run(sup, monkeypatch):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 1, "name": "python.exe",
         "cmdline": ["py", "Y:" + bs + "run" + bs + "experiments"
                     + bs + "ablation.py", "--arms", RG_ARM, "--seed", "0"]},
        {"pid": 2, "name": "python.exe",
         "cmdline": ["py", "experiments/ablation.py", "--arms", TK_ARM,
                     "--seed", "1"]},
        {"pid": 3, "name": "Python3.12",
         "cmdline": ["py", "experiments/ablation.py", "--arms",
                     "pong-whatever"]},
        {"pid": 4, "name": "python.exe",
         "cmdline": ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                     "x.pth"]},
        {"pid": 5, "name": "python.exe",
         "cmdline": ["py", "-m", "pytest", "tests/test_rgrowx.py"]},
        {"pid": 6, "name": "python.exe", "cmdline": None},
        # not trainers: an editor, git, a shell naming the file (A-L5)
        {"pid": 7, "name": "Code.exe",
         "cmdline": ["code", "experiments/ablation.py"]},
        {"pid": 8, "name": "git.exe",
         "cmdline": ["git", "diff", "--", "experiments/ablation.py"]},
        {"pid": 9, "name": None, "cmdline": ["experiments/ablation.py"]}])
    assert sup.live_trainers() == [1, 2, 3]


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 10, "name": "python.exe",           # this process
         "cmdline": ["py", "benchmarks" + bs + "supervisors" + bs
                     + "supervise_rgrowx.py", "tkx", "0"]},
        {"pid": 11, "name": "python.exe",           # its venv launcher
         "cmdline": ["py", "benchmarks/supervisors/supervise_rgrowx.py",
                     "tkx", "0"]},
        {"pid": 12, "name": "cmd.exe",              # the WMI shell
         "cmdline": ["cmd.exe", "/c", "py supervise_rgrowx.py tkx 0 >> x"]},
        {"pid": 14, "name": "notepad.exe",          # not python (r2 A-L7)
         "cmdline": ["notepad", "benchmarks/supervisors/"
                     "supervise_rgrowx.py"]},
        {"pid": 13, "name": "python.exe",           # RG's supervisor
         "cmdline": ["py", "benchmarks/supervisors/supervise_rgrow.py",
                     "rg", "0"]}])
    monkeypatch.setattr(sup, "_mine", lambda: {10, 11})
    assert sup.other_supervisors() == []
    _procs(monkeypatch, [
        {"pid": 20, "name": "python.exe",           # a second launch
         "cmdline": ["py", "benchmarks/supervisors/supervise_rgrowx.py",
                     "tkx", "0", "--no-wait"]}])
    assert sup.other_supervisors() == [20]

    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    for name in ("run_done", "wait_external", "preflight",
                 "live_rgrow_supervisors"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["tkx", "0"]) == 2
    assert sup.main(["tkx", "0", "--no-wait"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_rgrowx.py is alive (pids [20])" in out


def test_a_resume_skips_the_wait_for_rg(sup, monkeypatch, tmp_path, capsys):
    def no(*a, **k):
        raise AssertionError("a resume must not wait for RG again")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "live_rgrow_supervisors", no)
    monkeypatch.setattr(sup, "wait_external", no)
    monkeypatch.setattr(sup, "preflight", _stop)
    # a regression past the preflight fails here, never hangs (r3 A-L7)
    for name in ("wait_no_trainer", "manifest_check"):
        monkeypatch.setattr(sup, name, no)
    monkeypatch.setattr(sup.subprocess, "Popen", no)
    ckpt = pathlib.Path(sup.paths("tkx", 0)[3])
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x" * 1_000_001)               # a real checkpoint
    with pytest.raises(Stop):
        sup.main(["tkx", "0"])
    assert "RESUMING tkx s0" in capsys.readouterr().out
    ckpt.unlink()                                    # fresh: it waits
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["tkx", "0"])


def test_every_launch_attempt_waits_for_no_trainer(sup, monkeypatch,
                                                   tmp_path, capsys):
    """Behavioural (audit r1 A-L7): the guard runs before EVERY Popen."""
    events = []
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "wait_external", lambda v, s: None)
    monkeypatch.setattr(sup, "preflight", lambda v: (True, ""))
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "snapshot", lambda *a: None)
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)

    class Dead:
        pid, returncode = 99, 1

        def poll(self):
            return 1

    def popen(args, **k):
        events.append("popen")
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["tkx", "0"]) == 1              # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == ["guard", "popen"] * 5


def test_wait_no_trainer_waits_out_any_trainer(sup, monkeypatch, capsys):
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])
    sleeps = []
    monkeypatch.setattr(sup.time, "sleep", sleeps.append)
    sup.wait_no_trainer()
    out = capsys.readouterr().out
    assert out.count("ALERT: an experiments/ablation.py is already "
                     "running (pids [7])") == 1      # told once, not per poll
    assert sleeps == [60, 60]
    src = inspect.getsource(sup.main)
    assert src.count("wait_no_trainer(") == 1
    assert src.index("wait_external(variant, seed)") < src.index("preflight(")
    # the guard, THEN the manifest, then the launch (audit r2 A-L3)
    assert src.index("wait_no_trainer(rows)") < src.index("manifest_check()") \
        < src.index("subprocess.Popen(")


def test_wait_no_trainer_keeps_rows_streaming(sup, monkeypatch, tmp_path,
                                             capsys):
    """An orphan of THIS run stays monitored while the guard waits on it
    (audit r2 A-L4)."""
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "tkx", 0)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])

    def sleep(s):
        jsonl.write_text(json.dumps({"iteration": 140, "reward_ceiling": 99.0,
                                     "elapsed_s": 1.0}) + chr(10))

    monkeypatch.setattr(sup.time, "sleep", sleep)
    sup.wait_no_trainer(rows)
    out = capsys.readouterr().out
    assert "EVAL tkx s0 iter 140" in out
    assert "ALERT: tkx s0 row 140: reward_ceiling 99.0" in out


def test_abandoned_sanctioned_and_stray_snapshots(sup, monkeypatch, tmp_path,
                                                  capsys):
    def no(*a, **k):
        raise AssertionError("must stop before any wait or launch")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    rel, base, _j, ckpt, _d = sup.paths("tkx", 0)
    base = pathlib.Path(base)
    base.mkdir(parents=True)
    # sanctioned: the checkpoint, or a LAUNCHED / NO-WAIT line in its log
    assert not sup.sanctioned(str(base), ckpt)
    (base / "supervisor.log").write_text("WAITING for RG" + chr(10))
    assert not sup.sanctioned(str(base), ckpt)
    (base / "supervisor.log").write_text("NO-WAIT override: tkx s0" + chr(10))
    assert sup.sanctioned(str(base), ckpt)
    (base / "supervisor.log").write_text("LAUNCHED tkx s0 attempt 1" + chr(10))
    assert sup.sanctioned(str(base), ckpt)
    # stray snapshots (a fresh start would mix timelines) are refused
    snaps = base / "snapshots-x"
    snaps.mkdir()
    (snaps / "weights-iter005.pth").write_bytes(b"x")
    for name in ("wait_external", "preflight", "wait_no_trainer",
                 "live_rgrow_supervisors"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["tkx", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    # ABANDONED stops every launch, before any wait
    (base / "ABANDONED").write_text("")
    assert sup.main(["tkx", "0"]) == 0
    assert sup.main(["tkx", "0", "--no-wait"]) == 0
    assert "ABANDONED tkx s0" in capsys.readouterr().out


def test_probe_rows_without_a_checkpoint_are_refused(sup, monkeypatch,
                                                     tmp_path, capsys):
    """A fresh restart that forgot the probe jsonls (audit r3 A-M1)."""
    def no(*a, **k):
        raise AssertionError("must stop before any wait or launch")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    for name in ("wait_external", "preflight", "wait_no_trainer",
                 "live_rgrow_supervisors", "live_rgrow_handoffs"):
        monkeypatch.setattr(sup, name, no)
    rel, base, _j, ckpt, _d = sup.paths("tkx", 0)
    probes = tmp_path / "benchmarks" / "probes"
    probes.mkdir(parents=True)
    f = probes / ("probe_exhaustive_eval-mspacman-tokens-rgrow-x270-s0-"
                  "2026-10-20-acting.jsonl")
    f.write_text(json.dumps({"iteration": 5}) + chr(10))
    assert sup.stray_probe_rows("tkx", 0, ckpt) == [f.name]
    assert sup.main(["tkx", "0"]) == 2
    assert sup.main(["tkx", "0", "--no-wait"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(base).mkdir(parents=True)
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)  # a legit resume
    assert sup.stray_probe_rows("tkx", 0, ckpt) == []
    f.write_text("")                                  # empty: not a record
    pathlib.Path(ckpt).unlink()
    assert sup.stray_probe_rows("tkx", 0, ckpt) == []


def _attempt_loop(sup, monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "wait_external", lambda v, s: None)
    monkeypatch.setattr(sup, "preflight", lambda v: (True, ""))
    monkeypatch.setattr(sup, "snapshot", lambda *a: None)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    rel, base, _j, ckpt, _d = sup.paths("tkx", 0)
    return pathlib.Path(base), pathlib.Path(ckpt)


def test_a_failing_manifest_blocks_the_launch(sup, monkeypatch, tmp_path):
    """Behavioural (audit r3 A-L7): no Popen until the manifest passes."""
    _attempt_loop(sup, monkeypatch, tmp_path)
    events = []
    checks = [(False, "not the audited revision: x"),
              (False, "not the audited revision: x"), (True, "")]
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or checks.pop(0))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)

    def popen(args, **k):
        events.append("popen")
        raise Stop

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        sup.main(["tkx", "0"])
    assert events == ["guard", "manifest"] * 3 + ["popen"]


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, tmp_path, capsys):
    """Behavioural (audit r3 A-L2/L7): a flag or a marker that appears while
    an attempt waits stops the launch."""
    base, ckpt = _attempt_loop(sup, monkeypatch, tmp_path)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    # ABANDONED created during the guard's wait
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["tkx", "0"]) == 0
    assert "ABANDONED tkx s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    # an orphan of this run completed during the guard's wait
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["tkx", "0"]) == 0
    assert "RUN COMPLETE tkx s0" in capsys.readouterr().out


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row,per,want", [
    ({"iteration": 1, "reward_ceiling": 1.0, "reward_max": 1.0}, "n/a", []),
    ({"iteration": 10, "reward_ceiling": round(160 ** (18 / 270), 4)}, "11.0",
     []),
    ({"iteration": 140, "reward_ceiling": 160.0, "reward_max": 80.0}, "24.9",
     []),
    # TK-90's schedule is wrong here: 160 at row 50 is off the stretch
    ({"iteration": 50, "reward_ceiling": 160.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5}, "n/a", ["reward_ceiling"]),
    ({"iteration": 140, "reward_ceiling": 160.0}, "25.5", ["min/iteration"]),
    ({"iteration": 140, "reward_ceiling": 160.0, "reward_max": 450.0}, "n/a",
     ["reward_max"]),
])
def test_supervisor_row_alerts(sup, row, per, want):
    got = sup.row_alerts(row, per)
    assert len(got) == len(want), got
    for g, w in zip(got, want):
        assert w in g


def test_supervisor_rows_print_the_growth_and_token_fields(sup, tmp_path,
                                                           monkeypatch,
                                                           capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "tkx", 0)
    # never the real host's memory: < 4 GB would fail a preflight (r2 A-L5)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    jsonl.write_text(json.dumps({
        "iteration": 5, "return_mean": 500.0,
        "reward_ceiling": round(160 ** (8 / 270), 4),
        "reward_kept": 0.42, "reward_max": 1.1,
        "tok_frames_per_decision": 9.5, "tok_decisions_per_game": 40.0,
        "elapsed_s": 100.0}) + chr(10))
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL tkx s0 iter 5")
    assert "kept 0.42  max 1.1" in out
    assert "frames/decision 9.5  decisions/life 40.0" in out
    assert "ALERT" not in out


def test_supervisor_rows_alert_on_low_host_commit(sup, tmp_path, monkeypatch,
                                                  capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "tkx", 0)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.2)
    jsonl.write_text(json.dumps({
        "iteration": 140, "reward_ceiling": 160.0, "elapsed_s": 1.0}) + chr(10))
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: tkx s0 row 140: host commit free 3.2 GB < 4 GB" in out
