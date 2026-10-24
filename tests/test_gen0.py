"""Test 1 of the per-generation clip (Kai, 2026-10-24; CURRICULUM.md "Next
experiment (pre-registered) -- Test 1: a generation-0 token master at a
constant sign clip"): the gen0 arm is TK with its reward ceiling held at 1
for the whole run, every other schedule TK's; supervise_gen0.py runs it with
TK's recipe. Pins the arm's delta, the constant ceiling (and that it is the
sign clip, per base step), the gates, the recipe against TK's, the trainer
guard (distill.py counts), the manifest, the refusals and the row alerts.
CPU only, no launches."""
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
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
ENV_ID = "MsPacmanNoFrameskip-v4"
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
    return _load(SUPERVISORS / "supervise_gen0.py", "supervise_gen0")


@pytest.fixture
def tk90():
    return _load(SUPERVISORS / "supervise_rgrow.py", "supervise_rgrow")


# --------------------------------------------------------------------------
# the arm and its schedules
# --------------------------------------------------------------------------
def test_g0_arm_delta_is_exactly_the_constant_ceiling():
    tk, g0 = ablation.ARMS[TK_ARM], ablation.ARMS[G0_ARM]
    delta = {k for k in set(tk) | set(g0) if tk.get(k) != g0.get(k)}
    # TK already starts at 1 (the sign clip): only its end moves
    assert delta == {"REWARD_GROWTH_CEILING", "_doc"}
    assert (g0["REWARD_GROWTH_START"], g0["REWARD_GROWTH_CEILING"]) == (1.0,
                                                                        1.0)
    assert (tk["REWARD_GROWTH_START"], tk["REWARD_GROWTH_CEILING"]) == (1.0,
                                                                        160.0)
    assert g0["REWARD_GROWTH"] is True and g0["REWARD_UNIT"] == 10.0
    assert len(g0["TOKEN_VOCAB"]) == 36
    assert g0["_launch"] == {"ray": True, "cpu_actor": True}
    assert "_student" not in g0


def test_the_ceiling_is_1_for_the_whole_run_and_the_rest_is_tks():
    at = lambda it: (it - 1) * 2000                  # noqa: E731
    keys = []
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(90, 2000):
        for it in range(1, 91):
            keys.append((muzero.token_long_bias(at(it)),
                         muzero.prior_acting_temperature(at(it)),
                         muzero.per_beta(at(it))))
        tk_ceilings = [muzero.reward_ceiling(at(it)) for it in range(1, 91)]
        tk_globals = (muzero.TEMPERATURE_TOTAL_STEPS,
                      muzero.OFFPOLICY_DECAY_STEPS,
                      muzero.MAX_REPLAY_TRANSITIONS)
    with ablation.arm_config(G0_ARM), ablation._run_length_globals(90, 2000):
        for it in range(1, 91):
            assert muzero.reward_ceiling(at(it)) == 1.0, it
            assert (muzero.token_long_bias(at(it)),
                    muzero.prior_acting_temperature(at(it)),
                    muzero.per_beta(at(it))) == keys[it - 1], it
        assert muzero.reward_ceiling(0) == 1.0
        assert muzero.reward_ceiling(10 ** 9) == 1.0
        assert (muzero.TEMPERATURE_TOTAL_STEPS, muzero.OFFPOLICY_DECAY_STEPS,
                muzero.MAX_REPLAY_TRANSITIONS) == tk_globals
        assert muzero.VTREE_REFRESH_BATCH == 64
        assert muzero.VTREE_COMMUNAL_TRAJS == 1
    # what the comparison is against: TK's moving ceiling, 160 from row 46
    assert tk_ceilings[0] == 1.0 and tk_ceilings[45] == 160.0
    assert tk_ceilings[44] < 160.0


def test_ceiling_1_is_the_sign_clip_per_base_step():
    with ablation.arm_config(G0_ARM):
        unit = muzero.REWARD_UNIT
        # every Ms. Pac-Man reward (pellet 10 ... ghost 1600, fruit 5000)
        for r in (10.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 5000.0):
            assert muzero.grown_reward((r,), 1.0) == 1.0 == \
                muzero.clip_reward(r), r
        assert muzero.grown_reward((0.0,), 1.0) == 0.0
        # a token sums its inner base steps' clipped rewards: at most 8
        # for a 32-frame token (the supervisor's reward_max ALERT bound)
        assert muzero.grown_reward((10.0, 200.0, 0.0, 1600.0), 1.0) == 3.0
        assert muzero.grown_reward((5000.0,) * 8, 1.0) == 8.0
        assert unit == 10.0


def test_every_startup_gate_passes_for_g0(monkeypatch):
    for k in ablation.ARMS[G0_ARM]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with ablation.arm_config(G0_ARM) as spec, \
            ablation._run_length_globals(90, 2000), ablation.token_actions():
        assert ablation.ACTIONS == 36
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()
        ablation.token_gate(spec, True, True, 0)
        ablation.reward_growth_gate(True, True, 0)
        # not a student: the distillation gate passes it through untouched
        # and adds nothing to the resume recipe (audit r1 A-LOW)
        assert spec.get("_student") is None
        ablation.student_gate(spec, None, True)
        ablation.student_gate(spec, None, False)
        assert ablation.student_expect(spec, None) is None


# --------------------------------------------------------------------------
# what the supervisor pins
# --------------------------------------------------------------------------
def test_recipe_is_tks_with_the_g0_arm(sup, tk90):
    assert sup.VARIANTS == {"g0": G0_ARM} and tk90.VARIANTS["tk"] == TK_ARM
    assert sup.SEEDS == {0: "2026-10-24"}
    g, t = sup.command("g0", 0), tk90.command("tk", 0)
    i = g.index("--arms")
    assert g[i + 1] == G0_ARM and t[i + 1] == TK_ARM
    assert g[:i + 1] + g[i + 2:] == t[:i + 1] + t[i + 2:]
    assert g[g.index("--iterations") + 1] == "90"
    assert "--resume" not in g


def test_paths_and_the_run_folder(sup):
    rel, base, jsonl, ckpt, deaths = sup.paths("g0", 0)
    assert rel == "benchmarks/results/mspacman-tokens-gen0-s0-2026-10-24"
    assert jsonl.endswith("ablation-mspacmannoframeskip-skip4.jsonl")
    assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{G0_ARM}-seed0.ckpt")
    assert deaths.endswith("deaths.log")
    assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT          # runs from its own tree
    assert pathlib.Path(sup.MANIFEST).name == "gen0_manifest.json"
    assert sup.MAX_ATTEMPTS == 40


def test_the_supervisors_ceiling_is_the_arms(sup):
    with ablation.arm_config(G0_ARM), ablation._run_length_globals(90, 2000):
        for it in range(1, 91):
            assert sup.expected_ceiling(it) == muzero.reward_ceiling(
                (it - 1) * 2000) == 1.0, it
    # the longest token's bound
    with ablation.arm_config(G0_ARM):
        assert max(f for _, f in muzero.TOKEN_VOCAB) // 4 == \
            sup.REWARD_MAX_ALERT


def test_supervisor_pins_what_the_run_and_preflight_use(sup, tk90):
    import re
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert set(tk90.SUITES) <= set(sup.SUITES)
    assert set(tk90.AUDITED) <= set(sup.AUDITED)
    assert set(sup.SUITES) - set(tk90.SUITES) == {"tests/test_gen0.py",
                                                  "tests/test_distill.py"}
    assert set(sup.AUDITED) - set(tk90.AUDITED) == {
        "tests/test_gen0.py", "tests/test_distill.py",
        "experiments/distill.py", "benchmarks/supervisors/supervise_gen0.py"}
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)
        for name in re.findall(r"^import (\w+)", src, re.M):
            if (ROOT / "experiments" / f"{name}.py").exists():
                assert f"experiments/{name}.py" in sup.AUDITED, (rel, name)


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
                "benchmarks/supervisors/supervise_gen0.py"):
        p = tmp_path / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# edit\n")
        ok, why = sup.manifest_check()
        assert not ok and rel in why
        # CRLF and LF hash alike, whichever form the tree holds
        lf = orig.replace(b"\r\n", b"\n")
        p.write_bytes(lf.replace(b"\n", b"\r\n") if orig == lf else lf)
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

    for name in ("other_supervisors", "run_done", "preflight"):
        monkeypatch.setattr(sup, name, no)
    for argv in ([], ["g0"], ["0"], ["g0", "1"], ["tk", "0"], ["tkx", "0"],
                 ["g0", "x"], ["g0", "0", "1"], ["0", "g0"],
                 ["g0", "0", "--no-wait"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


def test_supervisor_quick_probe_passes(sup):
    import subprocess
    r = subprocess.run([sys.executable, "-c", sup.quick("g0")], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------
# the trainer guard, one supervisor per run, the refusals
# --------------------------------------------------------------------------
class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


def _procs(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


def test_live_trainers_finds_any_ablation_run_or_prep(sup, monkeypatch):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 1, "name": "python.exe",
         "cmdline": ["py", "Y:" + bs + "run" + bs + "experiments"
                     + bs + "ablation.py", "--arms", TK_ARM, "--seed", "0"]},
        {"pid": 2, "name": "python.exe",
         "cmdline": ["py", "experiments/ablation.py", "--arms", G0_ARM,
                     "--seed", "1"]},
        {"pid": 3, "name": "Python3.12",
         "cmdline": ["py", "experiments/ablation.py", "--arms",
                     "pong-whatever"]},
        {"pid": 4, "name": "python.exe",
         "cmdline": ["py", "experiments" + bs + "distill.py", "--master",
                     "m.pth"]},
        {"pid": 5, "name": "python.exe",
         "cmdline": ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                     "x.pth"]},
        {"pid": 6, "name": "python.exe",
         "cmdline": ["py", "-m", "pytest", "tests/test_gen0.py"]},
        {"pid": 7, "name": "python.exe", "cmdline": None},
        # not trainers: an editor, git, a shell naming the file
        {"pid": 8, "name": "Code.exe",
         "cmdline": ["code", "experiments/ablation.py"]},
        {"pid": 9, "name": "git.exe",
         "cmdline": ["git", "diff", "--", "experiments/distill.py"]},
        {"pid": 10, "name": None, "cmdline": ["experiments/ablation.py"]},
        {"pid": 11, "name": "python.exe",
         "cmdline": ["py", "tests/test_distill.py"]},
        # audit r1 B-L4: from experiments/, or as a module
        {"pid": 12, "name": "python.exe",
         "cmdline": ["py", "ablation.py", "--arms", G0_ARM]},
        {"pid": 13, "name": "python.exe",
         "cmdline": ["py", "-m", "experiments.distill", "--master", "m"]},
        {"pid": 14, "name": "python.exe",
         "cmdline": ["py", "-m", "ablation"]},
        {"pid": 15, "name": "python.exe",
         "cmdline": ["py", "-m", "pytest", "ablation"]},
        {"pid": 16, "name": "python.exe",
         "cmdline": ["py", "experiments/ablation.pyc"]}])
    assert sup.live_trainers() == [1, 2, 3, 4, 12, 13, 14]


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    bs = chr(92)
    _procs(monkeypatch, [
        {"pid": 10, "name": "python.exe",           # this process
         "cmdline": ["py", "benchmarks" + bs + "supervisors" + bs
                     + "supervise_gen0.py", "g0", "0"]},
        {"pid": 11, "name": "python.exe",           # its venv launcher
         "cmdline": ["py", "benchmarks/supervisors/supervise_gen0.py",
                     "g0", "0"]},
        {"pid": 12, "name": "cmd.exe",              # the WMI shell
         "cmdline": ["cmd.exe", "/c", "py supervise_gen0.py g0 0 >> x"]},
        {"pid": 14, "name": "notepad.exe",          # not python
         "cmdline": ["notepad", "benchmarks/supervisors/supervise_gen0.py"]},
        {"pid": 13, "name": "python.exe",           # another experiment's
         "cmdline": ["py", "benchmarks/supervisors/supervise_rgrow.py",
                     "tk", "0"]}])
    monkeypatch.setattr(sup, "_mine", lambda: {10, 11})
    assert sup.other_supervisors() == []
    _procs(monkeypatch, [
        {"pid": 20, "name": "python.exe",           # a second launch
         "cmdline": ["py", "benchmarks/supervisors/supervise_gen0.py",
                     "g0", "0"]}])
    assert sup.other_supervisors() == [20]

    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    for name in ("run_done", "preflight"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["g0", "0"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_gen0.py is alive (pids [20])" in out


def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys):
    def no(*a, **k):
        raise AssertionError("no preflight for a finished run")

    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "preflight", no)
    assert sup.main(["g0", "0"]) == 0
    assert "RUN COMPLETE g0 s0" in capsys.readouterr().out


def test_a_fresh_start_goes_straight_to_the_preflight(sup, monkeypatch,
                                                      tmp_path, capsys):
    """No predecessor: nothing waits before the preflight; a resume logs
    RESUMING."""
    def no(*a, **k):
        raise AssertionError("nothing may run before the preflight")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "preflight", _stop)
    for name in ("wait_no_trainer", "manifest_check"):
        monkeypatch.setattr(sup, name, no)
    monkeypatch.setattr(sup.subprocess, "Popen", no)
    with pytest.raises(Stop):
        sup.main(["g0", "0"])
    assert "RESUMING" not in capsys.readouterr().out
    ckpt = pathlib.Path(sup.paths("g0", 0)[3])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"x" * 1_000_001)               # a real checkpoint
    with pytest.raises(Stop):
        sup.main(["g0", "0"])
    assert "RESUMING g0 s0" in capsys.readouterr().out


def test_every_launch_attempt_waits_for_no_trainer(sup, monkeypatch,
                                                   tmp_path, capsys):
    """Behavioural: the guard runs before EVERY Popen."""
    events = []
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
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

    seen = []

    def popen(args, **k):
        events.append("popen")
        seen.append((list(args), k))
        if len(seen) == 1:          # the first attempt leaves a checkpoint
            ckpt = pathlib.Path(sup.paths("g0", 0)[3])
            ckpt.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    # the preflight's CPU-only env must never reach the trainer (B-L2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert sup.main(["g0", "0"]) == 1              # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == ["guard", "popen"] * 5
    rel = sup.paths("g0", 0)[0]
    for i, (args, k) in enumerate(seen):
        assert args == sup.command("g0", 0) + (["--resume"] if i else [])
        assert k["cwd"] == sup.REPO
        assert k["env"]["ABLATION_RESULTS"] == rel
        assert k["env"]["PYTHONFAULTHANDLER"] == "1"
        assert "CUDA_VISIBLE_DEVICES" not in k["env"]


def test_wait_no_trainer_waits_out_any_trainer(sup, monkeypatch, capsys):
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])
    sleeps = []
    monkeypatch.setattr(sup.time, "sleep", sleeps.append)
    sup.wait_no_trainer()
    out = capsys.readouterr().out
    assert out.count("ALERT: an experiments/ablation.py or distill.py is "
                     "already running (pids [7])") == 1   # told once
    assert sleeps == [60, 60]
    src = inspect.getsource(sup.main)
    assert src.count("wait_no_trainer(") == 1
    # the guard, THEN the manifest, then the launch
    assert src.index("wait_no_trainer(rows)") < src.index("manifest_check()") \
        < src.index("subprocess.Popen(")
    # rows primed before the preflight and emitted between its retries
    pre = src.index("ok, why = preflight(variant)")
    assert src.index("rows.prime()") < pre < src.index("rows.emit()")
    assert src.index("rows.emit()") < src.index("preflight ok")


def test_wait_no_trainer_keeps_rows_streaming(sup, monkeypatch, tmp_path,
                                             capsys):
    """An orphan of THIS run stays monitored while the guard waits on it."""
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "g0", 0)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])

    def sleep(s):
        jsonl.write_text(json.dumps({"iteration": 40, "reward_ceiling": 99.0,
                                     "elapsed_s": 1.0}) + chr(10))

    monkeypatch.setattr(sup.time, "sleep", sleep)
    sup.wait_no_trainer(rows)
    out = capsys.readouterr().out
    assert "EVAL g0 s0 iter 40" in out
    assert "ALERT: g0 s0 row 40: reward_ceiling 99.0" in out


def test_abandoned_and_stray_snapshots(sup, monkeypatch, tmp_path, capsys):
    def no(*a, **k):
        raise AssertionError("must stop before any preflight or launch")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    rel, base, _j, ckpt, _d = sup.paths("g0", 0)
    base = pathlib.Path(base)
    base.mkdir(parents=True)
    snaps = base / "snapshots-x"
    snaps.mkdir()
    (snaps / "weights-iter005.pth").write_bytes(b"x")
    for name in ("preflight", "wait_no_trainer"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["g0", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    # ABANDONED stops every launch, before any preflight
    (base / "ABANDONED").write_text("")
    assert sup.main(["g0", "0"]) == 0
    assert "ABANDONED g0 s0" in capsys.readouterr().out


def test_probe_rows_without_a_checkpoint_are_refused(sup, monkeypatch,
                                                     tmp_path, capsys):
    def no(*a, **k):
        raise AssertionError("must stop before any preflight or launch")

    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    for name in ("preflight", "wait_no_trainer"):
        monkeypatch.setattr(sup, name, no)
    rel, base, _j, ckpt, _d = sup.paths("g0", 0)
    probes = tmp_path / "benchmarks" / "probes"
    probes.mkdir(parents=True)
    f = probes / ("probe_exhaustive_eval-mspacman-tokens-gen0-s0-"
                  "2026-10-24-acting.jsonl")
    f.write_text(json.dumps({"iteration": 5}) + chr(10))
    assert sup.stray_probe_rows("g0", 0, ckpt) == [f.name]
    assert sup.main(["g0", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(base).mkdir(parents=True)
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)  # a legit resume
    assert sup.stray_probe_rows("g0", 0, ckpt) == []
    f.write_text("")                                  # empty: not a record
    pathlib.Path(ckpt).unlink()
    assert sup.stray_probe_rows("g0", 0, ckpt) == []


def _attempt_loop(sup, monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "other_supervisors", lambda: [])
    monkeypatch.setattr(sup, "preflight", lambda v: (True, ""))
    monkeypatch.setattr(sup, "snapshot", lambda *a: None)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    rel, base, _j, ckpt, _d = sup.paths("g0", 0)
    return pathlib.Path(base), pathlib.Path(ckpt)


def test_a_failing_manifest_blocks_the_launch(sup, monkeypatch, tmp_path):
    """Behavioural: no Popen until the manifest passes."""
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
        sup.main(["g0", "0"])
    assert events == ["guard", "manifest"] * 3 + ["popen"]


def test_stray_state_rechecked_before_a_launch_without_checkpoint(
        sup, monkeypatch, tmp_path, capsys):
    base, ckpt = _attempt_loop(sup, monkeypatch, tmp_path)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    probes = tmp_path / "benchmarks" / "probes"

    def old_rows_appear(*a):
        probes.mkdir(parents=True, exist_ok=True)
        (probes / ("probe_exhaustive_eval-mspacman-tokens-gen0-s0-"
                   "2026-10-24.jsonl")).write_text("{}" + chr(10))

    monkeypatch.setattr(sup, "wait_no_trainer", old_rows_appear)
    assert sup.main(["g0", "0"]) == 2
    assert "stray snapshots / probe jsonls but no checkpoint" in \
        capsys.readouterr().out


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, tmp_path, capsys):
    base, ckpt = _attempt_loop(sup, monkeypatch, tmp_path)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["g0", "0"]) == 0
    assert "ABANDONED g0 s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["g0", "0"]) == 0
    assert "RUN COMPLETE g0 s0" in capsys.readouterr().out


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row,per,want", [
    ({"iteration": 1, "reward_ceiling": 1.0, "reward_max": 1.0}, "n/a", []),
    ({"iteration": 90, "reward_ceiling": 1.0, "reward_max": 8.0}, "24.9", []),
    # TK's moving ceiling is wrong here
    ({"iteration": 10, "reward_ceiling": round(160 ** (18 / 90), 4)}, "n/a",
     ["reward_ceiling"]),
    ({"iteration": 50, "reward_ceiling": 160.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5}, "n/a", ["reward_ceiling"]),
    ({"iteration": 40, "reward_ceiling": 1.0}, "25.5", ["min/iteration"]),
    ({"iteration": 40, "reward_ceiling": 1.0, "reward_max": 8.5}, "n/a",
     ["reward_max"]),
])
def test_supervisor_row_alerts(sup, row, per, want):
    got = sup.row_alerts(row, per)
    assert len(got) == len(want), got
    for g, w in zip(got, want):
        assert w in g


def test_supervisor_rows_print_the_clip_prior_and_token_fields(
        sup, tmp_path, monkeypatch, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "g0", 0)
    # never the real host's memory: < 4 GB would fail a preflight
    monkeypatch.setattr(sup, "commit_free_gb", lambda: None)
    jsonl.write_text(json.dumps({
        "iteration": 5, "return_mean": 500.0, "reward_ceiling": 1.0,
        "reward_kept": 0.42, "reward_max": 3.0, "policy_maxp": 0.61,
        "tok_frames_per_decision": 9.5, "tok_decisions_per_game": 40.0,
        "elapsed_s": 100.0}) + chr(10))
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL g0 s0 iter 5")
    assert "maxp 0.61" in out
    assert "ceiling 1.0  kept 0.42  max 3.0" in out
    assert "frames/decision 9.5  decisions/life 40.0" in out
    assert "ALERT" not in out


def test_supervisor_rows_alert_on_low_host_commit(sup, tmp_path, monkeypatch,
                                                  capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "g0", 0)
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.2)
    jsonl.write_text(json.dumps({
        "iteration": 40, "reward_ceiling": 1.0, "elapsed_s": 1.0}) + chr(10))
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: g0 s0 row 40: host commit free 3.2 GB < 4 GB" in out


def test_no_progress_stops_slow_deaths_without_a_checkpoint(
        sup, monkeypatch, tmp_path, capsys):
    """Audit r1 B-L3: three long-lived attempts in a row that write no new
    checkpoint stop the supervisor; one that does resets the count."""
    base, ckpt = _attempt_loop(sup, monkeypatch, tmp_path)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    clock = [1_000_000.0]
    monkeypatch.setattr(sup.time, "time", lambda: clock[0])
    launches = []

    class Slow:
        pid, returncode = 99, 1

        def poll(self):
            clock[0] += 1200                # each attempt lives 20 min
            return 1

    def popen(args, **k):
        launches.append(args)
        if len(launches) == 2:              # attempt 2 writes a checkpoint
            ckpt.write_bytes(b"x" * 1_000_001)
            os.utime(ckpt, (clock[0] + 600, clock[0] + 600))
        return Slow()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["g0", "0"]) == 1
    out = capsys.readouterr().out
    assert "NO PROGRESS: 3 consecutive attempts died" in out
    assert "CRASH LOOP" not in out
    assert len(launches) == 5                # 1 stall, reset, then 3 stalls
