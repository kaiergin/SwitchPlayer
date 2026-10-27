"""Generation 2 (Kai, 2026-10-27; CURRICULUM.md "Next experiment
(pre-registered) -- Generation 2: DS2 as the master, one distilled
student"): supervise_gen2.py runs ONE student, G2, after DS2, with Test 2's
machinery. Derived from test_supervise_gen1.py: ds2's tests adapted to g2,
G0's (the old master) to the master DS2, fs2's dropped (its wait on DS2,
its vocabulary check, its missing prep). Pins the recipe against DS2's
(supervise_gen1.py ds2), the MASTER_* constants against supervise_gen1.py's
names for DS2's run, what g2 waits for (DS2's marker and its evals' rows 0
and 5-90), the FIXED master and its stops (NEAR RANDOM, ABANDONED), the
prep's once-only run, its retries and its master (vocab.json's master arm,
iterations AND offline steps), the guard / manifest / prep-integrity order
before each launch, the child's environment, the stray-state rules (G2's
own pre-checkpoint snapshots excepted), the singleton, NO PROGRESS and the
row alerts at the prep's ceiling. CPU only, no launches.

No test reads LIVE state (the distillation step's audit r5 HIGH): the `sup`
fixture points every path into DS2's run tree, this tree's results /
probes / prep (REPO) and the process table at empty stand-ins; tests that
need a DS2 or run state build it in tmp. Only the constants tests read the
modules as deployed (`real_sup`, `g1sup`), and only their names."""
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
import distill  # noqa: E402
import muzero  # noqa: E402

G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
FS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-fresh")
G2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen2-distill")
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


def _py(code):
    """`code` in a CPU-only child python, from this tree."""
    import subprocess
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})


@pytest.fixture
def real_sup():
    """The module as deployed: its constants name DS2's LIVE run tree and
    this tree. Only the constants tests read them -- as names, never the
    files behind them."""
    return _load(SUPERVISORS / "supervise_gen2.py", "supervise_gen2")


@pytest.fixture
def sup(real_sup, monkeypatch, tmp_path):
    """The module with every path into DS2's run tree redirected to an
    empty temp folder (DS2 "unfinished, not abandoned"), REPO (the run
    folder, probe jsonls and the prep) an empty temp tree, the manifest
    absent, the process table empty, host memory unknown and the real
    preflight / death snapshot disarmed: no test may depend on the real
    DS2's state or on this tree's own run state -- the preflight runs this
    suite in the run worktree beside G2's live files. Tests that need a
    state build it in tmp (fake_master)."""
    live = tmp_path / "_no_live_runs" / "ds2"
    monkeypatch.setattr(real_sup, "MASTER_TREE", str(live.parent))
    monkeypatch.setattr(real_sup, "MASTER_BASE", str(live))
    for name, rel in (("MASTER_MARKER", "resume.ckpt"),
                      ("MASTER_ABANDONED", "ABANDONED"),
                      ("MASTER_SUPLOG", "supervisor.log"),
                      ("MASTER_SNAPSHOT", "weights-iter090.pth")):
        monkeypatch.setattr(real_sup, name, str(live / rel))
    monkeypatch.setattr(real_sup, "MASTER_PROBES", (
        str(live / "p-acting.jsonl"), str(live / "p.jsonl")))
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
def g1sup():
    return _load(SUPERVISORS / "supervise_gen1.py", "supervise_gen1")


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
def fake_master(sup):
    """A DS2 state built in tmp: (marker, (acting jsonl, raw jsonl))."""
    marker = pathlib.Path(sup.MASTER_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker, tuple(pathlib.Path(p) for p in sup.MASTER_PROBES)


def _complete_master(sup, fake_master, value=700.0):
    marker, probes = fake_master
    marker.write_bytes(b"done")
    _rows(probes[0], sup.MASTER_ROWS, value)
    _rows(probes[1], sup.MASTER_ROWS, value)
    pathlib.Path(sup.MASTER_SNAPSHOT).write_bytes(b"x")


def test_the_sup_fixture_redirects_every_master_path(sup, tmp_path):
    """A MASTER_* path added later and not redirected would make a test
    read DS2's live tree."""
    seen = 0
    for name in dir(sup):
        val = getattr(sup, name)
        if not name.startswith("MASTER_"):
            continue
        for v in (val if isinstance(val, tuple) else (val,)):
            if isinstance(v, str) and (chr(92) in v or "/" in v):
                assert v.startswith(str(tmp_path)), name
                seen += 1
    assert seen == 8     # TREE, BASE, MARKER, SUPLOG, ABANDONED, SNAPSHOT,
    #                      the two PROBES


# --------------------------------------------------------------------------
# the recipe, the constants
# --------------------------------------------------------------------------
def test_recipe_is_ds2s_with_the_gen2_arm_and_prep(real_sup, g1sup):
    sup = real_sup
    assert sup.VARIANTS == {"g2": G2_ARM}
    assert sup.RUNS == {"g2": "mspacman-gen2-distill"}
    assert sup.NAMES == {"g2": "G2"}
    assert sup.SEEDS == {0: "2026-10-27"} and sup.ITERATIONS == 90
    assert sup.TRAIN_STEPS == g1sup.TRAIN_STEPS == 2000
    cmd = sup.command("g2", 0)
    # DS2's command (supervise_gen1.py ds2) with the arm and the prep moved
    want = [sup.PY] + [
        x.replace(g1sup.VARIANTS["ds2"], sup.VARIANTS["g2"])
         .replace(g1sup.PREP_REL, sup.PREP_REL)
        for x in g1sup.command("ds2", 0)[1:]]
    assert cmd == want
    assert cmd[cmd.index("--arms") + 1] == G2_ARM in ablation.ARMS
    assert cmd[cmd.index("--token-vocab") + 1] == (sup.PREP_REL
                                                   + "/vocab.json")
    # always the distillation set: g2 is the distilled student (no fs2)
    assert cmd.count("--distill-set") == 1
    assert cmd[cmd.index("--distill-set") + 1] == sup.PREP_REL + "/set.pkl"
    assert "--resume" not in cmd and "--snapshot-weights" in cmd
    for flag, val in (("--iterations", "90"), ("--train-steps", "2000"),
                      ("--env", ENV_ID), ("--frameskip", "4"),
                      ("--episode-cap", "3000"),
                      ("--episodes-per-iter", "3"), ("--seed", "0"),
                      ("--eval-every", "5"), ("--checkpoint-every", "2")):
        assert cmd[cmd.index(flag) + 1] == val, flag
    i = cmd.index("--ray")
    assert cmd[i:i + 2] == ["--ray", "--cpu-actor"]


def test_paths_the_run_folder_and_the_prep(real_sup):
    sup = real_sup
    rel, base, jsonl, ckpt, deaths = sup.paths("g2", 0)
    assert rel == "benchmarks/results/mspacman-gen2-distill-s0-2026-10-27"
    assert sup.run_name("g2", 0) == rel.split("/")[-1]
    assert jsonl.endswith(os.path.join(
        rel.split("/")[-1], "ablation-mspacmannoframeskip-skip4.jsonl"))
    assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{G2_ARM}"
                         f"-seed0.ckpt")
    assert deaths.endswith("deaths.log")
    assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT           # runs from its own tree
    assert pathlib.Path(sup.MANIFEST) == SUPERVISORS / "gen2_manifest.json"
    assert sup.PREP_REL == "benchmarks/results/gen2-prep-s0-2026-10-27"
    assert pathlib.Path(sup.prep_dir()) == (ROOT / "benchmarks" / "results"
                                            / "gen2-prep-s0-2026-10-27")
    # fs2's wait (DS2's rows in THIS tree, its curve) is gone
    for name in ("DS_ROWS", "DS_CURVE_ROWS", "vocab_is_ds", "VOCAB_CHECK"):
        assert not hasattr(sup, name), name
    assert (sup.MAX_ATTEMPTS, sup.PREP_ATTEMPTS, sup.PREP_RETRY_S) == (40, 3,
                                                                       900)
    assert (sup.NO_PROGRESS_DEATHS, sup.STALL_MIN_S) == (3, 300)
    assert sup.PACE_ALERT_MIN == 25.0 and sup.COMMIT_ALERT_GB == 4.0


def test_the_interpreter_override_is_gen2s(real_sup):
    src = inspect.getsource(real_sup)
    assert 'os.environ.get("GEN2_PY", ' in src and "GEN1_PY" not in src


def test_master_constants_name_ds2s_run(real_sup, g1sup, monkeypatch):
    """Names only: supervise_gen1.paths("ds2", 0) with DS2's tree (Test 2's
    run worktree), the probe's file naming, ablation's snapshot folder."""
    sup = real_sup
    assert sup.MASTER_TREE == r"Y:\SwitchPlayer-gen1b-run"
    assert sup.MASTER_ARM == DS2_ARM == g1sup.VARIANTS["ds2"]
    assert ablation.ARMS[G2_ARM]["_master_arm"] == sup.MASTER_ARM
    assert ablation.ARMS[G2_ARM]["_master_iterations"] == \
        sup.MASTER_ITERATIONS == g1sup.ITERATIONS == 90
    # the arm is DS2's but for its master (and its doc)
    own = ("_master_arm", "_master_iterations", "_doc")
    assert {k: v for k, v in ablation.ARMS[G2_ARM].items()
            if k not in own} == {k: v for k, v in
                                 ablation.ARMS[DS2_ARM].items()
                                 if k not in own}
    monkeypatch.setattr(g1sup, "REPO", sup.MASTER_TREE)
    rel, base, _jsonl, ckpt, _d = g1sup.paths("ds2", 0)
    assert sup.MASTER_RUN == rel.split("/")[-1] == g1sup.run_name(
        "ds2", 0) == "mspacman-gen1-distill-s0-2026-10-25"
    assert sup.MASTER_BASE == base
    assert sup.MASTER_MARKER == ckpt
    assert sup.MASTER_SUPLOG == os.path.join(base, "supervisor.log")
    # the PATH of DS2's flag; the message is MASTER_ABANDONED_MSG
    assert sup.MASTER_ABANDONED == os.path.join(base, "ABANDONED")
    assert sup.TAG == g1sup.TAG and sup.ENV_ID == g1sup.ENV_ID
    assert sup.MASTER_SNAPSHOT == os.path.join(
        base, f"snapshots{sup.TAG}-{DS2_ARM}-seed0", "weights-iter090.pth")
    # probe_exhaustive_eval writes HERE/probe_exhaustive_eval-<the
    # snapshot's grandparent folder><suffix>.jsonl; DS2's hand-off ran it in
    # DS2's tree (supervise_gen1.probe_jsonls: acting, then raw)
    probes = os.path.join(sup.MASTER_TREE, "benchmarks", "probes")
    assert sup.MASTER_PROBES == (
        os.path.join(probes, f"probe_exhaustive_eval-{sup.MASTER_RUN}"
                             f"-acting.jsonl"),
        os.path.join(probes, f"probe_exhaustive_eval-{sup.MASTER_RUN}"
                             f".jsonl"))
    assert list(sup.MASTER_PROBES) == g1sup.probe_jsonls("ds2", 0)[:2]
    # DS2's rows: row 0 (the end of its offline phase) and 5-90 -- what
    # fs2 waited for in Test 2
    assert sup.MASTER_ROWS == (0,) + tuple(range(5, 95, 5)) == g1sup.DS_ROWS
    assert sup.MASTER_ITERATIONS == 90


def test_master_offline_steps_are_ds2s(real_sup):
    """vocab.json's master_offline_steps (distill.prep writes it: the
    master's spec's offline steps) must be DS2's 20,000; G0 (not a student)
    and FS2 (fresh) record 0."""
    sup = real_sup
    assert sup.MASTER_OFFLINE_STEPS == 20000
    assert distill.master_offline_steps(ablation.ARMS[DS2_ARM]) == \
        ablation.ARMS[DS2_ARM]["_distill_offline_steps"] == \
        sup.MASTER_OFFLINE_STEPS
    assert distill.master_offline_steps(ablation.ARMS[G0_ARM]) == 0
    assert distill.master_offline_steps(ablation.ARMS[FS2_ARM]) == 0
    src = inspect.getsource(distill.prep)
    assert '"master_offline_steps": offline' in src
    assert "blob, arm, mspec, offline = master_checks(" in src
    assert "offline = master_offline_steps(mspec)" in inspect.getsource(
        distill.master_checks)


def test_the_alert_bounds(real_sup):
    sup = real_sup
    with ablation.arm_config(G2_ARM):
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
    """Every module name an import statement names at any depth: each name
    of `import a, b.c` (a, b), the module of `from m import x` (m) and its
    imported names (x: a submodule perhaps; audit r1 B-L1)."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
            out.update(a.name for a in node.names)
    return out


def test_imports_sees_every_name():
    src = ("import a, b.c" + NL + "from d.e import f, g as h" + NL
           + "def x():" + NL + "    import i" + NL)
    assert _imports(src) == {"a", "b", "d", "f", "g", "i"}


def test_supervisor_pins_what_the_runs_and_preflight_use(real_sup, g1sup):
    import re
    sup = real_sup
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert len(set(sup.SUITES)) == len(sup.SUITES) == 22
    # every suite supervise_gen1.py ran, after this supervisor's and
    # generation 2's (first: -x stops on them before the slow suites)
    assert sup.SUITES[:2] == ["tests/test_supervise_gen2.py",
                              "tests/test_gen2.py"]
    assert sup.SUITES[2:] == g1sup.SUITES
    assert set(g1sup.AUDITED) <= set(sup.AUDITED)
    assert set(sup.AUDITED) - set(g1sup.AUDITED) == {
        "tests/test_supervise_gen2.py", "tests/test_gen2.py",
        "benchmarks/supervisors/supervise_gen2.py"}
    assert [r for r in sup.AUDITED if not r.endswith(".py")] == [
        "pytest.ini", "benchmarks/probes/step0_prefix_bn/states.npz"]
    for rel in ("experiments/ablation.py", "experiments/distill.py",
                "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py",
                "benchmarks/probes/probe_prefix_bn.py",
                "benchmarks/supervisors/supervise_gen1.py"):
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
    # the code the run, the prep and the probes execute: every repo module
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
                "benchmarks/supervisors/supervise_gen2.py",
                "benchmarks/supervisors/supervise_gen1.py",
                "tests/test_supervise_gen2.py", "tests/test_gen2.py"):
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
    for argv in ([], ["g2"], ["0"], ["g2", "1"], ["ds2", "0"], ["fs2", "0"],
                 ["g0", "0"], ["G2", "0"], ["ds", "0"], ["g2", "x"],
                 ["g2", "0", "1"], ["0", "g2"], ["g2", "0", "--no-wait"],
                 ["g2", "0", "--wait"]):
        assert sup.main(argv) == 2, argv
    out = capsys.readouterr().out
    assert "usage: supervise_gen2.py <variant: ['g2']> <seed: [0]>" in out


def test_no_wait_and_fs2_are_gone(real_sup):
    """Generation 2 has no fallback master and no fresh twin: nothing of
    the --no-wait override, of fs2 (its wait on DS2, its vocabulary check)
    or of the G0 master survives; no per-variant branch is left."""
    src = inspect.getsource(real_sup)
    for word in ("no_wait", "NO-WAIT", "sanctioned", "stale_pick",
                 "unsettled", "TKX", "TK-90", "student-distill",
                 "vocab_is_ds", "VOCAB_CHECK", "DS_ROWS", "DS_CURVE_ROWS",
                 "G0_", "g0_mean", "variant ==", '"fs2"', "gen1-fresh",
                 "gen1-prep", "gen1_manifest", "torch"):
        assert word not in src, word


def _ds2_like(path, **change):
    """A DS2-like final snapshot (no nets: distill.master_checks reads only
    its arm, spec, steps and acting states): DS2's arm as student_arm
    leaves it -- a student vocabulary of TK's 36, a constant ceiling --
    after 20,000 offline + 90 x 2,000 steps."""
    import torch
    spec = {**ablation.ARMS[DS2_ARM],
            "TOKEN_VOCAB": tuple(t for t in ablation.ARMS[DS2_ARM][
                "TOKEN_VOCAB"] if t not in ((0, 8), (0, 32))),
            "_token_vocab_sha256": "a" * 64, "_vocab_set_sha256": "b" * 64,
            "REWARD_GROWTH_START": 11.25, "REWARD_GROWTH_CEILING": 11.25,
            "_student_ceiling": 11.25}
    blob = {"arm": DS2_ARM, "spec": spec, "training_steps": 200000,
            "iteration": 90, "acting_states": {"representation": {}}}
    for k, v in change.items():
        if k in blob:
            blob[k] = v
        else:
            spec[k] = v
    torch.save(blob, path)
    return str(path)


@pytest.fixture
def ds2_like(real_sup, monkeypatch, tmp_path):
    """The quick probe's master checks on a DS2-like snapshot in tmp, not
    the live DS2 (the preflight's real probe reads the real one)."""
    monkeypatch.setattr(real_sup, "MASTER_SNAPSHOT",
                        _ds2_like(tmp_path / "weights-iter090.pth"))
    return real_sup


def test_supervisor_quick_probe_passes(ds2_like):
    r = _py(ds2_like.quick("g2"))
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_quick_probe_names_the_real_master_snapshot(real_sup):
    """The deployed probe ends with the prep's master checks on DS2's real
    final snapshot (audit r1 A-L1 / B-L2)."""
    probe = real_sup.quick("g2")
    assert probe.endswith(f"distill.master_checks("
                          f"{real_sup.MASTER_SNAPSHOT!r}, 90)")
    assert real_sup.MASTER_SNAPSHOT.endswith(
        "mspacman-gen1-distill-s0-2026-10-25" + os.sep
        + f"snapshots-mspacmannoframeskip-skip4-{DS2_ARM}-seed0" + os.sep
        + "weights-iter090.pth")


@pytest.mark.parametrize("change, why", [
    # an edit to DS2's arm in this tree (its _doc included): the prep
    # would refuse DS2 three times
    ({"_doc": "edited"}, "beyond the student keys: ['_doc']"),
    ({"PER_BETA_END": 0.5}, "beyond the student keys"),
    ({"training_steps": 180000}, "after 20000 offline steps"),
    ({"iteration": 89}, "after 20000 offline steps"),
    ({"REWARD_GROWTH_START": 1.0}, "constant _student_ceiling"),
    ({"acting_states": {}}, "no acting_states"),
    ({"arm": "nope"}, "arm 'nope' unknown"),
])
def test_the_quick_probe_fails_on_a_master_the_prep_would_refuse(
        real_sup, monkeypatch, tmp_path, change, why):
    monkeypatch.setattr(real_sup, "MASTER_SNAPSHOT",
                        _ds2_like(tmp_path / "m.pth", **change))
    r = _py(real_sup.quick("g2"))
    assert r.returncode != 0 and why in r.stderr, r.stderr[-2000:]


def test_the_quick_probe_tells_the_arms_apart(ds2_like):
    """As the student, DS2's arm (its master is G0), FS2's (fresh) and G0's
    (no flag) fail g2's probe; as the master, G0's (no flag) and FS2's
    (fresh) fail it."""
    real_sup = ds2_like
    probe = real_sup.quick("g2")
    for want in ("spec['PREFIX_BN_PER_STEP'] is True",
                 "master['PREFIX_BN_PER_STEP'] is True",
                 "muzero.PREFIX_BN_PER_STEP is False",
                 "master['_student'] == 'distill'",
                 f"spec['_master_arm'] == {DS2_ARM!r}",
                 "spec['_master_iterations'] == 90",
                 "distill.master_offline_steps(master) == 20000",
                 "hasattr(ablation, 'student_master_vocab')",
                 "hasattr(distill, 'master_spec')",
                 "master['_ceiling_from_prep'] is True",
                 "distill.master_checks("):
        assert want in probe, want
    student, master = f"ARMS[{G2_ARM!r}]", f"ARMS[{DS2_ARM!r}]"
    assert probe.count(student) == 1 and probe.count(master) == 1
    for old, other in ((student, DS2_ARM), (student, FS2_ARM),
                       (student, G0_ARM), (master, G0_ARM),
                       (master, FS2_ARM)):
        swapped = probe.replace(old, f"ARMS[{other!r}]")
        assert swapped != probe
        r = _py(swapped)
        assert r.returncode != 0 and ("AssertionError" in r.stderr
                                      or "KeyError" in r.stderr), (old,
                                                                   other)


@pytest.mark.parametrize("break_", ["arm off", "master off", "default on",
                                    "master offline"])
def test_each_flag_assert_of_the_quick_probe_bites_alone(ds2_like, break_):
    """Audit r2 A-NIT-1 (Test 2's), for generation 2: with everything else
    the real arms, each of the three flag asserts alone fails the probe --
    the student's flag off, the master DS2's flag off, the module default
    on -- and so does the master's offline steps (so a vacuous rewrite of
    any of them is caught)."""
    real_sup = ds2_like
    probe = real_sup.quick("g2")
    prelude = {
        "arm off": f"ablation.ARMS[{G2_ARM!r}]['PREFIX_BN_PER_STEP'] = "
                   f"False; ",
        "master off": f"ablation.ARMS[{DS2_ARM!r}]['PREFIX_BN_PER_STEP'] = "
                      f"False; ",
        "default on": "muzero.PREFIX_BN_PER_STEP = True; ",
        "master offline": f"ablation.ARMS[{DS2_ARM!r}]"
                          f"['_distill_offline_steps'] = 4000; "}[break_]
    anchor = f"spec = ablation.ARMS[{G2_ARM!r}]; "
    assert probe.count(anchor) == 1
    broken = probe.replace(anchor, prelude + anchor)
    r = _py(broken)
    assert r.returncode != 0 and "AssertionError" in r.stderr, r.stderr
    ok = _py(probe)
    assert ok.returncode == 0, ok.stderr[-2000:]   # the control


def test_the_quick_probe_is_bounded(monkeypatch):
    """Audit r1 B-N3 (PB's r1 B-N5): a hung quick probe is a PREFLIGHT
    not-ready reason (the loop retries), never a preflight stalled for
    good; the suites are not run after it. The real preflight, every
    subprocess stubbed."""
    import subprocess
    pre = _load(SUPERVISORS / "supervise_gen2.py", "supervise_gen2_preflight")
    monkeypatch.setattr(pre, "manifest_check", lambda: (True, ""))
    calls, outcome = [], ["hang"]

    def run(args, **k):
        calls.append((list(args), k))
        if args[1] == "-c" and outcome[0] == "hang":
            raise subprocess.TimeoutExpired(args, k["timeout"])
        if args[1] == "-c" and outcome[0] == "fail":
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="x" + NL + "AssertionError")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pre.subprocess, "run", run)
    assert pre.preflight("g2") == (False, "quick probe timed out")
    assert len(calls) == 1                         # no suites after it
    args, k = calls[0]
    assert args == [pre.PY, "-c", pre.quick("g2")]
    assert k["timeout"] == pre.QUICK_TIMEOUT_S == 900
    assert k["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    outcome[0], calls[:] = "fail", []
    assert pre.preflight("g2") == (False, "quick probe: AssertionError")
    assert len(calls) == 1
    outcome[0], calls[:] = "ok", []
    assert pre.preflight("g2") == (True, "")
    assert [a[1:3] for a, _k in calls] == [["-c", pre.quick("g2")],
                                           ["-m", "pytest"]]
    # the suites: all of them, this file first, CPU only, in this tree
    args, k = calls[1]
    assert args[-len(pre.SUITES):] == pre.SUITES
    assert args[-len(pre.SUITES)] == "tests/test_supervise_gen2.py"
    assert k["env"]["CUDA_VISIBLE_DEVICES"] == "-1" and k["cwd"] == pre.REPO


# --------------------------------------------------------------------------
# the predecessor: the master DS2
# --------------------------------------------------------------------------
def test_the_predecessor_is_ds2_in_its_own_tree(sup):
    name, marker, probes, suplog, script, pvar, flag = sup.predecessor("g2",
                                                                       0)
    assert (name, marker, suplog, script, pvar, flag) == (
        "DS2", sup.MASTER_MARKER, sup.MASTER_SUPLOG, "supervise_gen1.py",
        "ds2", sup.MASTER_ABANDONED)
    # both of DS2's exact evals, each with row 0 (its offline phase's end)
    assert probes == [(p, sup.MASTER_ROWS) for p in sup.MASTER_PROBES]
    assert all(rows[0] == 0 and rows[1:] == tuple(range(5, 95, 5))
               for _p, rows in probes)


def test_g2_waits_for_ds2s_marker_and_every_row_of_both_evals(sup,
                                                              fake_master):
    marker, probes = fake_master
    assert any("no checkpoint" in m for m in sup.pred_missing("g2", 0))
    marker.write_bytes(b"x" * 1_000_000)
    assert any("full size" in m for m in sup.pred_missing("g2", 0))
    marker.write_bytes(b"done")
    _rows(probes[0], range(5, 95, 5))
    _rows(probes[1], range(5, 90, 5), extra=[
        {"iteration": 90, "env": ENV_ID, "max_noop": 2},    # a smoke row
        {"iteration": 90, "env": "Pong-v4"},
        {"iteration": 0, "env": ENV_ID, "max_noop": 2}])
    # row 0 is waited for in both: G0 had none, DS2 has one
    assert sup.pred_missing("g2", 0) == ["p-acting.jsonl rows [0]",
                                         "p.jsonl rows [0, 90]"]
    _rows(probes[0], [0])
    _rows(probes[1], [0, 90])
    assert sup.pred_missing("g2", 0) == []
    # the master's offline curve (rows 1-4, fs2's wait in Test 2) is not
    # waited for
    assert not any(set(rows) & {1, 2, 3, 4}
                   for _p, rows in sup.predecessor("g2", 0)[2])


def test_wait_external_blocks_and_alerts(sup, fake_master, monkeypatch,
                                         capsys):
    marker, probes = fake_master
    marker.write_bytes(b"done")
    os.utime(marker, (1, 1))                           # done long ago
    asked = []
    monkeypatch.setattr(sup, "live_scripts",
                        lambda *a: asked.append(a) or [])

    def rows_land(k):
        if k == 2:
            _rows(probes[0], sup.MASTER_ROWS)
            _rows(probes[1], sup.MASTER_ROWS)

    sleep = _bounded(rows_land)
    sleeps = sleep.calls
    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("g2", 0) == "ready"
    out = capsys.readouterr().out
    assert "WAITING for DS2 before g2 s0" in out
    assert "ALERT: DS2 completed" in out and "evals still lack rows" in out
    assert "DS2 complete and evaluated; g2 s0 may start" in out
    # the master's snapshot is checked while waiting
    assert "master snapshot missing: " + sup.MASTER_SNAPSHOT in out
    assert "keep DS2's run tree" in out
    assert sleeps == [60, 60]
    # unfinished and undriven
    marker.write_bytes(b"x" * 1_000_000)
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    out = capsys.readouterr().out
    assert "nothing drives it: no live supervise_gen1.py ds2" in out
    assert asked[-1] == ("supervise_gen1.py", "ds2")
    assert "master snapshot missing" not in out        # no marker yet
    # driven, but its log silent for hours
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [5])
    pathlib.Path(sup.MASTER_SUPLOG).write_text("x")
    os.utime(sup.MASTER_SUPLOG, (1, 1))
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    assert "is alive but its log is silent" in capsys.readouterr().out
    # its supervisor gave up
    pathlib.Path(sup.MASTER_SUPLOG).write_text(
        "LAUNCHED ds2 s0 attempt 3" + NL + "NO PROGRESS: 3 x" + NL)
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    assert "DS2's supervisor gave up -- g2 keeps waiting" in \
        capsys.readouterr().out


def test_wait_external_stops_when_this_run_is_abandoned(sup, fake_master,
                                                        monkeypatch):
    base = pathlib.Path(sup.paths("g2", 0)[1])
    base.mkdir(parents=True)
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: (base / "ABANDONED").write_text("")))
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [1])
    assert sup.wait_external("g2", 0) == "abandoned"


@pytest.mark.parametrize("master_complete", [False, True])
def test_an_abandoned_ds2_is_master_abandoned(sup, fake_master, monkeypatch,
                                              capsys, master_complete):
    """No fallback: g2 stops (PREP FAILED, the hand-off's failed state)
    whether DS2 was abandoned before or after completing."""
    if master_complete:
        _complete_master(sup, fake_master)
    sleep = _bounded(
        lambda k: pathlib.Path(sup.MASTER_ABANDONED).write_text(""))
    sleeps = sleep.calls
    monkeypatch.setattr(sup.time, "sleep", sleep)
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [])
    if master_complete:
        pathlib.Path(sup.MASTER_ABANDONED).write_text("")
    assert sup.wait_external("g2", 0) == "failed"
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "MASTER ABANDONED" in l]
    # the message, never the flag's path
    assert line == ["PREP FAILED: " + sup.MASTER_ABANDONED_MSG]
    assert len(sleeps) == (0 if master_complete else 1)
    # main stops there: no preflight, no prep, no launch
    monkeypatch.setattr(sup, "preflight", _stop)
    monkeypatch.setattr(sup, "run_prep", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["g2", "0"]) == 1
    assert "PREP FAILED: MASTER ABANDONED" in capsys.readouterr().out


def test_master_abandoned_is_the_flags_path_and_msg_the_message(real_sup):
    sup = real_sup
    assert os.path.basename(sup.MASTER_ABANDONED) == "ABANDONED"
    assert os.path.dirname(sup.MASTER_ABANDONED) == sup.MASTER_BASE
    assert sup.MASTER_ABANDONED_MSG.startswith(
        "MASTER ABANDONED: DS2's run folder holds an ABANDONED file")
    assert "no generation 2 on it" in sup.MASTER_ABANDONED_MSG
    assert "Kai's call" in sup.MASTER_ABANDONED_MSG
    # every existence check reads the path; every stop prints the message
    src = inspect.getsource(sup)
    assert src.count("os.path.exists(MASTER_ABANDONED)") == 2
    assert "raise SystemExit(MASTER_ABANDONED_MSG)" in src
    assert 'print(f"PREP FAILED: {MASTER_ABANDONED_MSG}"' in src


def test_an_undriven_ds2_is_named_and_its_master_gate_stop_is_kais_call(
        sup, fake_master, monkeypatch, capsys):
    """DS2's own supervisor (supervise_gen1.py ds2) has a master gate:
    stopped there it is Kai's call, not "rerun its launcher" (Test 2's
    audit r1 B-NIT2, now on the master's log)."""
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    assert "no live supervise_gen1.py ds2 -- rerun its launcher" in \
        capsys.readouterr().out
    suplog = pathlib.Path(sup.MASTER_SUPLOG)
    for stop in ("PREP FAILED: MASTER NEAR RANDOM: G0's deployed mean ...",
                 "PREP FAILED: MASTER ABANDONED: G0's run folder ..."):
        suplog.write_text("preflight ok (manifest + quick probe + 20 suites)"
                          + NL + stop + NL)
        with pytest.raises(Stop):
            sup.wait_external("g2", 0)
        out = capsys.readouterr().out
        assert "DS2's supervisor stopped on" in out and stop[:30] in out
        assert "Kai's call" in out and "-- rerun its launcher" not in out
        assert "DS2's supervisor gave up -- g2 keeps waiting" in out
    # a later preflight (a relaunch after the fix) clears it
    suplog.write_text("PREP FAILED: MASTER NEAR RANDOM: ..." + NL
                      + "preflight ok (...)" + NL)
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    assert "no live supervise_gen1.py ds2" in capsys.readouterr().out
    # another give-up keeps the rerun advice
    suplog.write_text("preflight ok (...)" + NL
                      + "PREP FAILED after 3 attempts -- see ..." + NL)
    with pytest.raises(Stop):
        sup.wait_external("g2", 0)
    assert "no live supervise_gen1.py ds2" in capsys.readouterr().out
    # DS2's flag: wait_external stops on it first (MASTER ABANDONED);
    # pred_unattended names it too
    pathlib.Path(sup.MASTER_ABANDONED).write_text("")
    assert sup.pred_unattended("g2", 0).startswith("DS2 is ABANDONED")


# --------------------------------------------------------------------------
# the master: fixed to DS2, after the NEAR RANDOM rule
# --------------------------------------------------------------------------
def test_choose_master_is_ds2s_final_snapshot(sup, fake_master):
    marker, probes = fake_master
    with pytest.raises(SystemExit, match="DS2 has no completion marker"):
        sup.choose_master()
    marker.write_bytes(b"done")
    with pytest.raises(SystemExit, match="rows 60-90 are incomplete"):
        sup.choose_master()
    _rows(probes[0], (0,) + tuple(range(5, 90, 5)), 900.0)
    _rows(probes[1], range(0, 95, 5), 900.0)       # raw rows do not count
    with pytest.raises(SystemExit, match="rows 60-90 are incomplete"):
        sup.choose_master()
    _rows(probes[0], [90], 900.0)
    with pytest.raises(SystemExit, match="final snapshot is missing"):
        sup.choose_master()
    pathlib.Path(sup.MASTER_SNAPSHOT).write_bytes(b"x")
    assert sup.choose_master() == {
        "which": "DS2", "snapshot": sup.MASTER_SNAPSHOT, "iterations": 90,
        "master_mean_60_90": pytest.approx(900.0)}


@pytest.mark.parametrize("deployed, raw, ok", [
    ((502.08,) * 7, 502.08, True),      # exactly g = 0.2: passes
    # a decimal 502.08 mean whose float sum lands a hair below
    ((502.08,) * 6 + (502.08 - 1e-12,), 0.0, True),
    ((502.08,) * 6 + (502.079,), 9999.0, False),   # 1/7000 below: fails
    ((502.07,) * 7, 9999.0, False),     # the DEPLOYED mean decides
    ((9999.0,) * 7, 100.0, True),
    ((307.3,) * 7, 307.3, False)])
def test_the_near_random_rule(sup, fake_master, deployed, raw, ok):
    marker, probes = fake_master
    marker.write_bytes(b"done")
    pathlib.Path(sup.MASTER_SNAPSHOT).write_bytes(b"x")
    _rows(probes[0], range(0, 60, 5), 9999.0)      # rows before 60: unread
    for i, v in zip(range(60, 95, 5), deployed):
        _rows(probes[0], [i], v)
    _rows(probes[1], range(0, 95, 5), raw)
    mean = sum(deployed) / 7
    if ok:
        assert sup.choose_master()["master_mean_60_90"] == pytest.approx(
            mean)
    else:
        with pytest.raises(SystemExit) as e:
            sup.choose_master()
        assert str(e.value).startswith("MASTER NEAR RANDOM: DS2's")
        assert f"{mean:.2f}" in str(e.value) and "502.08" in str(e.value)


def test_the_window_is_rows_60_to_90(sup, fake_master):
    marker, probes = fake_master
    marker.write_bytes(b"done")
    pathlib.Path(sup.MASTER_SNAPSHOT).write_bytes(b"x")
    _rows(probes[0], [0], 99999.0)                 # row 0 is no window row
    for i, v in zip(range(60, 95, 5), (800, 900, 1000, 1100, 1200, 700,
                                       1300)):
        _rows(probes[0], [i], float(v))
    assert sup.choose_master()["master_mean_60_90"] == pytest.approx(1000.0)
    # a smoke row (max_noop) or another env's never stands in
    probes[0].write_text("")
    _rows(probes[0], range(60, 95, 5), 900.0, extra=[
        {"iteration": 60, "env": ENV_ID, "max_noop": 2, "exact_mean": 1.0},
        {"iteration": 65, "env": "Pong-v4", "exact_mean": 1.0}])
    assert sup.choose_master()["master_mean_60_90"] == pytest.approx(900.0)


def test_an_abandoned_ds2_is_never_the_master(sup, fake_master):
    _complete_master(sup, fake_master, 2000.0)
    pathlib.Path(sup.MASTER_ABANDONED).write_text("")
    with pytest.raises(SystemExit) as e:
        sup.choose_master()
    assert str(e.value) == sup.MASTER_ABANDONED_MSG


def test_master_files_missing(sup, fake_master):
    marker, _ = fake_master
    assert sup.master_files_missing() == []          # DS2 unfinished
    marker.write_bytes(b"done")
    assert sup.master_files_missing() == [sup.MASTER_SNAPSHOT]
    pathlib.Path(sup.MASTER_SNAPSHOT).write_bytes(b"x")
    assert sup.master_files_missing() == []


# --------------------------------------------------------------------------
# the prep
# --------------------------------------------------------------------------
PICK = {"which": "DS2", "snapshot": "m.pth", "iterations": 90,
        "master_mean_60_90": 700.0}
GONE = "<gone>"         # a vocab.json key _write_prep leaves out


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
    full = {"set_sha256": sha, "master_arm": DS2_ARM,
            "master_iterations": 90, "master_offline_steps": 20000,
            "master_snapshot": sup.MASTER_SNAPSHOT,
            "student_ceiling": 7.5, "vocab": [[0, 4], [1, 4]], **doc}
    (d / "vocab.json").write_text(json.dumps(
        {k: v for k, v in full.items() if v != GONE}))


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
    assert "MASTER DS2: m.pth (deployed 60-90 700.00, g 0.403)" in out
    assert "PREP attempt 1 failed" in out and "PREP done (attempt 2)" in out
    assert calls == ["choose"] and len(runs) == 2
    args, k = runs[0]
    # the master's offline steps are distill's to read (from its spec)
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


def test_the_real_gate_picks_ds2s_snapshot_into_master_json(
        sup, fake_master, monkeypatch, capsys):
    """The real choose_master inside run_prep: master.json carries DS2's
    pick under master_mean_60_90 (not G0's g0_mean_60_90) and distill.py
    gets DS2's snapshot at 90 iterations."""
    _complete_master(sup, fake_master, 1231.0)
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))], seen)
    assert sup.run_prep({}) is True
    out = capsys.readouterr().out
    assert (f"MASTER DS2: {sup.MASTER_SNAPSHOT} (deployed 60-90 1231.00, "
            f"g 0.948)") in out
    pick = json.loads(pathlib.Path(sup.prep_dir(), "master.json")
                      .read_text())
    assert pick == {"which": "DS2", "snapshot": sup.MASTER_SNAPSHOT,
                    "iterations": 90, "master_mean_60_90": 1231.0}
    args = seen[0][0]
    assert args[args.index("--master") + 1] == sup.MASTER_SNAPSHOT
    assert args[args.index("--master-iterations") + 1] == "90"


def test_read_pick_needs_the_master_key(sup, tmp_path):
    """master_mean_60_90 is the pick's key: Test 2's g0_mean_60_90 pick (a
    generation-1 master.json) is no generation-2 pick."""
    path = str(tmp_path / "master.json")
    sup.write_pick(path, PICK)
    assert sup.read_pick(path) == PICK
    old = {**{k: v for k, v in PICK.items() if k != "master_mean_60_90"},
           "which": "G0", "g0_mean_60_90": 700.0}
    sup.write_pick(path, old)
    assert sup.read_pick(path) is None
    for k in PICK:
        sup.write_pick(path, {j: v for j, v in PICK.items() if j != k})
        assert sup.read_pick(path) is None, k
    sup.write_pick(path, [PICK])
    assert sup.read_pick(path) is None


def test_a_finished_prep_with_a_gen1_master_json_is_refused(
        sup, prep_env, monkeypatch, capsys):
    _write_prep(sup)
    sup.write_pick(os.path.join(sup.prep_dir(), "master.json"), {
        "which": "G0", "snapshot": "m.pth", "iterations": 90,
        "g0_mean_60_90": 700.0})
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    assert "PREP FAILED: a finished prep without a readable master.json" in \
        capsys.readouterr().out


def test_a_restarted_prep_runs_the_master_gate_again(sup, prep_env,
                                                     monkeypatch):
    """The master is fixed: a prep (re)started by a new supervisor runs the
    gate again and rewrites master.json; a torn one is no obstacle."""
    calls = prep_env
    d = pathlib.Path(sup.prep_dir())
    d.mkdir(parents=True)
    (d / "master.json").write_text('{"which": "DS2", "snap')    # torn
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))], seen)
    assert sup.run_prep({}) is True and calls == ["choose"]
    assert sup.read_pick(str(d / "master.json")) == PICK
    assert seen[0][0][3] == "m.pth" and seen[0][0][5] == "90"


@pytest.mark.parametrize("why, start", [
    ("MASTER NEAR RANDOM: DS2's deployed mean ...", "PREP FAILED: MASTER "
                                                    "NEAR RANDOM"),
    ("MASTER ABANDONED: x", "PREP FAILED: MASTER ABANDONED"),
    ("DS2's deployed rows 60-90 are incomplete", "PREP FAILED: DS2's")])
def test_a_master_gate_failure_is_prep_failed(sup, prep_env, monkeypatch,
                                              capsys, why, start):
    monkeypatch.setattr(sup, "choose_master", lambda: (_ for _ in ()).throw(
        SystemExit(why)))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert any(l.startswith(start) for l in out.splitlines()), out
    assert not pathlib.Path(sup.prep_dir(), "master.json").exists()


def test_near_random_stops_the_real_gate(sup, fake_master, monkeypatch,
                                         capsys):
    """The real choose_master inside run_prep: the line the hand-off reads
    as a failed state, with the mean; nothing starts."""
    _complete_master(sup, fake_master, 480.5)
    monkeypatch.setattr(sup, "wait_no_trainer", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("PREP FAILED")]
    assert line and line[0].startswith("PREP FAILED: MASTER NEAR RANDOM: "
                                       "DS2's")
    assert "480.50" in line[0] and "Kai's call" in line[0]
    pathlib.Path(sup.MASTER_ABANDONED).write_text("")
    assert sup.run_prep({}) is False
    assert ("PREP FAILED: " + sup.MASTER_ABANDONED_MSG) in \
        capsys.readouterr().out.splitlines()


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
            f.write("recorded 3 games\nSystemExit: m.pth: its spec differs "
                    "from its arm's beyond the student keys: ['X']\n\n")
    _popen(monkeypatch, sup, [(1, exits)] * 3)
    assert sup.run_prep({}) is False
    out = capsys.readouterr().out
    assert out.count("its last line: SystemExit: m.pth: its spec differs "
                     "from its arm's beyond the student keys: ['X']") == 3
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


def test_a_prep_of_another_master_is_moved_aside_and_redone(
        sup, prep_env, monkeypatch, capsys):
    """A prep of DS2 at its iterations but without its offline steps (a
    G0-style record) is not intact: before a launch, redone."""
    calls = prep_env
    _write_prep(sup, master_offline_steps=0)
    sup.write_pick(os.path.join(sup.prep_dir(), "master.json"), PICK)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep({}) is True
    out = capsys.readouterr().out
    assert "PREP damaged" in out and "0 offline steps" in out
    assert calls == ["choose"] and sup.set_intact() == (True, "")


def test_a_prep_that_never_checks_out_fails(sup, prep_env, monkeypatch,
                                            capsys):
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup, good=False))] * 3)
    assert sup.run_prep({}) is False
    assert "does not match" in capsys.readouterr().out


@pytest.mark.parametrize("doc, why", [
    ({}, ""),
    ({"master_arm": G0_ARM}, "not DS2's final snapshot"),   # Test 2's prep
    ({"master_arm": G2_ARM}, "not DS2's final snapshot"),
    ({"master_arm": GONE}, "not DS2's final snapshot"),
    ({"master_iterations": 270}, "not DS2's final snapshot"),
    ({"master_iterations": GONE}, "not DS2's final snapshot"),
    # the master's offline steps (new in generation 2)
    ({"master_offline_steps": 0}, "and 0 offline steps, not DS2's final"),
    ({"master_offline_steps": 4000}, "and 4000 offline steps"),
    ({"master_offline_steps": 20001}, "not DS2's final snapshot"),
    ({"master_offline_steps": "20000"}, "and '20000' offline steps"),
    ({"master_offline_steps": None}, "and None offline steps"),
    ({"master_offline_steps": GONE}, "and None offline steps"),
    # the master's own snapshot (audit r1 B-L3: another final snapshot of
    # DS2's arm, a seed repeat's, is not this prep's master)
    ({"master_snapshot": "other-ds2.pth"},
     "master snapshot is 'other-ds2.pth', not "),
    ({"master_snapshot": GONE}, "master snapshot is None"),
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


def test_set_intact_names_the_master_it_found(sup):
    _write_prep(sup, master_arm=G0_ARM, master_offline_steps=0)
    assert sup.set_intact() == (False, (
        f"vocab.json's master is {G0_ARM!r} after 90 iterations and 0 "
        f"offline steps, not DS2's final snapshot"))
    pathlib.Path(sup.prep_dir(), "vocab.json").write_text("[1]")
    assert sup.set_intact() == (False, "vocab.json is not a JSON object")


def test_a_launched_run_never_redoes_its_prep(sup, prep_env, monkeypatch,
                                             capsys):
    """A launched G2 is bound to the set; a transient lock or real damage
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
    """vocab.json missing after G2 was launched: no prep at all."""
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


# --------------------------------------------------------------------------
# the trainer guard, one supervisor per run
# --------------------------------------------------------------------------
def _psutil(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


GUARD_PROCS = [
    (1, "python.exe", ["py", "Y:" + chr(92) + "run" + chr(92) + "experiments"
                       + chr(92) + "ablation.py", "--arms", G2_ARM]),
    (2, "python.exe", ["py", "experiments/ablation.py", "--arms", DS2_ARM,
                       "--seed", "1"]),
    (3, "Python3.12", ["py", "experiments/ablation.py", "--arms", "pong-x"]),
    (4, "python.exe", ["py", "experiments" + chr(92) + "distill.py",
                       "--master", "m.pth"]),                    # the prep
    (5, "python.exe", ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                       "x.pth"]),
    (6, "python.exe", ["py", "-m", "pytest", "tests/test_gen2.py"]),
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
    (17, "python.exe", ["py", "benchmarks/supervisors/supervise_gen2.py",
                        "g2", "0"]),
    (18, "python.exe", ["py", "-m", "distill"])]


def test_live_trainers_finds_any_run_or_prep(sup, g1sup, monkeypatch):
    """is_trainer as supervise_gen1.py's: the same verdict on every command
    line; the prep counts; python only; unreadable not counted."""
    fresh = _load(SUPERVISORS / "supervise_gen2.py", "supervise_gen2_procs")
    monkeypatch.setattr(sup, "_procs", fresh._procs)
    _psutil(monkeypatch, [{"pid": p, "name": n, "cmdline": c}
                          for p, n, c in GUARD_PROCS])
    assert sup.live_trainers() == [1, 2, 3, 4, 12, 13, 14, 18]
    for _p, _n, cmd in GUARD_PROCS:
        if cmd is not None:
            assert sup.is_trainer(cmd) == g1sup.is_trainer(cmd), cmd
    assert sup.TRAINER_SCRIPTS == g1sup.TRAINER_SCRIPTS
    assert sup.TRAINER_MODULES == g1sup.TRAINER_MODULES


def _procs(monkeypatch, sup, procs):
    monkeypatch.setattr(sup, "_procs", lambda: iter(procs))


def test_live_scripts_match_the_exact_name_and_variant(sup, monkeypatch):
    procs = [(1, ["py", r"C:\x\supervise_gen2.py", "g2", "0"]),
             (2, ["py", "benchmarks/supervisors/supervise_gen2.py", "g2",
                  "0"]),
             (3, ["py", "supervise_gen2.py.bak", "g2", "0"]),
             (4, ["py", "benchmarks/supervisors/supervise_gen1.py", "ds2",
                  "0"]),
             (5, ["py", "supervise_gen2.py", "ds2", "0"]),
             (6, ["py", "experiments/ablation.py", "--arms", "x"]),
             (7, ["py", "supervise_gen1.py", "g2", "0"])]
    _procs(monkeypatch, sup, procs)
    assert sup.live_scripts("supervise_gen2.py") == [1, 2, 5]
    assert sup.live_scripts("supervise_gen2.py", "g2") == [1, 2]
    assert sup.live_scripts("supervise_gen1.py", "ds2") == [4]
    assert sup.live_scripts("supervise_gen1.py") == [4, 7]
    assert sup.live_trainers() == [6]
    monkeypatch.setattr(sup, "_mine", lambda: {1})
    assert sup.other_supervisors("g2") == [2]
    monkeypatch.setattr(sup, "_mine", lambda: {1, 2})
    assert sup.other_supervisors("g2") == []


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    _procs(monkeypatch, sup, [
        (20, ["py", "benchmarks/supervisors/supervise_gen2.py", "g2", "0"])])
    monkeypatch.setattr(sup, "_mine", lambda: {99})
    for name in ("run_done", "wait_external", "preflight"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["g2", "0"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_gen2.py g2 is alive (pids [20])" in \
        out
    # the master's supervisor (supervise_gen1.py ds2, its own tree) or a
    # backup copy is no obstacle
    _procs(monkeypatch, sup, [
        (21, ["py", r"Y:\SwitchPlayer-gen1b-run\benchmarks\supervisors"
                    r"\supervise_gen1.py", "ds2", "0"]),
        (22, ["py", "supervise_gen2.py.bak", "g2", "0"])])
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["g2", "0"]) == 0
    assert "RUN COMPLETE g2 s0" in capsys.readouterr().out


def test_wait_no_trainer_waits_out_any_trainer_and_streams_rows(
        sup, monkeypatch, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "g2", 0)
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
    assert "EVAL g2 s0 iter 40" in out            # an orphan stays watched


def test_the_preflight_waits_for_no_trainer(sup, monkeypatch, capsys):
    """Audit r1 B-L1 (PB's r1 B-L2): a RESUMING relaunch beside an
    orphaned trainer or prep -- the preflight (the suites, Ray-spinning
    slow tests among them) is not called until none is alive; the orphan's
    rows stream meanwhile (primed first: an old row is not re-printed).
    Sleeps bounded: a guard that never returns fails instead of hanging."""
    _rel, base, jsonl, ck, _d = sup.paths("g2", 0)
    pathlib.Path(base).mkdir(parents=True)
    pathlib.Path(ck).write_bytes(b"x" * 1_000_001)   # a resume: no wait
    _write_prep(sup)

    def row(it):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"iteration": it, "elapsed_s": float(it),
                                "reward_ceiling": 7.5}) + NL)

    row(3)                                           # before the prime
    alive = [7]
    monkeypatch.setattr(sup, "live_trainers", lambda: list(alive))

    def effect(k):
        if k == 1:
            row(4)                                   # the orphan writes
        if k == 3:
            alive.clear()                            # the trainer exits

    sleep = _bounded(effect, n=5)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    seen = []

    def preflight(v):
        seen.append((v, list(alive)))
        raise Stop

    monkeypatch.setattr(sup, "preflight", preflight)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    assert seen == [("g2", [])]
    assert sleep.calls == [60, 60, 60]
    out = capsys.readouterr().out
    assert "RESUMING g2 s0" in out
    assert "trainer is already running (pids [7])" in out
    assert "EVAL g2 s0 iter 4" in out
    assert "EVAL g2 s0 iter 3" not in out


# --------------------------------------------------------------------------
# main: the order, the launch gate, the child
# --------------------------------------------------------------------------
def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys):
    monkeypatch.setattr(sup, "wait_external", _stop)
    ck = pathlib.Path(sup.paths("g2", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    assert sup.main(["g2", "0"]) == 0
    assert "RUN COMPLETE g2 s0" in capsys.readouterr().out


def test_a_resume_skips_the_wait(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "wait_external",
                        lambda *a: (_ for _ in ()).throw(
                            AssertionError("a resume must not wait")))
    monkeypatch.setattr(sup, "preflight", _stop)
    ck = pathlib.Path(sup.paths("g2", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    assert "RESUMING g2 s0" in capsys.readouterr().out
    ck.unlink()
    # a LAUNCHED line in its own log is a resume too
    (ck.parent / "supervisor.log").write_text(
        "LAUNCHED g2 s0 attempt 1 (fresh), pid 1" + NL)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    assert "RESUMING g2 s0" in capsys.readouterr().out
    (ck.parent / "supervisor.log").unlink()
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])


def test_launched(sup, tmp_path):
    base, ck = tmp_path / "b", tmp_path / "b" / "c.ckpt"
    base.mkdir()
    assert not sup.launched(str(base), str(ck))
    # a NO-WAIT line (the distillation step's override) is nothing here
    (base / "supervisor.log").write_text("NO-WAIT override: g2 s0 x" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("x LAUNCHED g2" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("LAUNCHED g2 s0 attempt 1" + NL)
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
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or (True, ""))
    monkeypatch.setattr(sup, "set_intact",
                        lambda: events.append("intact") or (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)


def test_g2_runs_wait_preflight_prep_then_each_launch_guarded(
        sup, monkeypatch, capsys):
    events, preps, launched = [], [], []
    _loop(sup, monkeypatch, events, preps)
    # the preflight's CPU-only env must never reach the trainer or the prep
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    ck = pathlib.Path(sup.paths("g2", 0)[3])

    def popen(args, **k):
        events.append("popen")
        launched.append((list(args), k))
        if len(launched) == 1:          # the first attempt leaves a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["g2", "0"]) == 1                 # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == (["wait", "guard", "preflight", ("prep", True)]
                      + ["guard", "manifest", "intact", "popen"] * 5)
    rel = sup.paths("g2", 0)[0]
    for i, (args, k) in enumerate(launched):
        assert args == sup.command("g2", 0) + (["--resume"] if i else [])
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


def test_a_failed_prep_launches_nothing(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup, "run_prep", lambda env, redo_ok: False)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["g2", "0"]) == 1


def test_a_broken_prep_blocks_the_launch_until_intact(sup, monkeypatch,
                                                      capsys):
    events = []
    _loop(sup, monkeypatch, events)
    checks = [(False, "set.pkl does not match vocab.json's set_sha256"),
              (False, "set.pkl does not match vocab.json's set_sha256"),
              (True, "")]
    monkeypatch.setattr(sup, "set_intact",
                        lambda *a: events.append("check") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    pre = events.index("preflight")
    assert events[pre - 1] == "guard"            # the guard before it
    gate = events[events.index("guard", pre):]
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
        sup.main(["g2", "0"])
    assert events[:4] == ["wait", "guard", "preflight", ("prep", True)]
    assert events[4:] == ["guard", "manifest", "guard", "manifest",
                          "intact", "popen"]


def test_stray_state(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "wait_external", _stop)
    rel, base, _j, ckpt, _d = sup.paths("g2", 0)
    base = pathlib.Path(base)
    off = base / f"{base.name}-offline" / "snapshots-x"
    off.mkdir(parents=True)
    (off / "weights-iter001.pth").write_bytes(b"x")
    own = base / "snapshots-x"
    own.mkdir()
    (own / "weights-iter000.pth").write_bytes(b"x")
    assert len(sup.stray_snapshots(str(base), ckpt)) == 2
    # never launched: no excuse, even for exactly row 0 and the curve
    assert sup.main(["g2", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    # G2's own death before its first checkpoint: launched, and exactly
    # row 0 + the offline curve excused
    (base / "supervisor.log").write_text("LAUNCHED g2 s0 attempt 1 (fresh)"
                                         + NL)
    assert sup.stray_snapshots(str(base), ckpt, own_ok=True) == []
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    (own / "weights-iter005.pth").write_bytes(b"x")  # a later row: stray
    assert sup.main(["g2", "0"]) == 2
    (own / "weights-iter005.pth").unlink()
    capsys.readouterr()
    # probe rows without a checkpoint: refused even when launched
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    _rows(probes / f"probe_exhaustive_eval-{base.name}-offline-acting.jsonl",
          [1])
    assert sup.main(["g2", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    assert sup.stray_probe_rows("g2", 0, ckpt) == []


def test_the_attempt_loop_refuses_stray_probe_rows(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    run = sup.run_name("g2", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"

    def rows_appear(*a):
        _rows(probes / f"probe_exhaustive_eval-{run}.jsonl", [5])

    monkeypatch.setattr(sup, "wait_no_trainer", rows_appear)
    assert sup.main(["g2", "0"]) == 2
    assert "stray snapshots / probe jsonls" in capsys.readouterr().out


def test_the_attempt_loop_refuses_stray_snapshots_of_an_unlaunched_run(
        sup, monkeypatch, capsys):
    """Snapshots appearing while the first launch waited (no LAUNCHED line
    yet): no excuse in the loop either."""
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("g2", 0)[1])

    def snap_appears(*a):
        (base / "snapshots-x").mkdir(parents=True, exist_ok=True)
        (base / "snapshots-x" / "weights-iter000.pth").write_bytes(b"x")

    monkeypatch.setattr(sup, "wait_no_trainer", snap_appears)
    assert sup.main(["g2", "0"]) == 2
    assert "weights-iter000.pth" in capsys.readouterr().out


def test_a_leftover_read10_jsonl_is_refused_like_the_exact_evals(sup,
                                                                 capsys):
    """As Test 2's rebuild (PB's audit r1 B-M1): read (10)'s
    probe_prefix_bn jsonl is one of the run's probe jsonls -- rows there
    without a checkpoint refuse a fresh start (the hand-off would never
    re-probe those snapshots on the new timeline)."""
    run = sup.run_name("g2", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    assert [os.path.basename(p) for p in sup.probe_jsonls("g2", 0)] == [
        f"probe_exhaustive_eval-{run}-acting.jsonl",
        f"probe_exhaustive_eval-{run}.jsonl",
        f"probe_exhaustive_eval-{run}-offline-acting.jsonl",
        f"probe_exhaustive_eval-{run}-offline.jsonl",
        f"probe_prefix_bn-{run}.jsonl"]
    ckpt = sup.paths("g2", 0)[3]
    mech = probes / f"probe_prefix_bn-{run}.jsonl"
    mech.parent.mkdir(parents=True, exist_ok=True)
    mech.write_text(json.dumps({"iteration": 30, "p1_ratio": 0.5}) + NL)
    assert sup.stray_probe_rows("g2", 0, ckpt) == [mech.name]
    assert sup.main(["g2", "0"]) == 2
    assert f"probe rows ({mech.name})" in capsys.readouterr().out
    pathlib.Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_probe_rows("g2", 0, ckpt) == []


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("g2", 0)[1])
    ckpt = pathlib.Path(sup.paths("g2", 0)[3])
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["g2", "0"]) == 0
    assert "ABANDONED g2 s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["g2", "0"]) == 0
    assert "RUN COMPLETE g2 s0" in capsys.readouterr().out


def test_a_resumed_g2_never_redoes_its_prep(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    ck = pathlib.Path(sup.paths("g2", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)                  # a resume
    with pytest.raises(Stop):
        sup.main(["g2", "0"])
    assert events == ["guard", "preflight", ("prep", False), "guard",
                      "manifest", "intact"]


def test_an_abandoned_g2_runs_no_prep(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    base = pathlib.Path(sup.paths("g2", 0)[1])

    def preflight(v):
        base.mkdir(parents=True, exist_ok=True)
        (base / "ABANDONED").write_text("")
        return True, ""

    monkeypatch.setattr(sup, "preflight", preflight)
    assert sup.main(["g2", "0"]) == 0
    assert not any(isinstance(e, tuple) for e in events)
    assert "no prep" in capsys.readouterr().out


def test_main_returns_when_abandoned_while_waiting(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "abandoned")
    monkeypatch.setattr(sup, "preflight", _stop)
    assert sup.main(["g2", "0"]) == 0
    assert "ABANDONED g2 s0" in capsys.readouterr().out
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "failed")
    assert sup.main(["g2", "0"]) == 1


def test_main_source_order(sup):
    src = inspect.getsource(sup.main)
    assert src.index("other_supervisors(") < src.index("run_done(") < \
        src.index("abandoned(base)") < src.index("stray_snapshots(") < \
        src.index("wait_external(")
    # the guard once before the preflight (audit r1 B-L1), after the prime
    assert src.index("rows.prime()") < src.index("wait_no_trainer(rows)") \
        < src.index("preflight(variant)")
    assert src.index("preflight(variant)") < src.index("run_prep(")
    assert src.index("run_prep(") < src.index("for attempt in range(")
    loop = src[src.index("for attempt in range("):]
    assert loop.index("wait_no_trainer(rows)") < loop.index(
        "manifest_check()") < loop.index("set_intact()") < loop.index(
        "subprocess.Popen(")
    assert src.count("wait_no_trainer(") == 2       # before the preflight
    assert "redo_ok=not launched(base, ckpt)" in src
    # one run: the prep, the integrity gate and the own-death excuse hold
    # for g2 unconditionally (no per-variant branch, no vocabulary check)
    assert "variant ==" not in src and "vocab_is_ds" not in src
    assert src.count("stray_snapshots(base, ckpt,") == 2
    assert src.count("launched(base, ckpt)") == 4


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
    assert sup.main(["g2", "0"]) == 1
    out = capsys.readouterr().out
    assert stop in out and "MAX ATTEMPTS" not in out
    assert len(launches) == n
    assert sup.STALL_MIN_S == 300 < 465


def test_a_checkpoint_resets_the_no_progress_count(sup, monkeypatch,
                                                   capsys):
    events = []
    _loop(sup, monkeypatch, events)
    clock = _clock(sup, monkeypatch)
    ck = pathlib.Path(sup.paths("g2", 0)[3])
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
    assert sup.main(["g2", "0"]) == 1
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
    rows = sup.Rows(str(jsonl), "g2", 0)

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
    assert out.startswith("EVAL g2 s0 iter 2") and "ALERT" not in out
    assert "maxp 0.4" in out and "ceiling 7.5  kept 0.9  max 60.0" in out
    assert "frames/decision 12.0" in out
    # Test 2's rebuild's live reads (audit r1 B-N3, PB's r1 B-N4)
    assert "freshH None | prefix None  prefix_step1 None (control None)" \
        in out
    row(iteration=4, reward_ceiling=7.5, fresh_target_entropy=1.25,
        prefix=0.5, prefix_step1=0.125, prefix_step1_control=0.75)
    rows.emit()
    out = capsys.readouterr().out
    assert ("freshH 1.25 | prefix 0.5  prefix_step1 0.125 (control 0.75) "
            "| ceiling 7.5") in out
    row(iteration=3, reward_ceiling=1.0, reward_max=60.1)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: g2 s0 row 3: reward_ceiling 1.0" in out
    assert "ALERT: g2 s0 row 3: reward_max 60.1 > 8 x the ceiling" in out
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
    rows = sup.Rows(str(jsonl), "g2", 0)
    jsonl.write_text(json.dumps({"iteration": 40, "reward_ceiling": 7.5,
                                 "elapsed_s": 1.0}) + NL)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: g2 s0 row 40: host commit free 3.2 GB < 4 GB" in out


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
