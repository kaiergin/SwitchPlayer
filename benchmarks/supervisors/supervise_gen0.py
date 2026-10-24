"""Test 1 of the per-generation clip (Kai, 2026-10-24): ONE run at SEED 0 --
  g0: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen0
      (TK with its reward ceiling held at 1 -- the sign clip -- for the whole
      run), 90 iterations, fresh
--ray --cpu-actor (pinned by the arm). Design and pre-registered reads:
CURRICULUM.md "Next experiment (pre-registered) -- Test 1: a generation-0
token master at a constant sign clip".

    python benchmarks/supervisors/supervise_gen0.py g0 0
    python benchmarks/supervisors/supervise_gen0.py --write-manifest

No predecessor: it starts once its preflight passes. Before EVERY launch
attempt it waits while ANY python runs ablation.py or distill.py (any tree,
arm or seed, by path or as a module: one trainer on the GPU at a time,
never a second writer; other GPU users -- a GPU probe, Kai's Switch code --
are not seen), THEN checks the manifest, then launches; rows keep
streaming while it waits (an orphaned trainer of this run stays
monitored). One supervisor per run: a second supervise_gen0.py refuses to
start. A run already complete exits at once (RUN COMPLETE), before any
preflight. An ABANDONED file in the run folder (Kai's call: stop the
supervisor and its trainer first, then create it) stops every later
launch. A run folder holding snapshots, or probe jsonls holding records,
but no checkpoint is refused (a fresh start would mix two timelines: move
the run folder and its two probe jsonls aside). Completion, ABANDONED and
the stray state are re-checked right before every launch. Each row ALERTs
on a ceiling other than 1, a stored record reward past 8 units (a 32-frame
token's 8 base steps at 1 each: the clip did not apply), a pace past 25
min/iteration (Kai's call) and host commit below 4 GB. CRASH LOOP (5
deaths within 60 min) and NO PROGRESS (3 attempts of >= 5 min in a row
that die without a new checkpoint) stop it for diagnosis.

Create the run folder before a launch that redirects into it (`>>
<folder>/supervisor.log`): the shell opens the redirect before this file
runs.

The run executes from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits; development continues on branch
curriculum-p2 elsewhere), so REPO is this file's own tree. Supervisor v2
pattern: a relaunch auto-resumes from the last checkpoint; before EVERY
launch attempt the tree's AUDITED files must hash to the manifest written
from the audited commit (gen0_manifest.json)."""
import json
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "g0": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0",
}
RUNS = {"g0": "mspacman-tokens-gen0"}
ITERATIONS, TRAIN_STEPS = 90, 2000
# Row checks (ALERTs): the constant ceiling, a record reward past what the
# ceiling allows for the longest token (8 base steps x 1), a pace past which
# the cost is Kai's call, low host commit.
CEILING = 1.0
REWARD_MAX_ALERT = 8.0
PACE_ALERT_MIN = 25.0
COMMIT_ALERT_GB = 4.0
MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
SEEDS = {0: "2026-10-24"}                 # the pre-registered seed and date
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# The shared interpreter lives in the main tree (worktrees have no .venv).
PY = os.environ.get("GEN0_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 40
# Attempts that live at least STALL_MIN_S and die without writing a new
# checkpoint this many times (quick deaths aside) stop the supervisor: a
# deterministic crash minutes into each resume never reaches CRASH LOOP's 5
# deaths in 60 min (audit r1 B-L3). STALL_MIN_S overlaps CRASH LOOP's reach
# (it catches deaths up to ~465 s under the quick-death backoffs), so no
# lifetime falls between the two (audit r2 LOW-1).
NO_PROGRESS_DEATHS = 3
STALL_MIN_S = 300
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gen0_manifest.json")
# Every repo module the run executes, the probe the readout runs, the
# suites the preflight runs, and the supervisors the suites read.
SUITES = ["tests/test_gen0.py", "tests/test_rgrow.py", "tests/test_tokens.py",
          "tests/test_distill.py",
          "tests/test_mspacman.py", "tests/test_bnrecal.py",
          "tests/test_prioract.py", "tests/test_prioract2.py",
          "tests/test_tempscale.py", "tests/test_paper_mode.py",
          "tests/test_mcts.py", "tests/test_vtree.py",
          "tests/test_vtree_communal.py", "tests/test_reanalyze_refresh.py",
          "tests/test_value_prefix.py", "tests/test_checkpoint_resume.py",
          "tests/test_harness.py"]
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "residual_block.py",
           "experiments/ablation.py", "experiments/ray_actors.py",
           "experiments/distill.py",
           "pytest.ini", "tests/test_gumbelact8.py",
           # imported by the suites
           "tests/test_reanalysis.py",
           # older supervisors the suites read or exec
           "benchmarks/supervisors/supervise_bnrecal.py",
           "benchmarks/supervisors/supervise_gumbelact8.py",
           "benchmarks/supervisors/supervise_prioract2.py",
           "benchmarks/supervisors/supervise_searchact8.py",
           "benchmarks/supervisors/supervise_tempscale.py",
           "benchmarks/supervisors/supervise_tokens.py",
           "benchmarks/supervisors/supervise_mspacman.py",
           "benchmarks/supervisors/supervise_rgrow.py",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_gen0.py"] + SUITES


def expected_ceiling(iteration):
    """The reward ceiling row `iteration` must log: 1, constant."""
    return CEILING


def paths(variant, seed):
    """(REL, BASE, JSONL, CKPT, DEATHS) of `variant` at `seed`: one folder
    per run."""
    arm = VARIANTS[variant]
    rel = f"benchmarks/results/{RUNS[variant]}-s{seed}-{SEEDS[seed]}"
    base = os.path.join(REPO, *rel.split("/"))
    return (rel, base, os.path.join(base, f"ablation{TAG}.jsonl"),
            os.path.join(base, f"resume{TAG}-{arm}-seed{seed}.ckpt"),
            os.path.join(base, "deaths.log"))


def command(variant, seed):
    """The pre-registered recipe: TK's (supervise_rgrow.py tk), the arm the
    only change."""
    return [PY, "experiments/ablation.py", "--arms", VARIANTS[variant],
            "--seed", str(seed),
            "--iterations", str(ITERATIONS), "--train-steps",
            str(TRAIN_STEPS), "--eval-every", "5",
            "--env", ENV_ID, "--frameskip", "4", "--episode-cap", "3000",
            "--episodes-per-iter", "3", "--checkpoint-every", "2",
            "--snapshot-weights"] + MODE


def file_hashes(root):
    """sha256 of every AUDITED file's text, CRLF normalized to LF (git's
    autocrlf round trip must not matter)."""
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(
                f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def manifest_check():
    """(ok, reason): this tree's AUDITED files hash to the manifest. Run
    before EVERY launch attempt: a relaunch must never run an unaudited
    edit."""
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            want = json.load(f)
        have = file_hashes(REPO)
    except (OSError, ValueError) as e:
        return False, f"manifest check: {e}"
    if not isinstance(want, dict):
        return False, "manifest check: not a JSON object"
    stale = sorted(k for k in AUDITED if want.get(k) != have.get(k))
    if stale:
        return False, "not the audited revision: " + ", ".join(stale)
    return True, ""


def write_manifest():
    """Written from a CLEAN tree (the audited commit) only: uncommitted
    edits to an AUDITED file refuse it. The commit it was written at is
    recorded under "_commit" (provenance; manifest_check reads only the
    AUDITED keys)."""
    r = subprocess.run(["git", "status", "--porcelain", "--", *AUDITED],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode or r.stdout.strip():
        print("refusing --write-manifest: AUDITED files differ from the "
              "commit (or git failed):\n" + (r.stdout or r.stderr), flush=True)
        return 1
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True)
    out = file_hashes(REPO)
    out["_commit"] = head.stdout.strip() if head.returncode == 0 else "?"
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {MANIFEST} from {REPO}", flush=True)
    return 0


def quick(variant):
    """The import probe for `variant`'s arm (TK's 36 tokens, the ceiling
    held at 1)."""
    arm = VARIANTS[variant]
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree; "
            f"spec = ablation.ARMS[{arm!r}]; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_UNIT'] == 10.0 and "
            "spec['REWARD_GROWTH_START'] == 1.0 and "
            "spec['REWARD_GROWTH_CEILING'] == 1.0; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            "assert len(spec['TOKEN_VOCAB']) == 36; "
            "assert hasattr(ablation, 'reward_growth_gate') and "
            "hasattr(ablation.Run, 'apply_reward_ceiling'); "
            "assert hasattr(muzero, 'grown_reward')")


def run_done(variant, seed):
    """The run's completion marker: its checkpoint shrunk to a marker."""
    try:
        return os.path.getsize(paths(variant, seed)[3]) < 1_000_000
    except OSError:
        return False


def lines_of(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def preflight(variant):
    """(ok, reason): the manifest, the quick import probe, then the suites
    IN this tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([PY, "-c", quick(variant)], cwd=REPO,
                       capture_output=True, text=True, env=env)
    if r.returncode:
        return False, "quick probe: " + (r.stderr.strip().splitlines()
                                         or ["?"])[-1]
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p",
                            "no:cacheprovider", *SUITES],
                           cwd=REPO, capture_output=True, text=True, env=env,
                           timeout=3 * 3600)
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    if r.returncode:
        lines = (r.stdout + r.stderr).strip().splitlines()
        bad = [l for l in lines if l.startswith(("FAILED", "ERROR"))][:3]
        # Stable across polls: no timings, no object addresses.
        clean = [re.sub(r" at 0x[0-9a-fA-F]+", "", re.sub(
            r" in [0-9.]+s( \(\d+:\d\d:\d\d\))?", "", l)) for l in lines[-2:]]
        return False, f"tests failed (rc={r.returncode}): " + " | ".join(
            [re.sub(r" at 0x[0-9a-fA-F]+", "", l) for l in bad] or clean)
    return True, ""


TRAINER_SCRIPTS = ("ablation.py", "distill.py")
TRAINER_MODULES = ("ablation", "distill", "experiments.ablation",
                   "experiments.distill")


def is_trainer(cmd):
    """A python command line that runs ablation.py or distill.py: as a
    script path from any directory (`experiments/ablation.py`, or
    `ablation.py` run from experiments/) or as a module (`-m ablation`,
    `-m experiments.distill`; audit r1 B-L4). A test file (test_distill.py)
    is not one."""
    for i, c in enumerate(cmd):
        if os.path.basename(c.replace(chr(92), "/")) in TRAINER_SCRIPTS:
            return True
        if c == "-m" and cmd[i + 1:i + 2] and cmd[i + 1] in TRAINER_MODULES:
            return True
    return False


def live_trainers():
    """PIDs of ANY python running ablation.py or distill.py (any tree,
    arm or seed; is_trainer): an orphan of an earlier supervisor (the child
    is detached and survives a supervisor kill), another experiment's run
    or prep, a duplicate launch. Python processes only: an editor or a git
    command naming the file is not a trainer. A python whose command line
    cannot be read is not counted (counting it could block every launch
    for good on a foreign process)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if name.startswith("python") and is_trainer(cmd):
            out.append(p.info["pid"])
    return out


def _mine():
    """This process and its ancestors (the venv's python.exe is a launcher
    whose child runs the script with the same command line)."""
    import psutil
    out = {os.getpid()}
    try:
        out.update(a.pid for a in psutil.Process().parents())
    except psutil.Error:
        pass
    return out


def other_supervisors():
    """PIDs of OTHER python processes running supervise_gen0.py (never this
    process or its ancestors)."""
    import psutil
    mine = _mine()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if p.info["pid"] in mine or not name.startswith("python"):
            continue
        if any(c.replace(chr(92), "/").endswith("supervise_gen0.py")
               for c in cmd):
            out.append(p.info["pid"])
    return out


def commit_free_gb():
    """Free host commit (page file + RAM), GB; None off Windows."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return m.ullAvailPageFile / 2**30
    except (AttributeError, OSError):
        return None


class Rows:
    """Stream new jsonl rows as EVAL lines (the run's in-run eval, the
    clip fields, the prior's sharpness and the token diagnostics)."""

    def __init__(self, jsonl, variant, seed):
        self.jsonl, self.variant, self.seed = jsonl, variant, seed
        self.seen, self.last = set(), None

    def lines(self):
        try:
            with open(self.jsonl, "rb") as f:
                return [l for l in f.read().decode("utf-8", "replace")
                        .splitlines() if l.strip()]
        except OSError:
            return []

    def prime(self):
        self.seen.update(l.strip() for l in self.lines())

    def emit(self):
        for line in self.lines():
            key = line.strip()
            if key in self.seen:
                continue
            self.seen.add(key)
            try:
                r = json.loads(line)
            except ValueError:
                continue
            el = r.get("elapsed_s")
            per = "n/a"
            if (isinstance(el, (int, float)) and self.last
                    and r.get("iteration", 0) > self.last[1]):
                per = "%.1f" % ((el - self.last[0]) / 60.0
                                / (r["iteration"] - self.last[1]))
            if isinstance(el, (int, float)):
                self.last = (el, r.get("iteration", 0))
            # decisions per TRAJECTORY = per life on Ms. Pac-Man
            tok = (" | frames/decision %s  decisions/life %s  play share "
                   "%s  search share %s" % (
                       r.get("tok_frames_per_decision"),
                       r.get("tok_decisions_per_game"),
                       r.get("tok_duration_play_share"),
                       r.get("tok_duration_search_share")))
            print("EVAL %s s%d iter %s  train %s  env %s  return %s  H(pi) %s "
                  " maxp %s  erank %s | ceiling %s  kept %s  max %s%s | "
                  "elapsed %s s, %s min/iter" % (
                      self.variant, self.seed, r.get("iteration"),
                      r.get("train_steps"), r.get("env_steps"),
                      r.get("return_mean"), r.get("policy_entropy"),
                      r.get("policy_maxp"), r.get("proj_erank"),
                      r.get("reward_ceiling"), r.get("reward_kept"),
                      r.get("reward_max"), tok, el, per),
                  flush=True)
            for alert in row_alerts(r, per):
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: {alert}", flush=True)
            free = commit_free_gb()
            if free is not None and free < COMMIT_ALERT_GB:
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: host commit free {free:.1f} GB "
                      f"< {COMMIT_ALERT_GB:g} GB -- memory creep?", flush=True)


def row_alerts(r, per):
    """What a row shows that needs a human: a ceiling other than the
    constant 1 (the re-clip did not run, or the arm is not this one), a pace
    past PACE_ALERT_MIN min/iteration (pre-registered: Kai's call), a stored
    record reward past REWARD_MAX_ALERT units (the clip did not apply)."""
    out = []
    it, c = r.get("iteration"), r.get("reward_ceiling")
    if isinstance(it, int) and it >= 1:
        want = expected_ceiling(it)
        if not isinstance(c, (int, float)) or abs(c - want) > 1e-3 * want:
            out.append(f"reward_ceiling {c!r}, the arm holds {want:.4f}")
    try:
        if float(per) > PACE_ALERT_MIN:
            out.append(f"{per} min/iteration > {PACE_ALERT_MIN:g} -- the "
                       f"pre-registered pace rule: Kai's call")
    except ValueError:
        pass
    m = r.get("reward_max")
    if isinstance(m, (int, float)) and m > REWARD_MAX_ALERT:
        out.append(f"reward_max {m} > {REWARD_MAX_ALERT:g} units (a 32-frame "
                   f"token at ceiling 1 sums at most 8: the clip did not "
                   f"apply)")
    return out


def snapshot(deaths_log, variant, seed, attempt, rc, lifetime):
    """Append the death-time environment to deaths.log; one line to stdout."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"=== death attempt={attempt} rc={rc} lifetime={lifetime:.0f}s "
             f"at {stamp}"]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["powershell", "-NoProfile", "-Command",
                 "$o=Get-CimInstance Win32_OperatingSystem; "
                 "'ram_free_gb: ' + [math]::Round($o.FreePhysicalMemory/1MB,1)"
                 " + ' commit_free_gb: ' + "
                 "[math]::Round($o.FreeVirtualMemory/1MB,1)"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except Exception as e:
            out = f"<{type(e).__name__}>"
        lines.append(out)
    with open(deaths_log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    gpu = lines[1].split(",")[0].strip() if len(lines) > 1 else "?"
    print(f"DEATH {variant} s{seed} rc={rc} after {lifetime:.0f}s; snapshot "
          f"-> deaths.log (gpu used {gpu})", flush=True)


def wait_no_trainer(rows=None):
    """Block while ANY python runs ablation.py or distill.py (is_trainer:
    an orphan of an earlier supervisor, another run, a duplicate): one
    trainer on the GPU at a time, and never a second writer. `rows` keep streaming meanwhile (an orphan of THIS run keeps its
    EVAL lines and ALERTs)."""
    polls, told = 0, None
    while True:
        if rows is not None:
            rows.emit()
        pids = live_trainers()
        if not pids:
            return
        msg = (f"ALERT: an ablation.py or distill.py trainer is already "
               f"running (pids {pids}) -- one trainer on the GPU at a time, "
               f"and never a second writer")
        if msg != told or polls % 60 == 0:
            print(f"{msg} [{polls} min]", flush=True)
            told = msg
        time.sleep(60)
        polls += 1


def abandoned(base):
    """Kai's call: an ABANDONED file in the run folder stops every launch."""
    return os.path.exists(os.path.join(base, "ABANDONED"))


def stray_snapshots(base, ckpt):
    """Snapshots in the run folder with no checkpoint beside them (a fresh
    start there would mix two timelines)."""
    import glob
    if os.path.exists(ckpt):
        return []
    return sorted(glob.glob(os.path.join(base, "snapshots*", "*.pth")))


def stray_probe_rows(variant, seed, ckpt):
    """This run's probe jsonls holding records while no checkpoint exists
    (probe rows only exist for settled snapshots, i.e. after a
    checkpoint)."""
    if os.path.exists(ckpt):
        return []
    run = f"{RUNS[variant]}-s{seed}-{SEEDS[seed]}"
    out = []
    for suffix in ("-acting", ""):
        path = os.path.join(REPO, "benchmarks", "probes",
                            f"probe_exhaustive_eval-{run}{suffix}.jsonl")
        if any(l.strip() for l in lines_of(path)):
            out.append(os.path.basename(path))
    return out


def main(argv):
    try:
        # cp1252 under the launcher's redirect: a non-ASCII character in a
        # PREFLIGHT line must not kill the supervisor
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    if "--write-manifest" in argv:
        return write_manifest()
    if (len(argv) != 2 or argv[0] not in VARIANTS
            or not argv[1].isdigit() or int(argv[1]) not in SEEDS):
        print(f"usage: supervise_gen0.py <variant: {sorted(VARIANTS)}> "
              f"<seed: {sorted(SEEDS)}> | --write-manifest", flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    others = other_supervisors()
    if others:
        print(f"refusing: another supervise_gen0.py is alive (pids "
              f"{others}) -- one supervisor per run", flush=True)
        return 2
    if run_done(variant, seed):
        # Before the preflight: a relaunch after a reboot must not run the
        # CPU suites for nothing.
        print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
              flush=True)
        return 0
    if abandoned(base):
        print(f"ABANDONED {variant} s{seed}: the run folder holds an "
              f"ABANDONED file -- not launching", flush=True)
        return 0
    stray = stray_snapshots(base, ckpt)
    if stray:
        print(f"refusing: {len(stray)} snapshots but no checkpoint in "
              f"{rel} -- a fresh start would mix two timelines; move the run "
              f"folder and its two probe jsonls aside", flush=True)
        return 2
    stray = stray_probe_rows(variant, seed, ckpt)
    if stray:
        print(f"refusing: probe rows ({', '.join(stray)}) but no checkpoint "
              f"-- a fresh start would keep the old timeline's scores; move "
              f"them (and the run folder) aside", flush=True)
        return 2
    if os.path.exists(ckpt):
        print(f"RESUMING {variant} s{seed} from its checkpoint", flush=True)

    os.makedirs(base, exist_ok=True)
    # Before the preflight: rows an orphan of this run writes meanwhile are
    # still printed and checked.
    rows = Rows(jsonl, variant, seed)
    rows.prime()
    polls, told = 0, None
    while True:
        ok, why = preflight(variant)
        if ok:
            break
        if why != told or polls % 6 == 0:
            print(f"PREFLIGHT not ready ({polls * 10} min): {why}", flush=True)
            told = why
        time.sleep(600)
        polls += 1
        rows.emit()          # an orphan's rows between retries
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} s{seed} ({arm}, {ITERATIONS} iterations)",
          flush=True)

    # CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
    # would silently train on the CPU.
    env = {**{k: v for k, v in os.environ.items()
              if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": rel, "PYTHONFAULTHANDLER": "1"}
    stall_s = 2 * 3600    # the child logs at least every checkpoint (2 iters)
    deaths, quick_deaths, stalls = [], 0, 0

    def ckpt_stamp():
        try:
            return os.path.getmtime(ckpt)
        except OSError:
            return None

    def completed():
        try:
            return os.path.getsize(ckpt) < 1_000_000
        except OSError:
            return False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if completed():
            print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
                  flush=True)
            return 0
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not relaunching", flush=True)
            return 0
        # The guard first, THEN the manifest (the guard can block for hours;
        # the manifest is checked right before the launch).
        polls, told = 0, None
        while True:
            wait_no_trainer(rows)
            ok, why = manifest_check()
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the tree is the "
                      f"audited revision [{polls} min]", flush=True)
                told = why
            time.sleep(60)
            polls += 1
        # The waits above can last hours: completion (an orphan finished)
        # and ABANDONED are re-checked right before the launch.
        if completed():
            print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
                  flush=True)
            return 0
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not relaunching", flush=True)
            return 0
        # A launch without a checkpoint: no old timeline may sit beside it
        # (a retraction done while this supervisor waited).
        stray = (stray_snapshots(base, ckpt)
                 + stray_probe_rows(variant, seed, ckpt))
        if stray:
            names = ", ".join(os.path.basename(x) for x in stray[:4])
            print(f"refusing: {len(stray)} stray snapshots / probe jsonls "
                  f"but no checkpoint ({names}) -- not launching; follow the "
                  f"fresh-restart procedure", flush=True)
            return 2
        args = command(variant, seed) + (["--resume"] if os.path.exists(ckpt)
                                         else [])
        log = open(os.path.join(base, f"supv2_{attempt}.log"), "ab")
        born, stamp0 = time.time(), ckpt_stamp()
        p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=0x08000000)
        print(f"LAUNCHED {variant} s{seed} attempt {attempt} "
              f"({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
              flush=True)
        alerted = 0.0
        while p.poll() is None:
            rows.emit()
            try:
                idle = time.time() - max(os.path.getmtime(log.name), born)
            except OSError:
                idle = 0.0
            if idle > stall_s and time.time() - alerted > 3600:
                print(f"ALERT: attempt {attempt} (pid {p.pid}) log silent for "
                      f"{idle / 3600:.1f} h -- possible hang", flush=True)
                alerted = time.time()
            time.sleep(30)
        log.close()
        rows.emit()
        rc = p.returncode
        lifetime = time.time() - born
        snapshot(deaths_log, variant, seed, attempt, rc, lifetime)
        if completed():
            print(f"RUN COMPLETE {variant} s{seed}", flush=True)
            return 0
        deaths.append(time.time())
        if len([t for t in deaths if time.time() - t < 3600]) >= 5:
            print("CRASH LOOP: 5 deaths within 60 min -- stopping for "
                  "diagnosis", flush=True)
            return 1
        # Deaths after >= STALL_MIN_S without a new checkpoint (the
        # shortest are CRASH LOOP's; audit r1 B-L3, r2 LOW-1).
        if ckpt_stamp() != stamp0:
            stalls = 0
        elif lifetime >= STALL_MIN_S:
            stalls += 1
        if stalls >= NO_PROGRESS_DEATHS:
            print(f"NO PROGRESS: {stalls} attempts in a row (quick deaths "
                  f"aside) died without writing a new checkpoint -- stopping "
                  f"for diagnosis", flush=True)
            return 1
        if lifetime < 600:
            quick_deaths += 1
            wait = min(900, 60 * (2 ** quick_deaths))
            print(f"quick death #{quick_deaths}; backing off {wait}s",
                  flush=True)
            time.sleep(wait)
        else:
            quick_deaths = 0
            time.sleep(15)
    print("MAX ATTEMPTS REACHED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
