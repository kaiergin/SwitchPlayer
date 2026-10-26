"""Test 2 of the per-generation clip (Kai, 2026-10-24): the first swap under
the per-generation clip -- TWO runs at SEED 0, one after the other --
  ds2: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-
       tokens-gen1-distill (the distilled generation-1 student: G0's
       re-tokenized games pre-filled, 20,000 offline steps before iteration
       1, TK's within-run schedules keyed to its online steps)
  fs2: ...-tokens-gen1-fresh (the same student vocabulary at the same
       constant ceiling, from scratch, TK's schedules)
both at the prep's constant student_ceiling (the halved-clip rule), 90
iterations each, the TK recipe, --ray --cpu-actor (pinned by the arms), and
both with PREFIX_BN_PER_STEP (CURRICULUM Test 2 "REBUILD WITH THE FIX",
2026-10-25: PB's fix joined the base; the master G0 is unchanged, an off
arm; runs dated 2026-10-25, from the run worktree SwitchPlayer-gen1b-run).
Design and pre-registered reads: CURRICULUM.md "Next experiment
(pre-registered) -- Test 2: the first swap under the per-generation clip";
the swap's machinery is the distillation step's ("P3's distillation step";
this file is derived from supervise_distill.py, which ran DS then FS).

    python benchmarks/supervisors/supervise_gen1.py ds2 0
    python benchmarks/supervisors/supervise_gen1.py fs2 0
    python benchmarks/supervisors/supervise_gen1.py --write-manifest

ds2 waits (wait_external) for G0 -- Test 1's generation-0 master, training
from ITS run worktree (G0_TREE) -- to have its completion marker AND both of
its probe jsonls to hold every row 5-90 (its evals have landed); then its
preflight (manifest, quick probe, the suites); then THE PREP, once. THE
MASTER IS FIXED: G0's final snapshot (weights-iter090; no choice, no
fallback). Before the prep, Test 1's NEAR RANDOM rule: G0's DEPLOYED mean
over rows 60-90 (its -acting jsonl) must be >= 502.08 (g = (score - 307.3)
/ 973.9 >= 0.2), else "PREP FAILED: MASTER NEAR RANDOM" and a stop (Kai's
call: no Test 2 until diagnosed). A G0 with an ABANDONED file -- while ds2
waits or before the prep -- is "PREP FAILED: MASTER ABANDONED" and a stop
(Kai's call). The pick is logged in PREP/master.json, then
experiments/distill.py records G0, derives the student vocabulary and the
student's constant ceiling (vocab.json "student_ceiling"), re-tokenizes and
searches (GPU, ~0.5 h; log PREP/prep.log; up to PREP_ATTEMPTS tries 15 min
apart; vocab.json is its completion signal). Then DS2 launches. fs2 waits
for DS2's completion marker AND both DS2 probe jsonls holding rows 0 and
5-90, both offline-curve jsonls holding rows 1-4 (and the prep's
vocab.json), then its preflight, then FS2 launches -- only on the
vocabulary AND ceiling DS2 trained with (vocab_is_ds, before every launch).
A RESUME (this run's checkpoint exists, or its supervisor.log logged a
LAUNCHED line) skips the wait. Before EVERY launch attempt (and before the
prep) it waits while ANY python runs ablation.py or distill.py (is_trainer:
any tree, by path or as a module; the prep counts). One supervisor per
run: a second supervise_gen1.py of the same variant refuses to start. A
run already complete exits at once (RUN COMPLETE). ABANDONED in the run
folder stops every launch; a run folder holding snapshots (its own or its
offline curve's) or probe rows but no checkpoint is refused (move them
aside) -- only ds2, once LAUNCHED, may restart over exactly weights-iter000
and its offline curve (a death before its first checkpoint).

Each row ALERTs on a reward_ceiling other than the prep's student_ceiling
(constant for the whole run), a stored record reward past 8 x that ceiling
(a 32-frame token is 8 base steps: the clip did not apply), a pace past 25
min/iteration (Kai's call) and host commit below 4 GB. CRASH LOOP (5 deaths
within 60 min) and NO PROGRESS (3 attempts of >= STALL_MIN_S = 5 min in a
row that die without a new checkpoint; supervise_gen0.py's rule, CURRICULUM
Test 1 AMENDMENTS r1 / r2: no death lifetime escapes both) stop it.

Kept from the distillation step's audits (CURRICULUM "P3's distillation
step" AMENDMENTS r1-r7): the prep counts as a trainer (an orphaned prep
blocks a second one); once this run was LAUNCHED the prep is never redone
nor started at all -- the launch gate ALERTs and waits instead; before that
a transient read error is retried before the files are judged damaged, and
damaged files are moved aside, not overwritten; an orphaned prep that
finished while the next attempt waited counts; stale prep tmp files are
swept; master.json is written atomically, and a finished prep without a
readable one is refused; the prep is watched (an hourly ALERT while
prep.log is silent past 30 min); ABANDONED is checked while waiting and
before the prep; fs2 checks DS2's vocabulary in a child process (no torch in
the long-lived supervisor). Removed with the fallback master: the
distillation step's wait-skipping override and everything behind it (Test
2 has no second master to fall back to).

PROCEDURES. PAUSE (a Switch session): kill the supervisor, then its trainer
(experiments/ablation.py) or prep (experiments/distill.py), by command
line; resume by rerunning launch_gen1.ps1 (idempotent). After a REBOOT
rerun launch_gen1.ps1 (and launch_gen0.ps1 until G0's evals have landed:
its hand-off, post_gen0.py, writes the rows ds2 waits for).
ABANDON (Kai's call): FIRST stop the supervisor and its trainer, THEN
create ABANDONED in the run folder (ds2 abandoned: fs2 keeps waiting for
it -- Kai's call). UN-ABANDONING: stop the hand-off (post_gen1.py) and its
probes, drop the probe rows its abandon pass logged as newer than the
checkpoint, remove ABANDONED, rerun launch_gen1.ps1. A FRESH restart: stop
the supervisor, its trainer or prep and the hand-off, move the run folder
AND its probe jsonls (ds2: and its "-offline" ones) aside -- and, to redo
the prep too, the prep folder (only before DS2 trained on it: FS2 must
train on DS2's vocabulary) -- then rerun launch_gen1.ps1.

Create a run folder before a launch that redirects into it (`>>
<folder>/supervisor.log`): the shell opens the redirect before this file
runs.

The runs execute from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits), so REPO is this file's own tree.
Supervisor v2 pattern: a relaunch auto-resumes from the last checkpoint;
before EVERY launch attempt and the prep the tree's AUDITED files must hash
to the manifest written from the audited commit (gen1_manifest.json)."""
import json
import math
import os
import re
import subprocess
import sys
import time

G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
VARIANTS = {"ds2": ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                    "mspacman-tokens-gen1-distill"),
            "fs2": ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                    "mspacman-tokens-gen1-fresh")}
RUNS = {"ds2": "mspacman-gen1-distill", "fs2": "mspacman-gen1-fresh"}
NAMES = {"ds2": "DS2", "fs2": "FS2"}
ITERATIONS = 90
TRAIN_STEPS = 2000
# Row checks (ALERTs): the prep's constant student_ceiling at every row; a
# record reward past what that ceiling allows for the longest token (a
# 32-frame token is MAX_TOKEN_STEPS base steps; REWARD_MAX_SLACK absorbs the
# row's 4-decimal rounding); a pace past which the cost is Kai's call; low
# host commit.
MAX_TOKEN_STEPS = 8
REWARD_MAX_SLACK = 1e-4
# A stored record reward past the reward support's edge (muzero.REWARD_CLIP
# units) saturates the reward / value-prefix heads (the distillation step's
# alert; audit r1 B-NIT7: 8 x c1 passes it once c1 > 50)
REWARD_SUPPORT_EDGE = 400.0
PACE_ALERT_MIN = 25.0
COMMIT_ALERT_GB = 4.0
PRED_EVALS_ALERT_S = 2 * 3600    # predecessor done this long, evals missing
PRED_SILENT_ALERT_S = 4 * 3600   # predecessor unfinished, its log silent
PREP_ATTEMPTS = 3
PREP_RETRY_S = 900
# Attempts that live at least STALL_MIN_S and die without writing a new
# checkpoint this many times in a row (quick deaths aside) stop the
# supervisor: supervise_gen0.py's rule (CURRICULUM Test 1 AMENDMENTS r1
# B-L3, r2 LOW-1: STALL_MIN_S overlaps CRASH LOOP's reach, ~465 s under the
# quick-death backoffs, so no lifetime falls between the two). DS2's
# attempts each cost the ~1 h offline phase until its first checkpoint.
NO_PROGRESS_DEATHS = 3
STALL_MIN_S = 300
# Test 1's NEAR RANDOM rule on the master: g = (score - 307.3) / 973.9 (Ms.
# Pac-Man's random and human scores) below 0.2 over G0's deployed rows
# 60-90. NEAR_RANDOM_TOL only absorbs float noise in the 7-row mean (the
# probe's 3-decimal means put a genuinely lower one >= 1.4e-4 below).
G_RANDOM, G_HUMAN = 307.3, 973.9
NEAR_RANDOM = 502.08             # = 307.3 + 0.2 x 973.9
NEAR_RANDOM_TOL = 1e-6

MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
# the pre-registered seed and the rebuild's date (the pre-fix 2026-10-24
# runs never started)
SEEDS = {0: "2026-10-25"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# The shared interpreter lives in the main tree (worktrees have no .venv).
PY = os.environ.get("GEN1_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 40
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gen1_manifest.json")
# The prep's folder (REPO-relative): set.pkl, vocab.json (written last: the
# prep's completion signal; it carries student_ceiling), master.json,
# prep.log.
PREP_REL = "benchmarks/results/gen1-prep-s0-2026-10-25"


def prep_dir():
    return os.path.join(REPO, *PREP_REL.split("/"))


# The master: G0 (live in its own worktree until it completes; its hand-off
# writes its probe jsonls there). Names as supervise_gen0.paths builds them.
G0_TREE = r"Y:\SwitchPlayer-gen0-run"
G0_RUN = "mspacman-tokens-gen0-s0-2026-10-24"
G0_BASE = os.path.join(G0_TREE, "benchmarks", "results", G0_RUN)
G0_MARKER = os.path.join(G0_BASE, f"resume{TAG}-{G0_ARM}-seed0.ckpt")
G0_SUPLOG = os.path.join(G0_BASE, "supervisor.log")
G0_ABANDONED = os.path.join(G0_BASE, "ABANDONED")
G0_PROBES = tuple(os.path.join(G0_TREE, "benchmarks", "probes",
                               f"probe_exhaustive_eval-{G0_RUN}{s}.jsonl")
                  for s in ("-acting", ""))       # deployed first, then raw
G0_ROWS = tuple(range(5, 95, 5))
G0_ITERATIONS = 90
G0_SNAPSHOT = os.path.join(G0_BASE, f"snapshots{TAG}-{G0_ARM}-seed0",
                           "weights-iter090.pth")

DS_ROWS = (0,) + tuple(range(5, 95, 5))
DS_CURVE_ROWS = (1, 2, 3, 4)
PREP_SILENT_ALERT_S = 1800
# Every suite supervise_gen0.py runs, plus this supervisor's, Test 2's and
# the fix's (PB's test_pbn.py: the rebuild).
SUITES = ["tests/test_supervise_gen1.py", "tests/test_gen1.py",
          "tests/test_pbn.py",
          "tests/test_gen0.py", "tests/test_rgrow.py",
          "tests/test_tokens.py", "tests/test_distill.py",
          "tests/test_mspacman.py", "tests/test_bnrecal.py",
          "tests/test_prioract.py", "tests/test_prioract2.py",
          "tests/test_tempscale.py", "tests/test_paper_mode.py",
          "tests/test_mcts.py", "tests/test_vtree.py",
          "tests/test_vtree_communal.py", "tests/test_reanalyze_refresh.py",
          "tests/test_value_prefix.py", "tests/test_checkpoint_resume.py",
          "tests/test_harness.py"]
# Every repo module the runs and the prep execute, the probe the readout
# runs, the suites the preflight runs, and the modules / supervisors the
# suites import or read.
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
           "benchmarks/supervisors/supervise_gen0.py",
           "benchmarks/supervisors/supervise_distill.py",
           "benchmarks/probes/probe_exhaustive_eval.py",
           # read (10)'s mechanism probe (PB's read (1)) and its fixed state
           # set (binary in git, never converted, so it hashes the same in
           # an autocrlf tree)
           "benchmarks/probes/probe_prefix_bn.py",
           "benchmarks/probes/step0_prefix_bn/states.npz",
           "benchmarks/supervisors/supervise_gen1.py"] + SUITES


def run_name(variant, seed):
    return f"{RUNS[variant]}-s{seed}-{SEEDS[seed]}"


def paths(variant, seed):
    """(REL, BASE, JSONL, CKPT, DEATHS) of `variant` at `seed`: one folder
    per run."""
    arm = VARIANTS[variant]
    rel = f"benchmarks/results/{run_name(variant, seed)}"
    base = os.path.join(REPO, *rel.split("/"))
    return (rel, base, os.path.join(base, f"ablation{TAG}.jsonl"),
            os.path.join(base, f"resume{TAG}-{arm}-seed{seed}.ckpt"),
            os.path.join(base, "deaths.log"))


def command(variant, seed):
    """The pre-registered recipe: TK's (supervise_rgrow.py tk), as the
    distillation step's (supervise_distill.py), with the student vocabulary
    and its ceiling (both: --token-vocab's vocab.json) and the distillation
    set (ds2)."""
    vocab = PREP_REL + "/vocab.json"
    extra = ["--token-vocab", vocab]
    if variant == "ds2":
        extra += ["--distill-set", PREP_REL + "/set.pkl"]
    return [PY, "experiments/ablation.py", "--arms", VARIANTS[variant],
            "--seed", str(seed),
            "--iterations", str(ITERATIONS), "--train-steps",
            str(TRAIN_STEPS), "--eval-every", "5",
            "--env", ENV_ID, "--frameskip", "4", "--episode-cap", "3000",
            "--episodes-per-iter", "3", "--checkpoint-every", "2",
            "--snapshot-weights"] + MODE + extra


def file_hashes(root):
    """sha256 of every AUDITED file's text, CRLF normalized to LF (git's
    autocrlf round trip must not matter; the one binary, states.npz, is
    never converted, so the same bytes normalize the same)."""
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(
                f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def manifest_check():
    """(ok, reason): this tree's AUDITED files hash to the manifest. Run
    before EVERY launch attempt and the prep: a relaunch must never run an
    unaudited edit."""
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
    """The import probe for `variant`'s arm (a generation-1 student of G0
    whose ceiling comes from the prep) and the prep module."""
    arm = VARIANTS[variant]
    kind = "distill" if variant == "ds2" else "fresh"
    offset = 20000 if variant == "ds2" else 0
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree, distill; "
            f"spec = ablation.ARMS[{arm!r}]; "
            f"assert spec['_student'] == {kind!r}; "
            "assert spec['_ceiling_from_prep'] is True; "
            f"assert spec['_master_arm'] == {G0_ARM!r}; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_UNIT'] == 10.0; "
            f"assert spec.get('SCHEDULE_OFFSET_STEPS', 0) == {offset}; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            "assert len(spec['TOKEN_VOCAB']) == 36; "
            "assert hasattr(ablation, 'student_gate') and "
            "hasattr(ablation, 'student_arm') and "
            "hasattr(ablation, 'student_ceiling') and "
            "hasattr(ablation, 'distill_offline') and "
            "hasattr(muzero, 'schedule_steps'); "
            "assert hasattr(distill, 'prep') and "
            "hasattr(distill, 'halved_ceiling') and "
            "distill.DROP_SHARE == 0.01")


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


def probe_rows(path):
    """{iteration: exact_mean} of a probe jsonl's full-range Ms. Pac-Man
    records (dicts without "max_noop" whose env is ENV_ID)."""
    out = {}
    for line in lines_of(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (isinstance(r, dict) and "max_noop" not in r
                and r.get("env") == ENV_ID
                and isinstance(r.get("iteration"), int)):
            out[r["iteration"]] = r.get("exact_mean")
    return out


def marker_missing(marker):
    try:
        if os.path.getsize(marker) >= 1_000_000:
            return ["completion marker (checkpoint still full size)"]
    except OSError:
        return ["completion marker (no checkpoint)"]
    return []


def predecessor(variant, seed):
    """What `variant` waits for: (name, marker, [(probe jsonl, rows)],
    supervisor.log, supervisor script, its variant arg, its ABANDONED
    file)."""
    if variant == "ds2":
        return ("G0", G0_MARKER, [(p, G0_ROWS) for p in G0_PROBES],
                G0_SUPLOG, "supervise_gen0.py", "g0", G0_ABANDONED)
    _, base, _, ck, _ = paths("ds2", seed)
    run = run_name("ds2", seed)
    # DS2's rows 0 and 5-90 and its offline curve's rows 1-4 (FS2 must not
    # start while DS2's final pass still probes).
    probes = [(os.path.join(REPO, "benchmarks", "probes",
                            f"probe_exhaustive_eval-{name}{s}.jsonl"), rows)
              for name, rows in ((run, DS_ROWS),
                                 (run + "-offline", DS_CURVE_ROWS))
              for s in ("-acting", "")]
    return ("DS2", ck, probes, os.path.join(base, "supervisor.log"),
            "supervise_gen1.py", "ds2", os.path.join(base, "ABANDONED"))


def pred_missing(variant, seed):
    """What the predecessor still lacks before `variant` may start: [] once
    its marker is there and its probe jsonls hold every row (fs2: and the
    prep's vocab.json)."""
    name, marker, probes, _, _, _, _ = predecessor(variant, seed)
    out = marker_missing(marker)
    for path, rows in probes:
        got = probe_rows(path)
        lack = [i for i in rows if i not in got]
        if lack:
            out.append(f"{os.path.basename(path)} rows {lack[:8]}"
                       + (" ..." if len(lack) > 8 else ""))
    if variant == "fs2" and not prep_done():
        out.append("the prep's vocab.json")
    return out


def pred_marker_age(variant, seed):
    marker = predecessor(variant, seed)[1]
    try:
        if os.path.getsize(marker) < 1_000_000:
            return time.time() - os.path.getmtime(marker)
    except OSError:
        pass
    return None


def pred_unattended(variant, seed):
    """None while the predecessor is finished or driven; else why not."""
    name, marker, _, suplog, script, pvar, flag = predecessor(variant, seed)
    if not marker_missing(marker):
        return None
    if os.path.exists(flag):
        # (rerunning its launcher would not help)
        return (f"{name} is ABANDONED (its run folder's flag) -- no "
                f"verdict needing it runs; Kai's call")
    alive = live_scripts(script, pvar)
    if not alive:
        stop = pred_master_stop(variant, seed)
        if stop:
            # (audit r1 B-NIT2: a rerun repeats the preflight and the stop)
            return (f"{name}'s supervisor stopped on '{stop}' -- Kai's call "
                    f"(rerunning its launcher repeats the stop)")
        return f"no live {script} {pvar} -- rerun its launcher"
    try:
        quiet = time.time() - os.path.getmtime(suplog)
    except OSError:
        return None
    if quiet > PRED_SILENT_ALERT_S:
        return (f"{name}'s supervisor (pids {alive}) is alive but its log is "
                f"silent for {quiet / 3600:.1f} h -- check it; do NOT rerun "
                f"its launcher beside it")
    return None


def pred_master_stop(variant, seed):
    """The predecessor's supervisor's last give-up line when it is a
    master-gate stop ("PREP FAILED: MASTER ..."; after its last launch or
    preflight line), else None."""
    lines = lines_of(predecessor(variant, seed)[3])
    fail = max((i for i, l in enumerate(lines) if l.startswith(
        ("CRASH LOOP", "MAX ATTEMPTS", "PREP FAILED", "NO PROGRESS"))),
               default=-1)
    live = max((i for i, l in enumerate(lines) if l.startswith(
        ("LAUNCHED", "preflight ok", "PREFLIGHT"))), default=-1)
    if fail > live and lines[fail].startswith("PREP FAILED: MASTER"):
        return lines[fail].strip()[:200]
    return None


def pred_gave_up(variant, seed):
    """The predecessor's supervisor logged CRASH LOOP / MAX ATTEMPTS / PREP
    FAILED / NO PROGRESS after its last launch or preflight line."""
    lines = lines_of(predecessor(variant, seed)[3])
    fail = max((i for i, l in enumerate(lines) if l.startswith(
        ("CRASH LOOP", "MAX ATTEMPTS", "PREP FAILED", "NO PROGRESS"))),
               default=-1)
    live = max((i for i, l in enumerate(lines) if l.startswith(
        ("LAUNCHED", "preflight ok", "PREFLIGHT"))), default=-1)
    return fail > live


MASTER_ABANDONED = ("MASTER ABANDONED: G0's run folder holds an ABANDONED "
                    "file -- no Test 2 on it and no fallback master (Kai's "
                    "call)")


def wait_external(variant, seed):
    """Block until the predecessor has completed AND its evals have landed
    ("ready"), this run's ABANDONED file appears ("abandoned"), or -- ds2 --
    G0's ABANDONED file appears ("failed": PREP FAILED: MASTER ABANDONED
    is printed); hourly WAITING lines and ALERTs."""
    name = predecessor(variant, seed)[0]
    base = paths(variant, seed)[1]
    first, polls = True, 0
    while True:
        if abandoned(base):
            return "abandoned"
        if variant == "ds2" and os.path.exists(G0_ABANDONED):
            print(f"PREP FAILED: {MASTER_ABANDONED}", flush=True)
            return "failed"
        missing = pred_missing(variant, seed)
        if not missing:
            if not first:
                print(f"{name} complete and evaluated; {variant} s{seed} may "
                      f"start", flush=True)
            return "ready"
        if first or polls % 60 == 0:
            print(f"WAITING for {name} before {variant} s{seed} "
                  f"[{polls // 60} h]: missing " + "; ".join(missing)[:400],
                  flush=True)
            if pred_gave_up(variant, seed):
                print(f"ALERT: {name}'s supervisor gave up -- {variant} keeps "
                      f"waiting", flush=True)
            why = pred_unattended(variant, seed)
            if why:
                print(f"ALERT: {name} is unfinished and nothing drives it: "
                      f"{why}", flush=True)
            age = pred_marker_age(variant, seed)
            if age is not None and age > PRED_EVALS_ALERT_S:
                print(f"ALERT: {name} completed {age / 3600:.1f} h ago but "
                      f"its evals still lack rows -- is its hand-off alive? "
                      f"(a hand-off log line '... rows missing' means rerun "
                      f"them)", flush=True)
            if variant == "ds2":
                gone = master_files_missing()
                if gone:
                    print(f"ALERT: master snapshot missing: "
                          f"{'; '.join(gone)} -- keep G0's run tree",
                          flush=True)
        first = False
        time.sleep(60)
        polls += 1


def window_mean(rows, lo, hi):
    """The mean of rows lo..hi (step 5) when every one is there, else None."""
    want = range(lo, hi + 1, 5)
    if any(not isinstance(rows.get(i), (int, float)) for i in want):
        return None
    return sum(rows[i] for i in want) / len(want)


def master_files_missing():
    """G0's final snapshot, once G0 has its completion marker, when it does
    not exist."""
    if marker_missing(G0_MARKER) or os.path.exists(G0_SNAPSHOT):
        return []
    return [G0_SNAPSHOT]


def choose_master():
    """The pre-registered master, FIXED: G0's final snapshot
    (weights-iter090) -- after Test 1's NEAR RANDOM rule on its deployed
    rows 60-90. Raises SystemExit (the prep stops: PREP FAILED, Kai's call)
    when G0 is ABANDONED ("MASTER ABANDONED"), has no completion marker,
    lacks a deployed row 60-90, is NEAR RANDOM ("MASTER NEAR RANDOM": mean
    < NEAR_RANDOM) or its snapshot is missing; never falls back. Returns a
    dict (which, snapshot, iterations, g0_mean_60_90)."""
    if os.path.exists(G0_ABANDONED):
        raise SystemExit(MASTER_ABANDONED)
    if marker_missing(G0_MARKER):
        raise SystemExit("G0 has no completion marker -- the master is its "
                         "FINAL snapshot")
    mean = window_mean(probe_rows(G0_PROBES[0]), 60, 90)
    if mean is None:
        raise SystemExit(f"G0's deployed rows 60-90 are incomplete in "
                         f"{G0_PROBES[0]} -- land them (its hand-off) first")
    g = (mean - G_RANDOM) / G_HUMAN
    if mean < NEAR_RANDOM - NEAR_RANDOM_TOL:
        raise SystemExit(f"MASTER NEAR RANDOM: G0's deployed mean over rows "
                         f"60-90 is {mean:.2f} < {NEAR_RANDOM} (g {g:.3f} < "
                         f"0.2) -- diagnosed before any Test 2 (Kai's call)")
    if not os.path.exists(G0_SNAPSHOT):
        raise SystemExit(f"G0's final snapshot is missing: {G0_SNAPSHOT}")
    return {"which": "G0", "snapshot": G0_SNAPSHOT,
            "iterations": G0_ITERATIONS, "g0_mean_60_90": mean}


def prep_done():
    return os.path.exists(os.path.join(prep_dir(), "vocab.json"))


def read_pick(path):
    """master.json's pick, or None when it is unreadable / incomplete."""
    try:
        with open(path, encoding="utf-8") as f:
            pick = json.load(f)
    except (OSError, ValueError):
        return None
    keys = ("which", "snapshot", "iterations", "g0_mean_60_90")
    if not isinstance(pick, dict) or any(k not in pick for k in keys):
        return None
    return pick


def write_pick(path, pick):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pick, f, indent=1)
    os.replace(tmp, path)


def valid_ceiling(c):
    """A usable student_ceiling: a finite positive number (not a bool)."""
    return (not isinstance(c, bool) and isinstance(c, (int, float))
            and math.isfinite(c) and c > 0)


def prep_ceiling():
    """The prep's student_ceiling (vocab.json; the halved-clip rule), or
    None while vocab.json is missing, unreadable or holds no valid one."""
    try:
        with open(os.path.join(prep_dir(), "vocab.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    c = doc.get("student_ceiling") if isinstance(doc, dict) else None
    return float(c) if valid_ceiling(c) else None


def set_intact():
    """(ok, reason): vocab.json's set_sha256 is set.pkl's, and vocab.json is
    a prep of G0's final snapshot with a valid student_ceiling."""
    try:
        with open(os.path.join(prep_dir(), "vocab.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
        import hashlib
        h = hashlib.sha256()
        with open(os.path.join(prep_dir(), "set.pkl"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    except (OSError, ValueError) as e:
        return False, f"prep files: {e}"
    if not isinstance(doc, dict):
        return False, "vocab.json is not a JSON object"
    if doc.get("set_sha256") != h.hexdigest():
        return False, "set.pkl does not match vocab.json's set_sha256"
    if (doc.get("master_arm") != G0_ARM
            or doc.get("master_iterations") != G0_ITERATIONS):
        return False, (f"vocab.json's master is {doc.get('master_arm')!r} "
                       f"after {doc.get('master_iterations')!r} iterations, "
                       f"not G0's final snapshot")
    if not valid_ceiling(doc.get("student_ceiling")):
        return False, (f"vocab.json has no valid student_ceiling "
                       f"({doc.get('student_ceiling')!r})")
    return True, ""


def prep_log_tail(plog):
    """The prep log's last non-empty line (an exit's reason: distill.py's
    SystemExit text or a traceback's last line; audit r1 B-L1), at most
    200 characters, or ''."""
    lines = [l.strip() for l in lines_of(plog) if l.strip()]
    return lines[-1][:200] if lines else ""


def run_prep(env, redo_ok=True):
    """The prep, once (vocab.json is its completion signal): the master gate
    and its pick in master.json, then experiments/distill.py up to
    PREP_ATTEMPTS times. Returns True when the prep is done. A done prep
    that does not check out intact is retried for a few minutes (a
    transient lock), then -- only when `redo_ok` (this run was never
    LAUNCHED: nothing is bound to the set yet) -- moved aside and redone;
    once launched it is NEVER redone, nor started at all (a launched DS2's
    checkpoint is bound to the set's sha256 and FS2 to its vocabulary; the
    attempt loop's integrity gate ALERTs and waits). Before a prep starts,
    choose_master's gate: G0 ABANDONED or NEAR RANDOM is PREP FAILED."""
    if not redo_ok and not prep_done():
        # (nothing missing is ever re-recorded once DS2 ran)
        print("ALERT: the prep's vocab.json is missing after this run was "
              "launched -- never redone now; the launch waits for it",
              flush=True)
        return True
    if prep_done():
        pick_bad = False
        for k in range(3):
            ok, why = set_intact()
            # (a finished prep whose master.json was moved aside: refused;
            # read in the same retry loop, so a transient lock is not fatal)
            pick_bad = ok and redo_ok and read_pick(os.path.join(
                prep_dir(), "master.json")) is None
            if ok and not pick_bad:
                return True
            if k < 2:
                time.sleep(60)
        if pick_bad:
            print("PREP FAILED: a finished prep without a readable "
                  "master.json -- move the prep folder aside (the "
                  "fresh-restart procedure)", flush=True)
            return False
        if not redo_ok:
            print(f"ALERT: PREP not intact ({why}) after this run started -- "
                  f"never redone now; the launch waits for it", flush=True)
            return True
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("vocab.json", "set.pkl"):
            src = os.path.join(prep_dir(), name)
            if os.path.exists(src):
                os.replace(src, f"{src}.damaged-{stamp}")
        print(f"PREP damaged ({why}) -- moved aside (.damaged-{stamp}), "
              f"redoing it", flush=True)
    os.makedirs(prep_dir(), exist_ok=True)
    pick_path = os.path.join(prep_dir(), "master.json")
    # The master is fixed: the gate runs (and master.json is rewritten,
    # atomically) before any prep is started.
    try:
        pick = choose_master()
    except SystemExit as e:
        print(f"PREP FAILED: {e}", flush=True)
        return False
    write_pick(pick_path, pick)
    print(f"MASTER G0: {pick['snapshot']} (deployed 60-90 "
          f"{pick['g0_mean_60_90']:.2f}, g "
          f"{(pick['g0_mean_60_90'] - G_RANDOM) / G_HUMAN:.3f})",
          flush=True)
    for attempt in range(1, PREP_ATTEMPTS + 1):
        wait_no_trainer()
        # An orphaned prep may have finished while the guard waited: it
        # counts. With no prep alive, its tmp files are stale.
        if prep_done() and set_intact()[0]:
            print(f"PREP done (by an earlier prep): {PREP_REL}", flush=True)
            return True
        for name in os.listdir(prep_dir()):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(prep_dir(), name))
                except OSError:
                    pass
        ok, why = manifest_check()
        if not ok:
            print(f"PREP FAILED: {why}", flush=True)
            return False
        plog = os.path.join(prep_dir(), "prep.log")
        with open(plog, "ab") as log:
            p = subprocess.Popen(
                [PY, "experiments/distill.py", "--master", pick["snapshot"],
                 "--master-iterations", str(pick["iterations"]),
                 "--out", PREP_REL, "--seed", "0"],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=0x08000000)
            born, told = time.time(), 0.0
            while p.poll() is None:
                try:
                    quiet = time.time() - max(os.path.getmtime(plog), born)
                except OSError:
                    quiet = 0.0
                if (quiet > PREP_SILENT_ALERT_S
                        and time.time() - told > 3600):
                    print(f"ALERT: the prep (pid {p.pid}) has been silent for "
                          f"{quiet / 60:.0f} min -- hung?", flush=True)
                    told = time.time()
                time.sleep(30)
        if p.returncode == 0 and prep_done():
            ok, why = set_intact()
            if ok:
                print(f"PREP done (attempt {attempt}): {PREP_REL}", flush=True)
                return True
            print(f"PREP attempt {attempt}: {why}", flush=True)
        else:
            tail = prep_log_tail(plog)
            print(f"PREP attempt {attempt} failed (rc={p.returncode}; see "
                  f"{PREP_REL}/prep.log)"
                  + (f"; its last line: {tail}" if tail else ""), flush=True)
        if attempt < PREP_ATTEMPTS:
            time.sleep(PREP_RETRY_S)
    print(f"PREP FAILED after {PREP_ATTEMPTS} attempts -- see "
          f"{PREP_REL}/prep.log", flush=True)
    return False


VOCAB_CHECK = (
    "import json, sys, torch; "
    "doc = json.load(open(sys.argv[1], encoding='utf-8')); "
    "want = doc['vocab']; "
    "m = torch.load(sys.argv[2], map_location='cpu', weights_only=False); "
    "m = m if isinstance(m, dict) else {}; "
    "have = m.get('token_vocab'); "
    "ok = have is not None and "
    "[list(t) for t in have] == [list(t) for t in want]; "
    "c = m.get('reward_ceiling'); "
    "print('VOCAB MISMATCH' if not ok else 'VOCAB OK' "
    "if c is not None and c == doc.get('student_ceiling') "
    "else 'CEILING MISMATCH')")


def vocab_is_ds(seed):
    """(ok, reason): vocab.json's vocabulary AND student_ceiling are the
    ones DS2 trained with (the token_vocab and reward_ceiling its completion
    marker's recipe holds) -- checked in a child process, so the long-lived
    fs2 supervisor never holds torch's ~1.3 GB of commit."""
    try:
        r = subprocess.run(
            [PY, "-c", VOCAB_CHECK, os.path.join(prep_dir(), "vocab.json"),
             paths("ds2", seed)[3]], cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"vocab check: {type(e).__name__}: {e}"
    out = r.stdout.strip().splitlines()
    if r.returncode or not out:
        return False, ("vocab check: rc=%s %s" % (r.returncode, (
            r.stderr.strip().splitlines() or ["?"])[-1]))
    if out[-1] == "CEILING MISMATCH":
        return False, ("vocab.json's student_ceiling is not the ceiling DS2 "
                       "trained with (DS2's completion marker)")
    if out[-1] != "VOCAB OK":
        return False, ("vocab.json is not the vocabulary DS2 trained with "
                       "(DS2's completion marker)")
    return True, ""


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


def _procs():
    """(pid, command line) of every python process; a command line that
    cannot be read is empty."""
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = list(p.info["cmdline"] or [])
        except (psutil.Error, KeyError, TypeError):
            continue
        if name.startswith("python"):
            yield p.info["pid"], cmd


def _names(cmd):
    return [os.path.basename(c.replace(chr(92), "/")) for c in cmd]


TRAINER_SCRIPTS = ("ablation.py", "distill.py")
TRAINER_MODULES = ("ablation", "distill", "experiments.ablation",
                   "experiments.distill")


def is_trainer(cmd):
    """A python command line that runs ablation.py or distill.py: as a
    script path from any directory (`experiments/ablation.py`, or
    `ablation.py` run from experiments/) or as a module (`-m ablation`,
    `-m experiments.distill`; supervise_gen0.py's, CURRICULUM Test 1 audit
    r1 B-L4). A test file (test_distill.py) is not one."""
    for i, c in enumerate(cmd):
        if os.path.basename(c.replace(chr(92), "/")) in TRAINER_SCRIPTS:
            return True
        if c == "-m" and cmd[i + 1:i + 2] and cmd[i + 1] in TRAINER_MODULES:
            return True
    return False


def live_trainers():
    """PIDs of ANY python running ablation.py or distill.py (any tree, arm
    or seed; is_trainer): an orphan of an earlier supervisor (the child is
    detached and survives a supervisor kill), the prep (an orphaned one
    keeps the GPU and writes the prep's files), another experiment's run, a
    duplicate launch. Python processes only: an editor or a git command
    naming the file is not a trainer. A python whose command line cannot be
    read is not counted (counting it could block every launch for good on a
    foreign process)."""
    return [pid for pid, cmd in _procs() if is_trainer(cmd)]


def live_scripts(script, variant=None):
    """PIDs of python processes running `script` (exact basename), with
    `variant` as its first argument when given."""
    out = []
    for pid, cmd in _procs():
        names = _names(cmd)
        if script not in names:
            continue
        if variant is not None:
            i = names.index(script)
            if cmd[i + 1:i + 2] != [variant]:
                continue
        out.append(pid)
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


def other_supervisors(variant):
    """PIDs of OTHER python processes running supervise_gen1.py `variant`
    (never this process or its ancestors)."""
    mine = _mine()
    return [p for p in live_scripts("supervise_gen1.py", variant)
            if p not in mine]


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
    clip fields, the prior's sharpness and the token diagnostics), checked
    against the prep's student_ceiling (read once it is there)."""

    def __init__(self, jsonl, variant, seed):
        self.jsonl, self.variant, self.seed = jsonl, variant, seed
        self.seen, self.last = set(), None
        self.ceiling = None

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
            if self.ceiling is None:
                # constant for the whole run: vocab.json never changes once
                # a trainer ran on it
                self.ceiling = prep_ceiling()
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
            for alert in row_alerts(r, per, self.ceiling):
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: {alert}", flush=True)
            free = commit_free_gb()
            if free is not None and free < COMMIT_ALERT_GB:
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: host commit free {free:.1f} GB "
                      f"< {COMMIT_ALERT_GB:g} GB -- memory creep?", flush=True)


def row_alerts(r, per, ceiling):
    """What a row shows that needs a human: a reward_ceiling other than the
    prep's constant student_ceiling `ceiling` (the arm did not apply it, or
    the arm is not this one; None: the prep's vocab.json holds none to
    check against), a pace past PACE_ALERT_MIN min/iteration
    (pre-registered: Kai's call), a stored record reward past
    MAX_TOKEN_STEPS x the ceiling (the clip did not apply) or past the
    reward support's edge (REWARD_SUPPORT_EDGE: saturated)."""
    out = []
    it, c = r.get("iteration"), r.get("reward_ceiling")
    if isinstance(it, int) and it >= 1:
        if ceiling is None:
            out.append("the prep's vocab.json holds no valid student_ceiling "
                       "-- reward_ceiling unchecked")
        elif (not isinstance(c, (int, float))
              or abs(c - ceiling) > 1e-3 * ceiling):
            out.append(f"reward_ceiling {c!r}, the prep's student_ceiling is "
                       f"{ceiling:.4f}")
    try:
        if float(per) > PACE_ALERT_MIN:
            out.append(f"{per} min/iteration > {PACE_ALERT_MIN:g} -- the "
                       f"pre-registered pace rule: Kai's call")
    except ValueError:
        pass
    m = r.get("reward_max")
    if (ceiling is not None and isinstance(m, (int, float))
            and m > MAX_TOKEN_STEPS * ceiling + REWARD_MAX_SLACK):
        out.append(f"reward_max {m} > {MAX_TOKEN_STEPS} x the ceiling "
                   f"{ceiling:.4f} (a 32-frame token sums at most "
                   f"{MAX_TOKEN_STEPS} base steps at the ceiling: the clip "
                   f"did not apply)")
    if isinstance(m, (int, float)) and m > REWARD_SUPPORT_EDGE:
        out.append(f"reward_max {m} > {REWARD_SUPPORT_EDGE:g}, the reward "
                   f"support's edge (the reward heads saturate)")
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
    an orphan of an earlier supervisor, the prep, another run, a
    duplicate): one trainer on the GPU at a time, and never a second
    writer. `rows` keep streaming meanwhile (an orphan of THIS run keeps
    its EVAL lines and ALERTs)."""
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


def launched(base, ckpt):
    """This run's trainer was launched before: its checkpoint exists, or its
    own supervisor.log logged a LAUNCHED line."""
    if os.path.exists(ckpt):
        return True
    return any(l.startswith("LAUNCHED")
               for l in lines_of(os.path.join(base, "supervisor.log")))


def stray_snapshots(base, ckpt, own_ok=False):
    """Snapshots in the run folder -- its own or its offline curve's -- with
    no checkpoint beside them (a fresh start there would mix two
    timelines). `own_ok` (ds2, launched: its own death before its first
    checkpoint) excuses exactly what that leaves -- weights-iter000 and the
    offline curve, which the restart rewrites."""
    import glob
    if os.path.exists(ckpt):
        return []
    found = sorted(glob.glob(os.path.join(base, "snapshots*", "*.pth"))
                   + glob.glob(os.path.join(base, "*-offline", "snapshots*",
                                            "*.pth")))
    if own_ok:
        found = [f for f in found
                 if os.path.basename(f) != "weights-iter000.pth"
                 and not os.path.basename(os.path.dirname(os.path.dirname(
                     f))).endswith("-offline")]
    return found


def probe_jsonls(variant, seed):
    """The jsonls the eval hand-off appends this run's records to: the
    exact evals' two (acting, search) for the run and its offline curve,
    and read (10)'s (probe_prefix_bn: the run's rows only; the rebuild,
    as PB's audit r1 B-M1)."""
    run = run_name(variant, seed)
    names = [f"probe_exhaustive_eval-{name}{suffix}.jsonl"
             for name in (run, run + "-offline") for suffix in ("-acting", "")]
    return [os.path.join(REPO, "benchmarks", "probes", name)
            for name in names + [f"probe_prefix_bn-{run}.jsonl"]]


def stray_probe_rows(variant, seed, ckpt):
    """This run's probe jsonls (its offline curve's and read (10)'s too)
    holding records while no checkpoint exists."""
    if os.path.exists(ckpt):
        return []
    out = []
    for path in probe_jsonls(variant, seed):
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
        print(f"usage: supervise_gen1.py <variant: {sorted(VARIANTS)}> "
              f"<seed: {sorted(SEEDS)}> | --write-manifest", flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    others = other_supervisors(variant)
    if others:
        print(f"refusing: another supervise_gen1.py {variant} is alive "
              f"(pids {others}) -- one supervisor per run", flush=True)
        return 2
    if run_done(variant, seed):
        # Before the wait and the preflight: a relaunch after a reboot must
        # not run the CPU suites for nothing.
        print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
              flush=True)
        return 0
    if abandoned(base):
        print(f"ABANDONED {variant} s{seed}: the run folder holds an "
              f"ABANDONED file -- not launching", flush=True)
        return 0
    stray = stray_snapshots(base, ckpt,
                            variant == "ds2" and launched(base, ckpt))
    if stray:
        # (A death of DS2 before its first checkpoint leaves its row 0 and
        # offline curve behind: its own log's LAUNCHED line sanctions the
        # restart, which rewrites exactly those.)
        print(f"refusing: {len(stray)} snapshots but no checkpoint in "
              f"{rel} -- a fresh start would mix two timelines; move the run "
              f"folder and its probe jsonls aside", flush=True)
        return 2
    stray = stray_probe_rows(variant, seed, ckpt)
    if stray:
        print(f"refusing: probe rows ({', '.join(stray)}) but no checkpoint "
              f"-- a fresh start would keep the old timeline's scores; move "
              f"them (and the run folder) aside", flush=True)
        return 2
    if launched(base, ckpt):
        print(f"RESUMING {variant} s{seed}: it started before (checkpoint or "
              f"launch logged) -- no wait", flush=True)
    else:
        got = wait_external(variant, seed)
        if got == "abandoned":
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not launching", flush=True)
            return 0
        if got != "ready":           # PREP FAILED: MASTER ABANDONED (ds2)
            return 1

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
    if variant == "ds2":
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- no prep, not launching", flush=True)
            return 0
        if not run_prep({k: v for k, v in env.items()
                         if k != "ABLATION_RESULTS"},
                        redo_ok=not launched(base, ckpt)):
            return 1
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
        # The guard first, THEN the manifest and the prep's integrity (the
        # guard can block for hours; they are checked right before the
        # launch).
        polls, told = 0, None
        while True:
            wait_no_trainer(rows)
            ok, why = manifest_check()
            if ok and variant == "ds2":
                ok, why = set_intact()
            if ok and variant == "fs2":
                ok, why = vocab_is_ds(seed)
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the tree is the "
                      f"audited revision and the prep intact [{polls} min]",
                      flush=True)
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
        stray = (stray_snapshots(base, ckpt, variant == "ds2"
                                 and launched(base, ckpt))
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
                             stderr=subprocess.STDOUT,
                             creationflags=0x08000000)
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
        # shortest are CRASH LOOP's; supervise_gen0.py's rule).
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
