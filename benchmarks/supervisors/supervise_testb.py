"""Test B (Kai, 2026-10-27): a fast lineage from scratch -- FIVE runs at
SEED 0, one after the other --
  f0: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-
      tokens-fast-f0 (from scratch: TK's recipe at the sign clip, DISCOUNT
      0.97 per base step, PREFIX_BN_PER_STEP; its own play logged at
      raw-step resolution, OWN_PLAY_LOG)
  f1-f4: ...-tokens-fast-f1 .. -f4 (the distilled student of the previous
      generation's FINAL snapshot, prepped from that generation's OWN-PLAY
      LOG -- no recording: the prep's vocabulary, constant ceiling and
      discount; 20,000 offline steps before iteration 1; its own play
      logged in turn)
each ending at the iteration in which its own env steps pass its
--stop-env-steps (its final iteration K, ~25-35, run-dependent: the trainer
then writes its final eval row, its weights snapshot weights-iterKKK, its
weights file and its completion marker as usual). --iterations 60 is only a
CAP (the arms' planned length is _nominal_iterations 30); the within-run
schedules follow the env-step progress; the TK recipe, --ray --cpu-actor
(pinned by the arms). Design and pre-registered reads: CURRICULUM.md "Next
experiment (pre-registered) -- Test B: a fast lineage from scratch"; this
file is derived from supervise_gen2.py (one distilled student after a fixed
master), chained as supervise_gen1.py chains ds2 and fs2 (one supervisor
per variant, `predecessor()`, `wait_external`).

    python benchmarks/supervisors/supervise_testb.py f0 0
    ...
    python benchmarks/supervisors/supervise_testb.py f4 0
    python benchmarks/supervisors/supervise_testb.py --write-manifest

TEST B2 (Kai, 2026-10-29; CURRICULUM.md "Next experiment (pre-registered)
-- Test B2: two swaps in 100k env steps at the original constant
discount"): the same chain over THREE generations t0-t2 (...-tokens-two-t0
.. -t2; ~33,333 env steps each, --iterations 100 around a planned 50; the
token arms' constant DISCOUNT 0.997^4 per base step, which the prep keeps
for each student: _discount_rule "keep"), from its own run worktree:

    python benchmarks/supervisors/supervise_testb.py t0 0   (.. t2 0)

The variant's letter picks its lineage (LINEAGES, configure()); every rule
below holds for both. "F{k}" below is generation k of either lineage (T{k}
in Test B2); Test B2's launcher (launch_testb2.ps1) waits for Test B's F4
and its hand-off's "all done" instead of Generation 2, and its hand-off is
post_testb2.py. For Test B2, read the text below's "five" / "f0-f4" /
"F4" / launch_testb.ps1 / post_testb.py as "three" / "t0-t2" / "T2" /
launch_testb2.ps1 / post_testb2.py.

THE STOP (an amendment to the pre-registration, so that overshoots do not
accumulate): generation k's --stop-env-steps is target() --
round(TOTAL_ENV_STEPS x (k + 1) / GENERATIONS), Test B's 20,000 x (k + 1)
-- minus the cumulative env steps of F0..F(k-1) at their final rows (a
generation's final env steps: the env_steps of the last row of its own
ablation jsonl, which the trainer writes at its final iteration); F0's is
target(f0) = 20,000 (Test B2's t0: 33,333).
It is computed after the wait and before the preflight, on EVERY start (a
resume included: the trainer's resume recipe holds it, and the
predecessors' final rows never change once they are complete). A stop <= 0
(the lineage spent its budget) is "PREP FAILED: STOP NOT POSITIVE" and a
stop (Kai's call); a predecessor without a completion marker or a final row
is "PREP FAILED: STOP UNKNOWN" (after two retries a minute apart).

THE CHAIN. f0 has no predecessor (no wait, no prep). Generation 2 finishes
first: launch_testb.ps1 refuses to start before G2's completion marker (or
its ABANDONED file) exists (audit r1 B-M1); the any-trainer guard before
the preflight -- every run waits while ANY trainer runs -- is the backstop
for a manual start. f{k}
(k >= 1) waits (wait_external) for F{k-1} -- in this same run tree -- to
have its completion marker AND both of its exact-eval probe jsonls to hold
every row of F{k-1}'s own ablation jsonl (every iteration with an in-run
row: multiples of 5 and its final iteration K) plus, F{k-1} a student, its
row 0 -- and its offline curve's rows 1-4 (probe run "<run>-offline") to
have landed; then its preflight (manifest, quick probe, the suites); then
THE PREP, once. THE MASTER IS FIXED: F{k-1}'s final snapshot
(weights-iterKKK; no choice, no fallback). Before the prep, the NEAR RANDOM
rule: the mean of F{k-1}'s LAST TWO DEPLOYED rows (its -acting jsonl, its
two latest in-run iterations) must be >= 502.08 (g = (score - 307.3) /
973.9 >= 0.2), else "PREP FAILED: MASTER NEAR RANDOM" and a stop (Kai's
call). An F{k-1} with an ABANDONED file -- while f{k} waits or before the
prep -- is "PREP FAILED: MASTER ABANDONED" and a stop (Kai's call). The
pick is logged in PREP/master.json, then experiments/distill.py --from-log
reads F{k-1}'s own-play log (iterations 1..K: no env step is played),
derives the student vocabulary, ceiling and discount (vocab.json
"student_ceiling", "student_discount"), re-tokenizes and searches (GPU;
log PREP/prep.log; up to PREP_ATTEMPTS tries 15 min apart; vocab.json is
its completion signal). Then F{k} launches. A RESUME (this run's checkpoint
exists, or its supervisor.log logged a LAUNCHED line) skips the wait.
Before EVERY launch attempt (and before the preflight and the prep) it
waits while ANY python runs ablation.py or distill.py (is_trainer: any
tree, by path or as a module; the prep counts). One supervisor per run: a
second supervise_testb.py of the same variant refuses to start (another
variant's is no obstacle: all five run side by side, each waiting for its
predecessor). A run already complete exits at once (RUN COMPLETE).
ABANDONED in the run folder stops every launch; a run folder holding
snapshots (its own or its offline curve's) or probe rows but no checkpoint
is refused (move them aside) -- only a student, once LAUNCHED, may restart
over exactly weights-iter000 and its offline curve (a death before its
first checkpoint). The own-play log needs no such rule: the trainer
truncates each iteration's file at its first write, and the prep reads
iterations 1..K only.

Each row ALERTs on a reward_ceiling other than this run's constant ceiling
(F0: the sign clip 1; a student: its prep's student_ceiling), a stored
record reward past 8 x that ceiling (a 32-frame token is 8 base steps: the
clip did not apply), a pace past 25 min/iteration (Kai's call) and host
commit below 4 GB. CRASH LOOP (5 deaths within 60 min) and NO PROGRESS (3
attempts of >= STALL_MIN_S = 5 min in a row that die without a new
checkpoint) stop it.

Kept from supervise_gen2.py and the distillation step's audits (CURRICULUM
"P3's distillation step" AMENDMENTS r1-r7, Test 2's r1-r3, Generation 2's
r1-r2): the prep counts as a trainer; once a student was LAUNCHED its prep
is never redone nor started at all -- the launch gate ALERTs and waits
instead; before that a transient read error is retried before the files
are judged damaged, and damaged files are moved aside, not overwritten; an
orphaned prep that finished while the next attempt waited counts; stale
prep tmp files are swept; master.json is written atomically, and a finished
prep without a readable one is refused; the prep is watched (an hourly
ALERT while prep.log is silent past 30 min); ABANDONED is checked while
waiting and before the prep; the guard runs before the preflight too.

PROCEDURES (launcher and hand-off: launch_testb.ps1, post_testb.py). PAUSE
(a Switch session): kill the supervisors (all five, by command line), then
the trainer (experiments/ablation.py), prep (experiments/distill.py) or
preflight (a python running -m pytest in the run tree: the guard does not
count it, so a quick resume would run the suites twice), by command line;
resume by rerunning launch_testb.ps1 (idempotent: it starts only the
supervisors not running). After a REBOOT rerun launch_testb.ps1.
ABANDON (Kai's call): FIRST stop the run's supervisor and its trainer or
prep (by command line: either outlives a supervisor kill), THEN create
ABANDONED in the run folder (the next generation's supervisor then stops on
PREP FAILED: MASTER ABANDONED). UN-ABANDONING: stop the hand-off
(post_testb.py) and its probes, drop the probe rows its abandon pass logged
as newer than the checkpoint from every probe jsonl of the run (read (5)'s
probe_prefix_bn-<run>.jsonl included), remove ABANDONED, rerun
launch_testb.ps1. A FRESH restart of F{k}: stop its supervisor, its
trainer or prep and the hand-off, move the run folder AND its probe jsonls
(read (5)'s and the "-offline" ones too) aside -- and, to redo a student's
prep too, its prep folder (only before it launched: its checkpoint is bound
to the set's sha256) -- then rerun launch_testb.ps1. Only before F{k+1}'s
prep: F{k+1}'s master is F{k}'s final snapshot and its own-play log, and
every later stop counts F{k}'s final env steps -- after that, a fresh F{k}
means fresh F{k+1}..F4 too (their run folders, probe jsonls and prep
folders aside, their supervisors stopped first).

Create a run folder before a launch that redirects into it (`>>
<folder>/supervisor.log`): the shell opens the redirect before this file
runs.

The runs execute from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits), so REPO is this file's own tree.
Supervisor v2 pattern: a relaunch auto-resumes from the last checkpoint;
before EVERY launch attempt and the prep the tree's AUDITED files must hash
to the manifest written from the audited commit (testb_manifest.json)."""
import json
import math
import os
import re
import subprocess
import sys
import time

FAST_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-fast")
TWO_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-two")
# The lineages, keyed by a variant's letter: Test B's "f" (F0-F4, ~20,000
# env steps each, the discount ladder) and Test B2's "t" (Kai 2026-10-29:
# T0-T2, ~33,333 each, the token arms' constant 0.997^4 per base step).
# "iterations" is the --iterations CAP (each run ends at its env-step stop,
# near the arms' planned length "nominal" = _nominal_iterations);
# "discounts" the pre-registered discount per base step of each generation
# under the prep's "rule" (distill.discount_rule: Test B's gap 1 - g halved
# at each swap, capped at 0.997; Test B2's kept); "date" the runs'.
LINEAGES = {
    "f": {"arm": FAST_ARM, "run": "mspacman-fast", "prep": "fast-prep",
          "generations": 5, "iterations": 60, "nominal": 30,
          "discounts": (0.97, 0.985, 0.9925, 0.99625, 0.997),
          "rule": "halve", "date": "2026-10-28"},
    "t": {"arm": TWO_ARM, "run": "mspacman-two", "prep": "two-prep",
          "generations": 3, "iterations": 100, "nominal": 50,
          "discounts": (0.997 ** 4,) * 3,
          "rule": "keep", "date": "2026-10-29"},
}
# Every lineage's budget: generation k's stop brings its cumulative env
# steps to round(TOTAL_ENV_STEPS x (k + 1) / GENERATIONS) (target()).
TOTAL_ENV_STEPS = 100000


def configure(letter):
    """Bind lineage `letter`'s names and numbers (LINEAGES) module-wide:
    the module loads as Test B's ("f"); main() binds its variant's."""
    global LINEAGE, GENERATIONS, VARIANTS, RUNS, NAMES, ITERATIONS
    global NOMINAL_ITERATIONS, DISCOUNTS, DISCOUNT_RULE, SEEDS, PREP_PREFIX
    spec = LINEAGES[letter]
    LINEAGE, GENERATIONS = letter, spec["generations"]
    names = [f"{letter}{k}" for k in range(GENERATIONS)]
    VARIANTS = {v: f"{spec['arm']}-{v}" for v in names}
    RUNS = {v: f"{spec['run']}-{v}" for v in names}
    NAMES = {v: v.upper() for v in names}
    ITERATIONS, NOMINAL_ITERATIONS = spec["iterations"], spec["nominal"]
    DISCOUNTS = dict(zip(names, spec["discounts"]))
    DISCOUNT_RULE = spec["rule"]
    # the pre-registered seed and the runs' date
    SEEDS = {0: spec["date"]}
    PREP_PREFIX = spec["prep"]


configure("f")
TRAIN_STEPS = 2000
# F0's constant ceiling: the sign clip (REWARD_GROWTH_START = CEILING = 1);
# a student's is its prep's student_ceiling.
F0_CEILING = 1.0
# DISCOUNT_TOL absorbs the halving rule's float rounding.
DISCOUNT_TOL = 1e-9
DISCOUNT_CAP = 0.997
# A distilled student's offline steps before its iteration 1 (and so its
# final snapshot's when it is the next one's master).
OFFLINE_STEPS = 20000
# A student's offline curve: the snapshots after 4k / 8k / 12k / 16k
# offline steps, rows 1-4 of the probe run "<run>-offline".
CURVE_ROWS = (1, 2, 3, 4)
# Row checks (ALERTs): this run's constant ceiling at every row; a record
# reward past what that ceiling allows for the longest token (a 32-frame
# token is MAX_TOKEN_STEPS base steps; REWARD_MAX_SLACK absorbs the row's
# 4-decimal rounding); a pace past which the cost is Kai's call; low host
# commit.
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
QUICK_TIMEOUT_S = 900    # the preflight's quick import probe (PB r1 B-N5)
PREP_ATTEMPTS = 3
PREP_RETRY_S = 900
# Attempts that live at least STALL_MIN_S and die without writing a new
# checkpoint this many times in a row (quick deaths aside) stop the
# supervisor: supervise_gen0.py's rule (CURRICULUM Test 1 AMENDMENTS r1
# B-L3, r2 LOW-1: STALL_MIN_S overlaps CRASH LOOP's reach, ~465 s under the
# quick-death backoffs, so no lifetime falls between the two). A student's
# attempts each cost the ~1.5 h offline phase until its first checkpoint.
NO_PROGRESS_DEATHS = 3
STALL_MIN_S = 300
# Test 1's NEAR RANDOM rule on the master: g = (score - 307.3) / 973.9 (Ms.
# Pac-Man's random and human scores) below 0.2 over its last two deployed
# rows. NEAR_RANDOM_TOL only absorbs float noise in the 2-row mean (the
# probe's 3-decimal means put a genuinely lower one >= 5e-4 below).
G_RANDOM, G_HUMAN = 307.3, 973.9
NEAR_RANDOM = 502.08             # = 307.3 + 0.2 x 973.9
NEAR_RANDOM_TOL = 1e-6

MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# The shared interpreter lives in the main tree (worktrees have no .venv).
PY = os.environ.get("TESTB_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 40
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testb_manifest.json")
PREP_SILENT_ALERT_S = 1800
# A predecessor's supervisor stopped on one of these is Kai's call (a rerun
# of its launcher repeats the stop): the master gate's stops, a spent
# budget's (not STOP UNKNOWN: a rerun clears that; audit r1 B-L4) and a
# deterministic check of the prep's (audit r1 B-L1).
KAIS_CALL_STOPS = ("PREP FAILED: MASTER", "PREP FAILED: STOP NOT POSITIVE",
                   "PREP FAILED: PREP CHECK")
# distill.py's deterministic refusals (the last line of a failed prep's
# log): a retry repeats them, so the prep stops at once, Kai's call -- the
# value check (A2), the halved ceiling past the reward supports, an
# incomplete or empty own-play log.
PREP_CHECKS = ("the value support (", "past the reward supports",
               "own-play log ")
# Every suite supervise_gen2.py ran, plus this supervisor's and Test B's.
SUITES = ["tests/test_supervise_testb.py", "tests/test_testb.py",
          "tests/test_supervise_gen2.py", "tests/test_gen2.py",
          "tests/test_supervise_gen1.py", "tests/test_gen1.py",
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
# Every repo module the runs and the preps execute, the probes the readout
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
           # read (5)'s mechanism probe (PB's read (1)) and its fixed state
           # set (binary in git, never converted, so it hashes the same in
           # an autocrlf tree)
           "benchmarks/probes/probe_prefix_bn.py",
           "benchmarks/probes/step0_prefix_bn/states.npz",
           "benchmarks/supervisors/supervise_gen1.py",
           "benchmarks/supervisors/supervise_gen2.py",
           "benchmarks/supervisors/supervise_testb.py"] + SUITES


def gen(variant):
    """The generation k of `variant` ("f<k>"; Test B2's "t<k>")."""
    return int(variant[1:])


def master_of(variant):
    """The variant whose final snapshot is `variant`'s master (F{k-1}'s);
    None for f0 / t0 (from scratch)."""
    k = gen(variant)
    return f"{variant[0]}{k - 1}" if k else None


def is_student(variant):
    """f1-f4: the distilled student of the previous generation."""
    return master_of(variant) is not None


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


def prep_rel(variant):
    """A student's prep folder (REPO-relative): set.pkl, vocab.json
    (written last: the prep's completion signal; it carries student_ceiling
    and student_discount), master.json, prep.log. f0 has none."""
    return f"benchmarks/results/{PREP_PREFIX}-{variant}-s0-{SEEDS[0]}"


def prep_dir(variant):
    return os.path.join(REPO, *prep_rel(variant).split("/"))


def own_play_rel(variant, seed):
    """The run's own-play log folder, REPO-relative (ablation.own_play_dir
    under ABLATION_RESULTS = the run folder): iter-NNN.pkl per iteration."""
    return (f"{paths(variant, seed)[0]}/own-play{TAG}-{VARIANTS[variant]}"
            f"-seed{seed}")


def snapshot_path(variant, seed, it):
    """The run's weights snapshot at row `it` (ablation.save_weight_snapshot
    under the run folder)."""
    return os.path.join(paths(variant, seed)[1],
                        f"snapshots{TAG}-{VARIANTS[variant]}-seed{seed}",
                        f"weights-iter{it:03d}.pth")


def probe_path(probe_run, suffix):
    """An exact-eval probe jsonl in this tree (probe_exhaustive_eval files
    its records under the snapshot's grandparent folder: the probe run)."""
    return os.path.join(REPO, "benchmarks", "probes",
                        f"probe_exhaustive_eval-{probe_run}{suffix}.jsonl")


def target(variant):
    """The lineage's cumulative env steps after `variant`'s generation k:
    round(TOTAL_ENV_STEPS x (k + 1) / GENERATIONS) -- Test B's 20,000 x
    (k + 1), Test B2's 33,333 / 66,667 / 100,000. SystemExit for a variant
    of another lineage than the bound one (audit r2b NIT-3)."""
    if variant[:1] != LINEAGE:
        raise SystemExit(f"{variant} is not of the bound lineage "
                         f"{LINEAGE!r}")
    return round(TOTAL_ENV_STEPS * (gen(variant) + 1) / GENERATIONS)


def stop_env_steps(variant, seed):
    """Generation k's --stop-env-steps: target() minus F0..F(k-1)'s final
    env steps (final_row), so one generation's overshoot past its stop
    shortens the next instead of accumulating; F0's is target(f0).
    SystemExit "STOP UNKNOWN" while a predecessor is not complete, "STOP NOT
    POSITIVE" when the lineage has spent its budget (Kai's call)."""
    k = gen(variant)
    earlier = [f"{variant[0]}{j}" for j in range(k)]
    spent = []
    for v in earlier:
        try:
            spent.append(final_row(v, seed)[1])
        except SystemExit as e:
            raise SystemExit(f"STOP UNKNOWN: {e}")
    target_steps = target(variant)
    stop = target_steps - sum(spent)
    if stop <= 0:
        names = ", ".join(NAMES[v] for v in earlier)
        raise SystemExit(f"STOP NOT POSITIVE: {NAMES[variant]}'s "
                         f"--stop-env-steps would be {stop} = "
                         f"{target_steps} - {sum(spent)} (the final env "
                         f"steps of {names}: "
                         f"{spent}) -- the lineage spent its budget (Kai's "
                         f"call)")
    return stop


def command(variant, seed, stop=None):
    """The pre-registered recipe: TK's (supervise_gen2.py g2's) at the
    --iterations CAP, with this generation's env-step stop (`stop`; None:
    stop_env_steps) -- and, for a student, its prep's vocabulary, ceiling
    and discount (all three: --token-vocab's vocab.json) and its
    distillation set."""
    if stop is None:
        stop = stop_env_steps(variant, seed)
    extra = ["--stop-env-steps", str(stop)]
    if is_student(variant):
        extra += ["--token-vocab", prep_rel(variant) + "/vocab.json",
                  "--distill-set", prep_rel(variant) + "/set.pkl"]
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


def quick(variant, seed):
    """The import probe for `variant`'s arm (Test B's: PREFIX_BN_PER_STEP,
    the own-play log, the env-step stop at the lineage's planned length;
    every flag's module default off) and the prep module. f0: from scratch
    at the lineage's first DISCOUNT and the sign clip. A student: the
    distilled student of F{k-1}'s arm, at its prep's ceiling and discount
    (the lineage's rule applied to the master's) -- ending with the
    prep's own master checks on F{k-1}'s REAL final snapshot
    (distill.master_checks: its spec against this tree's arm, its final
    step after its offline steps; Generation 2's audit r1 A-L1 / B-L2: an
    edit to the master's arm fails here, not as three failed preps).
    SystemExit while F{k-1} has no final row (master_final): the preflight
    reports it."""
    arm = VARIANTS[variant]
    code = ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree, distill; "
            f"spec = ablation.ARMS[{arm!r}]; "
            "assert spec['PREFIX_BN_PER_STEP'] is True; "
            "assert spec['OWN_PLAY_LOG'] is True; "
            "assert spec['_env_stop'] is True; "
            f"assert spec['_nominal_iterations'] == {NOMINAL_ITERATIONS}; "
            "assert muzero.PREFIX_BN_PER_STEP is False; "
            "assert muzero.SCHEDULE_ENV_TARGET is None; "
            "assert muzero.OWN_PLAY_LOG is False; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_UNIT'] == 10.0; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            "assert len(spec['TOKEN_VOCAB']) == 36; "
            "assert hasattr(ablation, 'OwnPlayLog') and "
            "hasattr(ablation, 'read_own_play') and "
            "hasattr(ablation, 'env_stop_gate') and "
            "hasattr(ablation, 'env_schedule') and "
            "hasattr(muzero, 'schedule_steps'); "
            "assert hasattr(distill, 'log_lives') and "
            "hasattr(distill, 'student_discount_rule') and "
            "hasattr(distill, 'master_checks'); "
            f"assert distill.discount_rule(spec) == {DISCOUNT_RULE!r}; ")
    m = master_of(variant)
    if m is None:
        return code + (f"assert spec['DISCOUNT'] == {DISCOUNTS[variant]}; "
                       "assert '_student' not in spec; "
                       "assert spec['REWARD_GROWTH_START'] == "
                       f"spec['REWARD_GROWTH_CEILING'] == {F0_CEILING}")
    k, snap = master_final(variant, seed)
    kind = "distill" if is_student(m) else None
    return code + (
        f"master = ablation.ARMS[{VARIANTS[m]!r}]; "
        "assert master['PREFIX_BN_PER_STEP'] is True; "
        "assert master['OWN_PLAY_LOG'] is True; "
        "assert master['_env_stop'] is True; "
        f"assert master.get('_student') == {kind!r}; "
        "assert spec['_student'] == 'distill'; "
        "assert spec['_ceiling_from_prep'] is True; "
        "assert spec['_discount_from_prep'] is True; "
        f"assert spec['_master_arm'] == {VARIANTS[m]!r}; "
        "assert spec['_master_iterations'] is None; "
        f"assert spec['_distill_offline_steps'] == {OFFLINE_STEPS}; "
        f"assert spec['SCHEDULE_OFFSET_STEPS'] == {OFFLINE_STEPS}; "
        f"assert distill.master_offline_steps(master) == "
        f"{master_offline(variant)}; "
        f"assert ablation.STUDENT_DISCOUNT_CAP == {DISCOUNT_CAP}; "
        f"assert distill.discount_rule(master) == {DISCOUNT_RULE!r}; "
        f"assert abs(distill.swap_discount({DISCOUNTS[m]}, "
        f"{DISCOUNT_RULE!r}) - {DISCOUNTS[variant]}) <= {DISCOUNT_TOL}; "
        "assert hasattr(ablation, 'student_gate') and "
        "hasattr(ablation, 'student_arm') and "
        "hasattr(ablation, 'student_discount') and "
        "hasattr(ablation, 'student_ceiling') and "
        "hasattr(ablation, 'distill_offline'); "
        "assert hasattr(distill, 'prep') and "
        "hasattr(distill, 'master_spec') and "
        "hasattr(distill, 'halved_ceiling') and "
        "distill.DROP_SHARE == 0.01; "
        f"distill.master_checks({snap!r}, {k})")


def master_offline(variant):
    """The offline steps before F{k-1}'s iteration 1: 0 for F0 (from
    scratch), OFFLINE_STEPS for a student (distill.master_offline_steps)."""
    return OFFLINE_STEPS if is_student(master_of(variant)) else 0


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


def run_rows(variant, seed):
    """{iteration: env_steps} of the run's IN-RUN rows: its own ablation
    jsonl's records of its arm and seed with an iteration >= 1 (rows at
    multiples of 5 and at its final iteration; a student's row 0 and
    offline curve are snapshots, not rows)."""
    out = {}
    for line in lines_of(paths(variant, seed)[2]):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (not isinstance(r, dict) or r.get("arm") != VARIANTS[variant]
                or r.get("seed") != seed):
            continue
        it = r.get("iteration")
        if isinstance(it, int) and not isinstance(it, bool) and it >= 1:
            out[it] = r.get("env_steps")
    return out


def marker_missing(marker):
    try:
        if os.path.getsize(marker) >= 1_000_000:
            return ["completion marker (checkpoint still full size)"]
    except OSError:
        return ["completion marker (no checkpoint)"]
    return []


def final_row(variant, seed):
    """(K, env steps) of a COMPLETE generation: its final iteration -- its
    last in-run row, which the trainer writes at the iteration that passed
    its stop -- and the env steps that row records. SystemExit while the
    run has no completion marker or no such row."""
    name = NAMES[variant]
    if marker_missing(paths(variant, seed)[3]):
        raise SystemExit(f"{name} has no completion marker -- its final row "
                         f"is not settled")
    rows = run_rows(variant, seed)
    if not rows:
        raise SystemExit(f"{name}'s ablation jsonl holds no in-run row")
    k = max(rows)
    env = rows[k]
    if isinstance(env, bool) or not isinstance(env, int) or env <= 0:
        raise SystemExit(f"{name}'s final row {k} records no env steps "
                         f"({env!r})")
    return k, env


def master_final(variant, seed):
    """(K, the path of its final snapshot) of `variant`'s master F{k-1}:
    the iteration of its final row and its weights-iterKKK. SystemExit
    while F{k-1} is not complete (final_row)."""
    m = master_of(variant)
    k = final_row(m, seed)[0]
    return k, snapshot_path(m, seed, k)


def predecessor(variant, seed):
    """What `variant` waits for: (name, marker, [(probe jsonl, rows)],
    supervisor.log, supervisor script, its variant arg, its ABANDONED
    file) -- F{k-1} in this tree (supervise_testb.py f{k-1}); None for f0.
    Its rows are DYNAMIC: every in-run row its own ablation jsonl holds
    (complete once its marker is there), after row 0 when it is a student
    -- and then its offline curve's rows 1-4 too. (Read (5) is not waited
    for: the hand-off's last probe_prefix_bn may still run beside the next
    preflight.)"""
    m = master_of(variant)
    if m is None:
        return None
    _, base, _, ck, _ = paths(m, seed)
    run = run_name(m, seed)
    rows = tuple(sorted(run_rows(m, seed)))
    if is_student(m):
        rows = (0,) + rows
    probes = [(probe_path(run, s), rows) for s in ("-acting", "")]
    if is_student(m):
        probes += [(probe_path(run + "-offline", s), CURVE_ROWS)
                   for s in ("-acting", "")]
    return (NAMES[m], ck, probes, os.path.join(base, "supervisor.log"),
            "supervise_testb.py", m, os.path.join(base, "ABANDONED"))


def pred_missing(variant, seed):
    """What the predecessor still lacks before `variant` may start: [] once
    its marker is there, its ablation jsonl holds in-run rows and its probe
    jsonls hold every row."""
    name, marker, probes, _, _, _, _ = predecessor(variant, seed)
    out = marker_missing(marker)
    if not run_rows(master_of(variant), seed):
        out.append("in-run rows (its ablation jsonl holds none)")
    for path, rows in probes:
        got = probe_rows(path)
        lack = [i for i in rows if i not in got]
        if lack:
            out.append(f"{os.path.basename(path)} rows {lack[:8]}"
                       + (" ..." if len(lack) > 8 else ""))
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
    """The predecessor's supervisor's last give-up line when it is Kai's
    call (KAIS_CALL_STOPS: its master gate's stop, a spent budget's or a
    deterministic prep check's -- "PREP FAILED: MASTER ..." / "PREP FAILED:
    STOP NOT POSITIVE ..." / "PREP FAILED: PREP CHECK ..."; after its last
    launch or preflight line) -- else None."""
    lines = lines_of(predecessor(variant, seed)[3])
    fail = max((i for i, l in enumerate(lines) if l.startswith(
        ("CRASH LOOP", "MAX ATTEMPTS", "PREP FAILED", "NO PROGRESS"))),
               default=-1)
    live = max((i for i, l in enumerate(lines) if l.startswith(
        ("LAUNCHED", "preflight ok", "PREFLIGHT"))), default=-1)
    if fail > live and lines[fail].startswith(KAIS_CALL_STOPS):
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


def master_abandoned_msg(variant):
    """The stop when F{k-1}'s run folder holds an ABANDONED file."""
    return (f"MASTER ABANDONED: {NAMES[master_of(variant)]}'s run folder "
            f"holds an ABANDONED file -- no {NAMES[variant]} on it and no "
            f"fallback master (Kai's call)")


def wait_external(variant, seed):
    """Block until the predecessor F{k-1} has completed AND its evals have
    landed ("ready"; at once for f0, which has none), this run's ABANDONED
    file appears ("abandoned"), or F{k-1}'s ABANDONED file appears
    ("failed": PREP FAILED: MASTER ABANDONED is printed); hourly WAITING
    lines and ALERTs."""
    if predecessor(variant, seed) is None:
        return "ready"
    name, _, _, _, _, _, flag = predecessor(variant, seed)
    base = paths(variant, seed)[1]
    first, polls = True, 0
    while True:
        if abandoned(base):
            return "abandoned"
        if os.path.exists(flag):
            print(f"PREP FAILED: {master_abandoned_msg(variant)}", flush=True)
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
            gone = master_files_missing(variant, seed)
            if gone:
                print(f"ALERT: master files missing: {'; '.join(gone)} -- "
                      f"keep {name}'s run folder", flush=True)
        first = False
        time.sleep(60)
        polls += 1


def master_files_missing(variant, seed):
    """F{k-1}'s final snapshot and its own-play log's iteration files
    1..K, once F{k-1} is complete (its marker and its final row), where
    they do not exist: the prep reads both."""
    m = master_of(variant)
    try:
        k, snap = master_final(variant, seed)
    except SystemExit:
        return []
    out = [] if os.path.exists(snap) else [snap]
    log = os.path.join(REPO, *own_play_rel(m, seed).split("/"))
    lack = [i for i in range(1, k + 1)
            if not os.path.exists(os.path.join(log, f"iter-{i:03d}.pkl"))]
    if lack:
        out.append(f"{log} iterations {lack[:8]}"
                   + (" ..." if len(lack) > 8 else ""))
    return out


def choose_master(variant, seed):
    """The pre-registered master, FIXED: F{k-1}'s final snapshot
    (weights-iterKKK, K its final iteration) -- after Test 1's NEAR RANDOM
    rule on its LAST TWO deployed rows (its two latest in-run iterations).
    Raises SystemExit (the prep stops: PREP FAILED, Kai's call) when F{k-1}
    is ABANDONED ("MASTER ABANDONED"), has no completion marker or final
    row, lacks one of those deployed rows, is NEAR RANDOM ("MASTER NEAR
    RANDOM": mean < NEAR_RANDOM), or its snapshot or own-play log is
    missing; never falls back. Returns a dict (which, snapshot, iterations,
    master_mean_last2)."""
    m = master_of(variant)
    name = NAMES[m]
    if os.path.exists(os.path.join(paths(m, seed)[1], "ABANDONED")):
        raise SystemExit(master_abandoned_msg(variant))
    k, snap = master_final(variant, seed)
    last2 = sorted(run_rows(m, seed))[-2:]
    acting = probe_path(run_name(m, seed), "-acting")
    rows = probe_rows(acting)
    if len(last2) < 2 or any(isinstance(rows.get(i), bool) or not isinstance(
            rows.get(i), (int, float)) for i in last2):
        raise SystemExit(f"{name}'s last two deployed rows {last2} are "
                         f"incomplete in {acting} -- land them (its hand-off) "
                         f"first")
    mean = sum(rows[i] for i in last2) / 2
    g = (mean - G_RANDOM) / G_HUMAN
    if mean < NEAR_RANDOM - NEAR_RANDOM_TOL:
        raise SystemExit(f"MASTER NEAR RANDOM: {name}'s deployed mean over "
                         f"its last two rows {last2} is {mean:.2f} < "
                         f"{NEAR_RANDOM} (g {g:.3f} < 0.2) -- the chain stops "
                         f"before {NAMES[variant]} (Kai's call)")
    gone = master_files_missing(variant, seed)
    if gone:
        raise SystemExit(f"{name}'s master files are missing: "
                         f"{'; '.join(gone)}")
    return {"which": name, "snapshot": snap, "iterations": k,
            "master_mean_last2": mean}


def prep_done(variant):
    return os.path.exists(os.path.join(prep_dir(variant), "vocab.json"))


def read_pick(path):
    """master.json's pick, or None when it is unreadable / incomplete."""
    try:
        with open(path, encoding="utf-8") as f:
            pick = json.load(f)
    except (OSError, ValueError):
        return None
    keys = ("which", "snapshot", "iterations", "master_mean_last2")
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


def valid_discount(d):
    """A usable student_discount: a number in (0, DISCOUNT_CAP] (not a
    bool; ablation.student_discount's rule)."""
    return (not isinstance(d, bool) and isinstance(d, (int, float))
            and math.isfinite(d) and 0 < d <= DISCOUNT_CAP)


def same_int(value, want):
    """`value` is the int `want` (a bool or a float equal to it is not)."""
    return type(value) is int and value == want


def prep_ceiling(variant):
    """The prep's student_ceiling (vocab.json; the halved-clip rule), or
    None while vocab.json is missing, unreadable or holds no valid one."""
    try:
        with open(os.path.join(prep_dir(variant), "vocab.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    c = doc.get("student_ceiling") if isinstance(doc, dict) else None
    return float(c) if valid_ceiling(c) else None


def run_ceiling(variant):
    """The run's constant reward ceiling: F0's sign clip (F0_CEILING; it has
    no prep), a student's prep's student_ceiling (None while its vocab.json
    holds no valid one)."""
    return prep_ceiling(variant) if is_student(variant) else F0_CEILING


def set_intact(variant, seed):
    """(ok, reason): vocab.json's set_sha256 is set.pkl's, and vocab.json is
    a prep of F{k-1}'s final snapshot (its arm, its final iteration K, its
    offline steps, that snapshot's path) from F{k-1}'s own-play log (source
    and folder), with a valid student_ceiling and the pre-registered
    student_discount (DISCOUNTS) under the lineage's rule (DISCOUNT_RULE;
    a vocab.json without one is Test B's "halve")."""
    try:
        with open(os.path.join(prep_dir(variant), "vocab.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
        import hashlib
        h = hashlib.sha256()
        with open(os.path.join(prep_dir(variant), "set.pkl"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    except (OSError, ValueError) as e:
        return False, f"prep files: {e}"
    if not isinstance(doc, dict):
        return False, "vocab.json is not a JSON object"
    if doc.get("set_sha256") != h.hexdigest():
        return False, "set.pkl does not match vocab.json's set_sha256"
    m = master_of(variant)
    name = NAMES[m]
    try:
        k, snap = master_final(variant, seed)
    except SystemExit as e:
        return False, f"the master: {e}"
    if (doc.get("master_arm") != VARIANTS[m]
            or not same_int(doc.get("master_iterations"), k)
            or not same_int(doc.get("master_offline_steps"),
                            master_offline(variant))):
        return False, (f"vocab.json's master is {doc.get('master_arm')!r} "
                       f"after {doc.get('master_iterations')!r} iterations "
                       f"and {doc.get('master_offline_steps')!r} offline "
                       f"steps, not {name}'s final snapshot (iteration {k})")
    if doc.get("master_snapshot") != snap:
        return False, (f"vocab.json's master snapshot is "
                       f"{doc.get('master_snapshot')!r}, not {snap}")
    log = own_play_rel(m, seed)
    if doc.get("source") != "own-play log" or doc.get("log_dir") != log:
        return False, (f"vocab.json's source is {doc.get('source')!r} "
                       f"({doc.get('log_dir')!r}), not {name}'s own-play "
                       f"log {log}")
    if not valid_ceiling(doc.get("student_ceiling")):
        return False, (f"vocab.json has no valid student_ceiling "
                       f"({doc.get('student_ceiling')!r})")
    d = doc.get("student_discount")
    if (not valid_discount(d)
            or abs(d - DISCOUNTS[variant]) > DISCOUNT_TOL):
        return False, (f"vocab.json's student_discount is {d!r}, not "
                       f"{NAMES[variant]}'s {DISCOUNTS[variant]}")
    if doc.get("discount_rule", "halve") != DISCOUNT_RULE:
        return False, (f"vocab.json's discount_rule is "
                       f"{doc.get('discount_rule', 'halve')!r}, not the "
                       f"lineage's {DISCOUNT_RULE!r}")
    # the snapshot's bytes too, not only its path and iteration: a fresh
    # restart of the master that ends at the same K would otherwise pass
    # (audit r1 B-L2)
    hs = hashlib.sha256()
    try:
        with open(snap, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                hs.update(chunk)
    except OSError as e:
        return False, f"the master snapshot: {e}"
    if doc.get("master_sha256") != hs.hexdigest():
        return False, (f"vocab.json's master_sha256 is not {snap}'s (a prep "
                       f"of another timeline's snapshot)")
    return True, ""


def prep_log_tail(plog):
    """The prep log's last non-empty line (an exit's reason: distill.py's
    SystemExit text or a traceback's last line; audit r1 B-L1), at most
    200 characters, or ''."""
    lines = [l.strip() for l in lines_of(plog) if l.strip()]
    return lines[-1][:200] if lines else ""


def run_prep(variant, seed, env, redo_ok=True):
    """A student's prep, once (vocab.json is its completion signal): the
    master gate and its pick in master.json, then experiments/distill.py
    --from-log F{k-1}'s own-play log up to PREP_ATTEMPTS times. Returns True
    when the prep is done. A done prep that does not check out intact is
    retried for a few minutes (a transient lock), then -- only when
    `redo_ok` (this run was never LAUNCHED: nothing is bound to the set
    yet) -- moved aside and redone; once launched it is NEVER redone, nor
    started at all (a launched student's checkpoint is bound to the set's
    sha256; the attempt loop's integrity gate ALERTs and waits). Before a
    prep starts, choose_master's gate: F{k-1} ABANDONED or NEAR RANDOM is
    PREP FAILED."""
    pdir, prel = prep_dir(variant), prep_rel(variant)
    if not redo_ok and not prep_done(variant):
        # (nothing missing is ever re-derived once the student ran)
        print("ALERT: the prep's vocab.json is missing after this run was "
              "launched -- never redone now; the launch waits for it",
              flush=True)
        return True
    if prep_done(variant):
        pick_bad = False
        for k in range(3):
            ok, why = set_intact(variant, seed)
            # (a finished prep whose master.json was moved aside: refused;
            # read in the same retry loop, so a transient lock is not fatal)
            pick_bad = ok and redo_ok and read_pick(os.path.join(
                pdir, "master.json")) is None
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
            src = os.path.join(pdir, name)
            if os.path.exists(src):
                os.replace(src, f"{src}.damaged-{stamp}")
        print(f"PREP damaged ({why}) -- moved aside (.damaged-{stamp}), "
              f"redoing it", flush=True)
    os.makedirs(pdir, exist_ok=True)
    pick_path = os.path.join(pdir, "master.json")
    # The master is fixed: the gate runs (and master.json is rewritten,
    # atomically) before any prep is started.
    try:
        pick = choose_master(variant, seed)
    except SystemExit as e:
        print(f"PREP FAILED: {e}", flush=True)
        return False
    write_pick(pick_path, pick)
    print(f"MASTER {pick['which']}: {pick['snapshot']} (iteration "
          f"{pick['iterations']}; deployed mean of its last two rows "
          f"{pick['master_mean_last2']:.2f}, g "
          f"{(pick['master_mean_last2'] - G_RANDOM) / G_HUMAN:.3f})",
          flush=True)
    for attempt in range(1, PREP_ATTEMPTS + 1):
        wait_no_trainer()
        # An orphaned prep may have finished while the guard waited: it
        # counts. With no prep alive, its tmp files are stale.
        if prep_done(variant) and set_intact(variant, seed)[0]:
            print(f"PREP done (by an earlier prep): {prel}", flush=True)
            return True
        for name in os.listdir(pdir):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(pdir, name))
                except OSError:
                    pass
        ok, why = manifest_check()
        if not ok:
            print(f"PREP FAILED: {why}", flush=True)
            return False
        plog = os.path.join(pdir, "prep.log")
        with open(plog, "ab") as log:
            p = subprocess.Popen(
                [PY, "experiments/distill.py", "--master", pick["snapshot"],
                 "--master-iterations", str(pick["iterations"]),
                 "--out", prel, "--seed", "0",
                 "--from-log", own_play_rel(master_of(variant), seed)],
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
        if p.returncode == 0 and prep_done(variant):
            ok, why = set_intact(variant, seed)
            if ok:
                print(f"PREP done (attempt {attempt}): {prel}", flush=True)
                return True
            print(f"PREP attempt {attempt}: {why}", flush=True)
        else:
            tail = prep_log_tail(plog)
            print(f"PREP attempt {attempt} failed (rc={p.returncode}; see "
                  f"{prel}/prep.log)"
                  + (f"; its last line: {tail}" if tail else ""), flush=True)
            if any(c in tail for c in PREP_CHECKS):
                print(f"PREP FAILED: PREP CHECK: {tail} -- a retry repeats "
                      f"it (Kai's call)", flush=True)
                return False
        if attempt < PREP_ATTEMPTS:
            time.sleep(PREP_RETRY_S)
    print(f"PREP FAILED after {PREP_ATTEMPTS} attempts -- see "
          f"{prel}/prep.log", flush=True)
    return False


def settle_stop(variant, seed):
    """This run's --stop-env-steps (stop_env_steps), logged in a STOP line;
    read up to 3 times a minute apart (a transient read error is not the
    answer; a spent budget is final at once). None once "PREP FAILED: STOP
    ..." is printed (Kai's call)."""
    why = ""
    for k in range(3):
        try:
            stop = stop_env_steps(variant, seed)
        except SystemExit as e:
            why = str(e)
            if why.startswith("STOP NOT POSITIVE"):
                break
            if k < 2:
                time.sleep(60)
            continue
        steps = target(variant)
        print(f"STOP {variant} s{seed}: --stop-env-steps {stop} = {steps} - "
              f"{steps - stop} (the lineage's target after "
              f"{NAMES[variant]}, less the earlier generations' final env "
              f"steps)", flush=True)
        return stop
    print(f"PREP FAILED: {why}", flush=True)
    return None


def preflight(variant, seed):
    """(ok, reason): the manifest, the quick import probe, then the suites
    IN this tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    try:
        probe = quick(variant, seed)
    except SystemExit as e:
        return False, f"quick probe: {e}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    try:
        # bounded (as PB's audit r1 B-N5): a hung import must not stall the
        # preflight for good
        r = subprocess.run([PY, "-c", probe], cwd=REPO,
                           capture_output=True, text=True, env=env,
                           timeout=QUICK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "quick probe timed out"
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
    detached and survives a supervisor kill), a prep (an orphaned one
    keeps the GPU and writes the prep's files), another experiment's run
    (a manual start beside Generation 2: the launcher waits for its
    completion), a duplicate launch. Python processes only: an editor or a
    git command naming the file is not a trainer. A python whose command
    line cannot be read is not counted (counting it could block every
    launch for good on a foreign process)."""
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
    """PIDs of OTHER python processes running supervise_testb.py `variant`
    (never this process or its ancestors; another variant's supervisor is
    no obstacle)."""
    mine = _mine()
    return [p for p in live_scripts("supervise_testb.py", variant)
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
    against this run's constant ceiling (run_ceiling: F0's sign clip, a
    student's prep's -- read once it is there)."""

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
                # a trainer ran on it (F0's is the sign clip)
                self.ceiling = run_ceiling(self.variant)
            # decisions per TRAJECTORY = per life on Ms. Pac-Man
            tok = (" | frames/decision %s  decisions/life %s  play share "
                   "%s  search share %s" % (
                       r.get("tok_frames_per_decision"),
                       r.get("tok_decisions_per_game"),
                       r.get("tok_duration_play_share"),
                       r.get("tok_duration_search_share")))
            # the rebuild's live reads (as PB's audit r1 B-N4): the search
            # targets' entropy (read (4)), the step-K (5) and step-1
            # value-prefix errors (eval-mode BN, the per-step sets)
            pre = (" | prefix %s  prefix_step1 %s (control %s)" % (
                r.get("prefix"), r.get("prefix_step1"),
                r.get("prefix_step1_control")))
            print("EVAL %s s%d iter %s  train %s  env %s  return %s  H(pi) %s "
                  " maxp %s  erank %s  freshH %s%s | ceiling %s  kept %s  "
                  "max %s%s | elapsed %s s, %s min/iter" % (
                      self.variant, self.seed, r.get("iteration"),
                      r.get("train_steps"), r.get("env_steps"),
                      r.get("return_mean"), r.get("policy_entropy"),
                      r.get("policy_maxp"), r.get("proj_erank"),
                      r.get("fresh_target_entropy"), pre,
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
    run's constant ceiling `ceiling` (the arm did not apply it, or the arm
    is not this one; None: a student's prep's vocab.json holds none to
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
            out.append(f"reward_ceiling {c!r}, the run's constant ceiling is "
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
    an orphan of an earlier supervisor, a prep, another run -- the previous
    generation's or another experiment's --, a duplicate): one trainer on
    the GPU at a time, and never a second writer. `rows` keep streaming
    meanwhile (an orphan of THIS run keeps its EVAL lines and ALERTs)."""
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
    timelines). `own_ok` (a student, launched: its own death before its
    first checkpoint) excuses exactly what that leaves -- weights-iter000
    and the offline curve, which the restart rewrites."""
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
    exact evals' two (acting, search) for the run and its offline curve
    (a student's), and read (5)'s (probe_prefix_bn: the run's rows only;
    the rebuild, as PB's audit r1 B-M1)."""
    run = run_name(variant, seed)
    names = [f"probe_exhaustive_eval-{name}{suffix}.jsonl"
             for name in (run, run + "-offline") for suffix in ("-acting", "")]
    return [os.path.join(REPO, "benchmarks", "probes", name)
            for name in names + [f"probe_prefix_bn-{run}.jsonl"]]


def stray_probe_rows(variant, seed, ckpt):
    """This run's probe jsonls (its offline curve's and read (5)'s too)
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
    if argv and argv[0][:1] in LINEAGES:
        # the variant's lineage: Test B's f0-f4 or Test B2's t0-t2
        configure(argv[0][:1])
    if (len(argv) != 2 or argv[0] not in VARIANTS
            or not argv[1].isdigit() or int(argv[1]) not in SEEDS):
        every = [f"{c}{k}" for c, spec in LINEAGES.items()
                 for k in range(spec["generations"])]
        print(f"usage: supervise_testb.py <variant: {every}> "
              f"<seed: {sorted(SEEDS)}> | --write-manifest", flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    student = is_student(variant)
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    others = other_supervisors(variant)
    if others:
        print(f"refusing: another supervise_testb.py {variant} is alive "
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
                            student and launched(base, ckpt))
    if stray:
        # (A death of a student before its first checkpoint leaves its row
        # 0 and offline curve behind: its own log's LAUNCHED line sanctions
        # the restart, which rewrites exactly those.)
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
    elif not student:
        print(f"{variant} s{seed} starts the lineage: no predecessor to wait "
              f"for (the guard still waits out any trainer)", flush=True)
    else:
        got = wait_external(variant, seed)
        if got == "abandoned":
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not launching", flush=True)
            return 0
        if got != "ready":           # PREP FAILED: MASTER ABANDONED
            return 1
    # THE STOP, before the preflight (a spent budget wastes no suites); the
    # same on every start: the predecessors' final rows never change.
    stop = settle_stop(variant, seed)
    if stop is None:
        return 1

    os.makedirs(base, exist_ok=True)
    # Before the preflight: rows an orphan of this run writes meanwhile are
    # still printed and checked.
    rows = Rows(jsonl, variant, seed)
    rows.prime()
    # The guard once before the preflight too (as PB's audit r1 B-L2): the
    # suites (Ray-spinning slow tests among them) never run beside a live
    # trainer or prep -- an orphan of this run on a resume (its rows
    # stream: primed above), the previous generation's prep, or another
    # experiment's run (f0 beside Generation 2).
    wait_no_trainer(rows)
    polls, told = 0, None
    while True:
        ok, why = preflight(variant, seed)
        if ok:
            break
        if why != told or polls % 6 == 0:
            print(f"PREFLIGHT not ready ({polls * 10} min): {why}", flush=True)
            told = why
        time.sleep(600)
        polls += 1
        rows.emit()          # an orphan's rows between retries
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} s{seed} ({arm}, --stop-env-steps {stop}, at "
          f"most {ITERATIONS} iterations)", flush=True)

    # CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
    # would silently train on the CPU.
    env = {**{k: v for k, v in os.environ.items()
              if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": rel, "PYTHONFAULTHANDLER": "1"}
    if student:
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- no prep, not launching", flush=True)
            return 0
        if not run_prep(variant, seed,
                        {k: v for k, v in env.items()
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
        # The guard first, THEN the manifest and (a student) the prep's
        # integrity (the guard can block for hours; they are checked right
        # before the launch).
        polls, told = 0, None
        while True:
            wait_no_trainer(rows)
            ok, why = manifest_check()
            if ok and student:
                ok, why = set_intact(variant, seed)
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
        stray = (stray_snapshots(base, ckpt, student and launched(base, ckpt))
                 + stray_probe_rows(variant, seed, ckpt))
        if stray:
            names = ", ".join(os.path.basename(x) for x in stray[:4])
            print(f"refusing: {len(stray)} stray snapshots / probe jsonls "
                  f"but no checkpoint ({names}) -- not launching; follow the "
                  f"fresh-restart procedure", flush=True)
            return 2
        args = command(variant, seed, stop) + (
            ["--resume"] if os.path.exists(ckpt) else [])
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
