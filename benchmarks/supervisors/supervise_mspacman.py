"""The real-time line's algorithm on Ms. Pac-Man (Kai, 2026-10-16: "I'd like
to test the current algorithm on ms pacman when the current code is done
running"; pre-registered in CURRICULUM.md "Next experiment (pre-registered)
-- the real-time algorithm on Ms. Pac-Man"). SEED 0, --ray --cpu-actor (the
arms' launch pin), MsPacmanNoFrameskip-v4 through the lineage's wrapper
stack, one game's 3 lives per iteration.

    python benchmarks/supervisors/supervise_mspacman.py MODE 0 [--kai-override]
    (MODE: auto | gumbel | temp)

"current algorithm" = the base the Gumbel seed-1 decision (CURRICULUM, 10-15)
leaves: REPLICATES -> gumbel (effzero-repro-faithful-gumbelact8-birth-
mspacman), NO BREAKOUT -> temp (effzero-repro-faithful-prioract-noredo-
temp05-bnrecal-mspacman), PARTIAL -> Kai's call. "auto" waits for Gumbel
seed 1's completion marker AND its full readout (exact rows 45-90, recal
60-90; ALERTed if still incomplete 2 h after seed 1 completed), applies
that decision, logs it, and on PARTIAL exits 2 without launching (relaunch
with Kai's pick, "gumbel" or "temp") -- unless exactly one variant's
checkpoint is already in the folder (Kai's pick, launched before a
reboot): "auto" then resumes that variant. An explicit variant waits for
the same readout (the run must never overlap seed 1 or its evals) and must
agree with a definite decision (REPLICATES / NO BREAKOUT); after PARTIAL
either is accepted (Kai's pick). --kai-override accepts a disagreeing
explicit variant, logged (Kai's explicit instruction only; ignored, with a
note, by "auto"). Either way the main tree must then hold the audited
revision (the manifest in the MAIN tree, which only the merge brings), and
the quick probe + 15 suites must pass -- once per supervisor process;
every relaunch attempt re-checks the manifest. A live ablation.py of
EITHER variant's arm at this seed (an orphan of an earlier supervisor, or
a duplicate) holds every launch. Relaunch with the arguments the run was
started with.

Launch with cwd Y:/SwitchPlayer and stdout appended to
benchmarks/results/mspacman-s0-2026-10-16/supervisor.log -- create that
folder first (the shell redirect opens the log before Python runs; the
eval hand-off reads it for RUN COMPLETE / CRASH LOOP). Both variants share
the folder (the arm is in every file name); a checkpoint of the OTHER
variant there refuses the launch.

Supervisor v2 pattern; a relaunch auto-resumes from the last checkpoint."""
import json
import os
import re
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
VARIANTS = {
    "gumbel": "effzero-repro-faithful-gumbelact8-birth-mspacman",
    "temp": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman",
}
MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
SEEDS = {0: "2026-10-16"}                 # the pre-registered seed and date
REL = "benchmarks/results/mspacman-s{seed}-{date}"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
MAX_ATTEMPTS = 40

# Gumbel seed 1 (pong-gumbelact8-s1-2026-10-15): the run this one queues
# behind, and the readout its pre-registered decision is made from.
S1_RUN = "pong-gumbelact8-s1-2026-10-15"
S1_CKPT = os.path.join(REPO, "benchmarks", "results", S1_RUN,
                       "resume-pong-skip4-effzero-repro-faithful-gumbelact8-"
                       "birth-seed1.ckpt")
S1_RAW = os.path.join(REPO, "benchmarks", "probes",
                      f"probe_exhaustive_eval-{S1_RUN}.jsonl")
S1_RECAL = os.path.join(REPO, "benchmarks", "probes",
                        f"probe_bn_recal_eval-{S1_RUN}.jsonl")
# The pre-registered external-recal calibration snapshot (post_gumbel_s1's
# CAL): a rerun with another one must not replace those rows.
S1_CAL = ("benchmarks/results/pong-prioract-s0-2026-09-27/snapshots-pong-"
          "skip4-effzero-repro-faithful-prioract-seed0/weights-iter085.pth")
RAW_ROWS = tuple(range(45, 95, 5))        # exact evals 45-90 (primary)
MEAN_ROWS = tuple(range(60, 95, 5))       # the level read: raw mean 60-90
RECAL_ROWS = MEAN_ROWS                    # external recal 60-90
NO_WORSE = 3.68                           # the faithful control's band


def paths(arm, seed):
    """(BASE, JSONL, CKPT, DEATHS) of `arm` at `seed`."""
    base = os.path.join(REPO, *REL.format(seed=seed,
                                          date=SEEDS[seed]).split("/"))
    return (base, os.path.join(base, f"ablation{TAG}.jsonl"),
            os.path.join(base, f"resume{TAG}-{arm}-seed{seed}.ckpt"),
            os.path.join(base, "deaths.log"))


def command(arm, seed):
    """The pre-registered recipe: Pong's, with the env, and 3 episodes
    (one game's lives) per iteration."""
    return [PY, "experiments/ablation.py", "--arms", arm, "--seed", str(seed),
            "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
            "--env", ENV_ID, "--frameskip", "4", "--episode-cap", "3000",
            "--episodes-per-iter", "3", "--checkpoint-every", "2",
            "--snapshot-weights"] + MODE


def seed_state(ck):
    """'done' (completion marker -- a shrunk checkpoint -- or the run's own
    supervisor logging RUN COMPLETE, if the checkpoint was moved), 'failed'
    (its supervisor gave up: CRASH LOOP / MAX ATTEMPTS), else 'running'."""
    try:
        if os.path.getsize(ck) < 1_000_000:
            return "done"
    except OSError:
        pass
    try:
        with open(os.path.join(os.path.dirname(ck), "supervisor.log"), "rb") as f:
            log = f.read()
    except OSError:
        return "running"
    if b"RUN COMPLETE" in log:
        return "done"
    if b"CRASH LOOP" in log or b"MAX ATTEMPTS" in log:
        return "failed"
    return "running"


def read_rows(path, raw=False, calibration=None):
    """{iteration: exact_mean record} from a probe jsonl; a later record of
    the same iteration wins (a rerun). raw=True keeps only the full 30-noop
    prior-argmax records (variant "raw", no max_noop); `calibration` keeps
    only recal records calibrated on that snapshot. Torn or foreign lines
    are skipped."""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or not isinstance(r.get("iteration"), int):
            continue
        if not isinstance(r.get("exact_mean"), (int, float)):
            continue
        if raw and (r.get("variant") != "raw" or "max_noop" in r):
            continue
        if calibration is not None and not str(r.get("calibration", "")).replace(
                chr(92), "/").endswith(calibration):
            continue
        out[r["iteration"]] = r
    return out


def s1_readout():
    """(raw, recal) when Gumbel seed 1's pre-registered readout is complete
    (every exact row 45-90 and every recal row 60-90), else None."""
    raw = read_rows(S1_RAW, raw=True)
    recal = read_rows(S1_RECAL, calibration=S1_CAL)
    if all(i in raw for i in RAW_ROWS) and all(i in recal for i in RECAL_ROWS):
        return raw, recal
    return None


def decide(raw, recal):
    """Gumbel seed 1's pre-registered TWO-SEED DECISION (CURRICULUM 10-15),
    as (variant or None, label, detail):
      first success = the first row 45-90 with exact >= +10 or >= 3/4
        phases won;
      COLLAPSE = a row after it below 0, unless its recal row is >= 0 (BN
        artifact); rows before 60 have no recal and count as they are;
      STABLE = no collapse AND first success <= 80;
      REPLICATES = breakout, mean 60-90 >= +3.68 (NO WORSE) and STABLE ->
        gumbel; PARTIAL = breakout otherwise -> None (Kai's call);
      NO BREAKOUT -> temp."""
    first = next((i for i in RAW_ROWS if raw[i]["exact_mean"] >= 10
                  or raw[i].get("phases_won", 0) >= 3), None)
    if first is None:
        return "temp", "NO BREAKOUT", "no first success at rows 45-90"
    mean = sum(raw[i]["exact_mean"] for i in MEAN_ROWS) / len(MEAN_ROWS)
    collapses = [i for i in RAW_ROWS if i > first
                 and raw[i]["exact_mean"] < 0
                 and not (i in recal and recal[i]["exact_mean"] >= 0)]
    detail = (f"first success @{first}, mean 60-90 {mean:+.4f}, "
              f"collapses {collapses}")
    if mean >= NO_WORSE and not collapses and first <= 80:
        return "gumbel", "REPLICATES", detail
    return None, "PARTIAL", detail


# Files whose AUDITED content must be in the main tree (sha256 of the text
# with CRLF normalized to LF, so git's autocrlf round trip does not matter).
# Written by `supervise_mspacman.py --write-manifest` (it hashes the tree the
# script sits in -- the mspacman worktree -- and writes next to the script)
# when the audit loop closes; READ from the main tree (audit r1 L2), so the
# check never depends on the worktree surviving the run, and before the
# merge brings it the launch simply waits. Any later edit to an AUDITED file
# in the main tree blocks every launch.
MANIFEST_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mspacman_manifest.json")
MANIFEST = os.path.join(REPO, "benchmarks", "supervisors",
                        "mspacman_manifest.json")
# Every repo module the run executes, the tests the preflight runs (plus
# the two harness suites the eval change touches), the eval probe the
# readout runs, and the supervisors the tests exec.
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "experiments/ablation.py",
           "experiments/ray_actors.py", "tests/test_prioract.py",
           "tests/test_sve.py", "tests/test_searchact.py",
           "tests/test_factor.py", "tests/test_prioract2.py",
           "tests/test_tempscale.py", "tests/test_seqref.py",
           "tests/test_bnrecal.py", "tests/test_searchact8.py",
           "tests/test_sve2.py", "tests/test_gumbelact8.py",
           "tests/test_mspacman.py", "tests/test_mcts.py",
           "tests/test_vtree_communal.py", "tests/test_harness.py",
           "tests/test_ray_harness.py", "tests/test_checkpoint_resume.py",
           "tests/test_paper_mode.py", "tests/test_reanalyze_refresh.py",
           "tests/test_reanalysis.py", "tests/test_vtree.py", "pytest.ini",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_gumbelact8.py",
           "benchmarks/supervisors/supervise_mspacman.py"]
SUITES = ["tests/test_prioract.py", "tests/test_sve.py",
          "tests/test_searchact.py", "tests/test_factor.py",
          "tests/test_prioract2.py", "tests/test_tempscale.py",
          "tests/test_seqref.py", "tests/test_bnrecal.py",
          "tests/test_searchact8.py", "tests/test_gumbelact8.py",
          "tests/test_mspacman.py", "tests/test_paper_mode.py",
          "tests/test_mcts.py", "tests/test_reanalyze_refresh.py",
          "tests/test_vtree_communal.py"]


def file_hashes(root):
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def manifest_check():
    """(ok, reason): the main tree's AUDITED files hash to the manifest.
    Run before EVERY launch attempt: a relaunch must never run an unaudited
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


def quick_probe(arm):
    """The import probe: the arm, its two converted knobs, whole-game eval,
    and the env's 9 actions -- in the main tree."""
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, gymnasium as gym; "
            f"spec = ablation.ARMS[{arm!r}]; "
            "assert spec['MAX_REPLAY'] == 255 and "
            "spec['POLICY_REFRESH_COUNT'] == 12; "
            "assert spec['_launch'] == {'ray': True, 'cpu_actor': True}; "
            "assert hasattr(ablation.Run, 'play_game'); "
            f"assert ablation.env_tag_fragment({ENV_ID!r}) == "
            f"{TAG.split('-')[1]!r}; "
            f"e = gym.make({ENV_ID!r}); "
            "assert e.action_space.n == 9; e.close()")


def preflight(arm):
    """(ok, reason). The audited-revision manifest, the quick import probe,
    then the 15 suites IN the main tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    try:
        r = subprocess.run([PY, "-c", quick_probe(arm)], cwd=REPO,
                           capture_output=True, text=True, env=env,
                           timeout=600)
    except subprocess.TimeoutExpired:
        return False, "quick probe timed out"
    if r.returncode:
        return False, "quick probe: " + (r.stderr.strip().splitlines() or ["?"])[-1]
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p",
                            "no:cacheprovider"] + SUITES,
                           cwd=REPO, capture_output=True, text=True, env=env,
                           timeout=2400)
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


def live_children(arm, seed):
    """PIDs of running ablation.py processes for this arm and seed (an
    orphaned child of an earlier supervisor -- it is started detached and
    survives a supervisor kill -- or a duplicate launch: audit r1 L4)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if (any(c.replace(chr(92), "/").endswith("experiments/ablation.py")
                for c in cmd)
                and arm in cmd and "--seed" in cmd
                and cmd[cmd.index("--seed") + 1:][:1] == [str(seed)]):
            out.append(p.info["pid"])
    return out


def variant_ckpts(seed):
    """The variants whose checkpoint (live or completion marker) is in the
    shared folder."""
    return [v for v, arm in VARIANTS.items()
            if os.path.exists(paths(arm, seed)[2])]


def other_variant_ckpt(variant, seed):
    """The other variant's checkpoint in the shared folder, if any (a run of
    it was started there: this launch would make it two runs)."""
    for v, arm in VARIANTS.items():
        if v != variant and os.path.exists(paths(arm, seed)[2]):
            return paths(arm, seed)[2]
    return None


class Rows:
    """New jsonl rows -> one summary block each on stdout."""

    def __init__(self, jsonl, seed):
        self.jsonl, self.seed, self.seen, self.last = jsonl, seed, set(), None

    def lines(self):
        try:
            with open(self.jsonl, "rb") as f:
                return [l for l in f.read().decode("utf-8", "replace").splitlines()
                        if l.strip()]
        except OSError:
            return []

    def prime(self):
        for line in self.lines():
            self.seen.add(line.strip())

    def emit(self):
        for line in self.lines():
            key = line.strip()
            if key in self.seen:
                continue
            self.seen.add(key)
            try:
                r = json.loads(line)
                el = r.get("elapsed_s")
                per = ("%.1f" % ((el - self.last[0]) / 60.0 /
                                 max(1, r.get("iteration", 0) - self.last[1]))
                       if isinstance(el, (int, float)) and self.last
                       and r.get("iteration", 0) > self.last[1] else "n/a")
                if isinstance(el, (int, float)):
                    self.last = (el, r.get("iteration", 0))
                print("EVAL s%d iter %s  train %s  env %s  H(pi) %.3f  maxp %.3f  "
                      "freshH %s  erank %s  score %.1f (max %s)" % (
                          self.seed, r.get("iteration"), r.get("train_steps"),
                          r.get("env_steps"),
                          r.get("policy_entropy", float("nan")),
                          r.get("policy_maxp", float("nan")),
                          ("%.3f" % r["fresh_target_entropy"]
                           if isinstance(r.get("fresh_target_entropy"), float)
                           else "n/a"),
                          r.get("proj_erank", "n/a"),
                          r.get("return_mean", float("nan")),
                          r.get("return_max", "n/a")), flush=True)
                print("      game len %s  capped %s | step gn %s loss %s | birth "
                      "window %s pos | eager [it, step] %s | elapsed %s s, %s "
                      "min/iter since last row" % (
                          r.get("eval_len_mean"), r.get("eval_capped"),
                          r.get("step_grad_norm_mean", "n/a"),
                          r.get("step_loss_mean", "n/a"),
                          r.get("birth_positions_window", "n/a"),
                          r.get("eager_collect_steps", "n/a"), el, per),
                      flush=True)
            except Exception:
                pass


def snapshot(deaths_log, seed, attempt, rc, lifetime):
    """Append the death-time environment to deaths.log; one-line summary
    to stdout for the event stream."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"=== death attempt={attempt} rc={rc} lifetime={lifetime:.0f}s "
             f"at {stamp}"]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["nvidia-smi", "--query-compute-apps=pid,process_name,"
                 "used_memory", "--format=csv,noheader"],
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
    print(f"DEATH s{seed} rc={rc} after {lifetime:.0f}s; snapshot -> deaths.log "
          f"(gpu used {gpu})", flush=True)


READOUT_ALERT_S = 2 * 3600     # seed 1's evals take ~30 min


def wait_for_s1():
    """Block until Gumbel seed 1 is complete AND its readout is; returns
    (raw, recal). A FAILED seed 1 is ALERTed and waited on (its relaunch
    must not race this run); so is a readout still incomplete 2 h after
    seed 1 completed (a failed probe: rerun it by hand -- audit r1 L1)."""
    waited, told, done_at = 0, None, None
    while True:
        state = seed_state(S1_CKPT)
        readout = s1_readout() if state == "done" else None
        if readout is not None:
            return readout
        if state == "done" and done_at is None:
            # The completion marker's write time when there is one, so a
            # relaunch does not restart the 2-h clock (audit r2 NIT).
            done_at = time.time()
            try:
                if os.path.getsize(S1_CKPT) < 1_000_000:
                    done_at = min(done_at, os.path.getmtime(S1_CKPT))
            except OSError:
                pass
        raw = read_rows(S1_RAW, raw=True)
        recal = read_rows(S1_RECAL, calibration=S1_CAL)
        if state == "failed":
            msg = ("ALERT: Gumbel seed 1 FAILED -- holding the Ms. Pac-Man "
                   "launch until it is relaunched and completes")
        elif done_at is not None and time.time() - done_at > READOUT_ALERT_S:
            msg = ("ALERT: Gumbel seed 1 completed > 2 h ago but its readout "
                   "is incomplete -- missing exact %s, recal %s (rerun the "
                   "failed probe; this launch waits for it)" % (
                       [i for i in RAW_ROWS if i not in raw],
                       [i for i in RECAL_ROWS if i not in recal]))
        else:
            msg = ("WAITING for Gumbel seed 1 (%s; readout %d/%d exact, %d/%d "
                   "recal)" % (state, sum(i in raw for i in RAW_ROWS),
                               len(RAW_ROWS),
                               sum(i in recal for i in RECAL_ROWS),
                               len(RECAL_ROWS)))
        if msg != told or waited % 60 == 0:
            print(f"{msg} [{waited // 60} h]", flush=True)
            told = msg
        time.sleep(60)
        waited += 1


def main(argv):
    if "--write-manifest" in argv:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        # The manifest hashes the working tree: refuse uncommitted edits to
        # an AUDITED file, or the manifest would describe code no commit
        # holds (audit r3 M1).
        r = subprocess.run(["git", "-C", root, "status", "--porcelain",
                            "--"] + AUDITED, capture_output=True, text=True)
        if r.returncode or r.stdout.strip():
            print("refusing --write-manifest: commit these AUDITED files "
                  "first:\n" + (r.stdout or r.stderr), flush=True)
            return 1
        with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
            json.dump(file_hashes(root), f, indent=1, sort_keys=True)
        print(f"wrote {MANIFEST_OUT} from {root}")
        return 0
    override = "--kai-override" in argv
    pos = [a for a in argv if a != "--kai-override"]
    if (len(pos) != 2 or pos[0] not in ("auto", *VARIANTS)
            or not pos[1].isdigit()):
        print("usage: supervise_mspacman.py {auto|gumbel|temp} SEED "
              "[--kai-override]", flush=True)
        return 64
    variant, seed = pos[0], int(pos[1])
    if seed not in SEEDS:
        print(f"supervise_mspacman: pre-registered for seeds {sorted(SEEDS)}; "
              f"refusing seed {seed}", flush=True)
        return 64

    raw, recal = wait_for_s1()
    picked, label, detail = decide(raw, recal)
    print(f"DECISION (Gumbel seed 1, pre-registered): {label} -- {detail}",
          flush=True)
    if variant == "auto":
        if override:
            print("note: --kai-override is ignored by 'auto' (it applies to "
                  "an explicit variant)", flush=True)
        have = variant_ckpts(seed)
        if picked is None and len(have) == 1:
            # Kai's pick was launched (explicitly) before this relaunch:
            # resume it rather than hold (audit r1 L3).
            picked = have[0]
            print(f"resuming {picked}: Kai's pick after PARTIAL, its "
                  f"checkpoint is in the folder", flush=True)
        if picked is None:
            print("HOLD: PARTIAL is Kai's call -- relaunch with 'gumbel' or "
                  "'temp'; nothing launched", flush=True)
            return 2
        variant = picked
    elif picked is not None and picked != variant and not override:
        # A definite decision picks the arm (audit r2 NIT): an explicit
        # variant that disagrees is a mistake unless Kai said so.
        print(f"ALERT: {variant} given explicitly but the decision picks "
              f"{picked} -- refusing (--kai-override if Kai chose this)",
              flush=True)
        return 3
    else:
        print(f"variant {variant} given explicitly"
              + (" (KAI OVERRIDE of the decision)" if picked not in
                 (None, variant) else ""), flush=True)
    arm = VARIANTS[variant]
    base, jsonl, ckpt, deaths_log = paths(arm, seed)
    clash = other_variant_ckpt(variant, seed)
    if clash:
        print(f"ALERT: {clash} exists -- the other variant already ran here; "
              f"refusing to start {variant} (relaunch with the arguments it "
              f"was started with, incl. any --kai-override)", flush=True)
        return 3

    polls, told = 0, None
    while True:
        ok, why = preflight(arm)
        if ok:
            break
        if why != told or polls % 6 == 0:
            # Before the merge the manifest is simply not in main yet;
            # once it is, a failing preflight is an ALERT (audit r3 M1).
            alert = "ALERT: " if os.path.exists(MANIFEST) else ""
            print(f"{alert}PREFLIGHT not ready ({polls * 10} min): {why}",
                  flush=True)
            told = why
        time.sleep(600)
        polls += 1
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} ({arm})", flush=True)

    os.makedirs(base, exist_ok=True)
    rows = Rows(jsonl, seed)
    rows.prime()
    # CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
    # would silently train on the CPU.
    env = {**{k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": REL.format(seed=seed, date=SEEDS[seed]),
           "PYTHONFAULTHANDLER": "1"}
    stall_s = 2 * 3600    # the child logs at least every checkpoint (2 iters)
    deaths, quick = [], 0

    def completed():
        try:
            return os.path.getsize(ckpt) < 1_000_000
        except OSError:
            return False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if completed():
            print(f"RUN COMPLETE s{seed} (checkpoint is a marker)", flush=True)
            return 0
        polls, told = 0, None
        while True:
            ok, why = manifest_check()
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the main tree is "
                      f"the audited revision [{polls} min]", flush=True)
                told = why
            time.sleep(60)
            polls += 1
        polls, told = 0, None
        while True:
            # Either variant's arm (audit r3 L1): the other variant's
            # orphan has no checkpoint until its iteration 2.
            pids = sorted({p for a in VARIANTS.values()
                           for p in live_children(a, seed)})
            if not pids:
                break
            msg = (f"ALERT: an ablation.py of a Ms. Pac-Man arm at seed {seed} "
                   f"is already running (pids {pids}) -- not launching a "
                   f"second writer")
            if msg != told or polls % 60 == 0:
                print(f"{msg} [{polls} min]", flush=True)
                told = msg
            time.sleep(60)
            polls += 1
        args = command(arm, seed) + (["--resume"] if os.path.exists(ckpt) else [])
        log = open(os.path.join(base, f"supv2_{attempt}.log"), "ab")
        born = time.time()
        p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=0x08000000)
        print(f"LAUNCHED s{seed} {variant} attempt {attempt} "
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
        snapshot(deaths_log, seed, attempt, rc, lifetime)
        if completed():
            print(f"RUN COMPLETE s{seed}", flush=True)
            return 0
        deaths.append(time.time())
        if len([t for t in deaths if time.time() - t < 3600]) >= 5:
            print("CRASH LOOP: 5 deaths within 60 min -- stopping for diagnosis",
                  flush=True)
            return 1
        if lifetime < 600:
            quick += 1
            wait = min(900, 60 * (2 ** quick))
            print(f"quick death #{quick}; backing off {wait}s", flush=True)
            time.sleep(wait)
        else:
            quick = 0
            time.sleep(15)
    print("MAX ATTEMPTS REACHED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
