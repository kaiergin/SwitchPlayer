"""SVE on the temperature + stabilizer config (Kai, 2026-10-10):
effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve, --ray
--cpu-actor (pinned by the arm). SEED 1 (twin pong-bnrecal-s1; done
10-11) and SEED 0 (Kai, 2026-10-12; reference pong-tempscale-s0). Design
and pre-registered reads: CURRICULUM.md "Next experiment (pre-registered)
-- SVE on the temperature + stabilizer config" and "... SVE seed 0".

    python benchmarks/supervisors/supervise_sve2.py ray 0
    python benchmarks/supervisors/supervise_sve2.py ray 1

Supervisor v2 pattern; a relaunch auto-resumes from the last checkpoint."""
import json
import math
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "ray": ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve",
            "pong-sve2", ["--ray", "--cpu-actor"]),
}
WRITE = "--write-manifest" in sys.argv
VARIANT = "ray" if WRITE else sys.argv[1]
SEED = 0 if WRITE else int(sys.argv[2])
# Each pre-registered seed and the run its stop rule reads the same-row
# `value` from: seed 1's twin (same arm without SVE, same seed); seed 0 has
# no stabilizer twin, so its reference is the temperature config at seed 0
# (the stabilizer changes only the actor's BN statistics, not training).
TWINS = {1: "pong-bnrecal-s1-2026-10-08", 0: "pong-tempscale-s0-2026-10-05"}
if not WRITE and SEED not in TWINS:
    sys.exit(f"supervise_sve2: this arm is pre-registered as SEED 1 and "
             f"SEED 0 only; refusing seed {SEED}")
REPO = r"Y:\SwitchPlayer"
ARM, _stem, _mode = VARIANTS[VARIANT]
REL = f"benchmarks/results/{_stem}-s{SEED}-2026-10-10"


WAIT_CKPTS = []      # one run; the GPU is free
BASE = os.path.join(REPO, *REL.split("/"))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(BASE, f"resume-pong-skip4-{ARM}-seed{SEED}.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms", ARM, "--seed", str(SEED),
          "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
          "--env", "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--snapshot-weights"] + _mode
MAX_ATTEMPTS = 40

# The pre-registered STOP RULE (CURRICULUM 2026-10-10; review with Kai, not
# an automatic kill): the TD-referenced `value` above STOP_RATIO x the twin's
# value at the same row for STOP_ROWS consecutive rows from STOP_FROM (the
# first row with SVE in effect: T1 = 40k steps = iteration 20; rows every 5).
TWIN_JSONL = os.path.join(REPO, "benchmarks", "results", TWINS.get(SEED, ""),
                          "ablation-pong-skip4.jsonl")
STOP_FROM = 25
STOP_RATIO = 2.0
STOP_ROWS = 3


def twin_values(path=TWIN_JSONL):
    """{iteration: TD-referenced value} of the twin's rows."""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                v = r.get("value")
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v)):
                    out[r.get("iteration")] = float(v)
    except OSError:
        pass
    return out


def stop_streak(over, upto):
    """Trailing run of rows above the ratio among the recorded iterations
    <= upto (consecutive recorded rows, newest first)."""
    streak = 0
    for k in sorted((k for k in over if k <= upto), reverse=True):
        if not over[k]:
            break
        streak += 1
    return streak


def stop_rule_step(it, value, twin, over):
    """(ratio, streak, alert) for one row: ratio = value / the twin's value
    at the same row. `over` maps each usable row's ITERATION to ratio >
    STOP_RATIO, so a row re-run by a resume overwrites its own entry instead
    of extending the streak (sve2 audit r2 LOW-A); the streak is the
    trailing run of True entries up to `it`, and the alert fires once it
    reaches STOP_ROWS. A row before STOP_FROM, without a finite numeric
    value, or without a positive twin value records nothing."""
    tv = twin.get(it)
    if (not isinstance(it, int) or isinstance(it, bool) or it < STOP_FROM
            or not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value)
            or not isinstance(tv, float) or tv <= 0):
        return None, (stop_streak(over, it) if isinstance(it, int) else 0), False
    ratio = value / tv
    over[it] = ratio > STOP_RATIO
    streak = stop_streak(over, it)
    return ratio, streak, streak >= STOP_ROWS


TWIN = twin_values()
STOP_OVER = {}      # {iteration: ratio > STOP_RATIO} for the stop rule
seen = set()


def rows():
    try:
        with open(JSONL, "rb") as f:
            return [l for l in f.read().decode("utf-8", "replace").splitlines()
                    if l.strip()]
    except OSError:
        return []


def emit_new_rows():
    for line in rows():
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        # The stop rule FIRST, so a print error below cannot skip a row
        # (sve2 audit r2 LOW-C).
        try:
            ratio, streak, alert = stop_rule_step(
                r.get("iteration"), r.get("value"), TWIN, STOP_OVER)
        except Exception as e:
            ratio, streak, alert = None, 0, False
            print(f"ALERT: stop rule failed on a row: {e!r}", flush=True)
        try:
            print("EVAL s%d iter %s  train %s  H(pi) %.3f  maxp %.3f  freshH %s  "
                  "erank %s  dead %s  return %.1f" % (
                      SEED, r.get("iteration"), r.get("train_steps"),
                      r.get("policy_entropy", float("nan")),
                      r.get("policy_maxp", float("nan")),
                      ("%.3f" % r["fresh_target_entropy"]
                       if isinstance(r.get("fresh_target_entropy"), float) else "n/a"),
                      r.get("proj_erank", "n/a"), r.get("proj_dead_frac", "n/a"),
                      r.get("return_mean", float("nan"))), flush=True)
            el = r.get("elapsed_s")
            per = ("%.1f" % ((el - emit_new_rows.last[0]) / 60.0 /
                             max(1, r.get("iteration", 0) - emit_new_rows.last[1]))
                   if isinstance(el, (int, float)) and emit_new_rows.last
                   and r.get("iteration", 0) > emit_new_rows.last[1] else "n/a")
            if isinstance(el, (int, float)):
                emit_new_rows.last = (el, r.get("iteration", 0))
            print("      env %s  len %s  capped %s | redo window %s %s (total %s)  "
                  "gn pre/post %s/%s  step gn %s loss %s | elapsed %s s, "
                  "%s min/iter since last row" % (
                      r.get("env_steps"), r.get("eval_len_mean"),
                      r.get("eval_capped"), r.get("redo_recycled_window", "n/a"),
                      r.get("redo_by_net_window", ""),
                      r.get("redo_recycled_total", "n/a"),
                      r.get("redo_gn_pre20", "n/a"), r.get("redo_gn_post20", "n/a"),
                      r.get("step_grad_norm_mean", "n/a"),
                      r.get("step_loss_mean", "n/a"), el, per), flush=True)
            print("      behaviour T %s  H_act %s  maxp %s  act_agree %s  "
                  "prior_agree %s  KL(tgt||prior) %s  rare %s | birth window "
                  "%s pos | eager [it, step] %s" % (
                      r.get("beh_T"), r.get("beh_H_act"), r.get("beh_prior_maxp"),
                      r.get("beh_act_agree"), r.get("beh_prior_agree"),
                      r.get("beh_kl_target_prior"), r.get("beh_rare_rate"),
                      r.get("birth_positions_window", "n/a"),
                      r.get("eager_collect_steps", "n/a")), flush=True)
        except Exception:
            pass
        # The STOP RULE's quantity, next to the twin's (sve2 audit r1 M1).
        try:
            print("      TD-referenced value %s control %s | twin %s  ratio %s  "
                  "(streak %d of %d)" % (
                      r.get("value"), r.get("value_control"),
                      TWIN.get(r.get("iteration")),
                      "n/a" if ratio is None else "%.2f" % ratio,
                      streak, STOP_ROWS), flush=True)
        except Exception:
            pass
        if alert:
            print("ALERT: STOP RULE -- TD-referenced value > %.0fx the twin's "
                  "for %d consecutive rows (from row %d) -> REVIEW WITH KAI "
                  "(not an automatic kill)" % (STOP_RATIO, streak, STOP_FROM),
                  flush=True)
        try:
            # SVE diagnostics (pre-registered read (4)).
            print("      sve frac %s  buffer_frac %s  td_gap %s  td_bias %s  "
                  "value_age %s  searches %s (%s s)" % (
                      r.get("sve_frac"), r.get("sve_buffer_frac"),
                      r.get("sve_td_gap"), r.get("sve_td_bias"),
                      r.get("sve_value_age"), r.get("sve_searches"),
                      r.get("sve_search_s")), flush=True)
        except Exception:
            pass


emit_new_rows.last = None     # (elapsed_s, iteration) of the previous row


def replay_rows():
    """A relaunch marks the rows already written as seen (they are not
    re-printed) and replays the stop rule over them."""
    for line in rows():
        seen.add(line.strip())
        try:
            r = json.loads(line)
            stop_rule_step(r.get("iteration"), r.get("value"), TWIN, STOP_OVER)
        except (ValueError, AttributeError):
            pass


def stop_rule_armed_line():
    """The startup line: how many twin rows the stop rule can use."""
    n = sum(1 for k in TWIN if isinstance(k, int) and k >= STOP_FROM)
    head = "stop rule armed" if n >= 14 else "ALERT: stop rule NOT fully armed"
    return (f"{head}: {n} twin rows from row {STOP_FROM} "
            f"({TWINS.get(SEED)}), alert at {STOP_ROWS} consecutive rows > "
            f"{STOP_RATIO:.0f}x")


def completed():
    try:
        return os.path.getsize(CKPT) < 1_000_000
    except OSError:
        return False


def snapshot(attempt, rc, lifetime):
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
    with open(DEATHS, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    gpu = lines[1].split(",")[0].strip() if len(lines) > 1 else "?"
    print(f"DEATH s{SEED} rc={rc} after {lifetime:.0f}s; snapshot -> deaths.log "
          f"(gpu used {gpu})", flush=True)


def _seed_state(ck):
    """'done' (completion marker -- a shrunk checkpoint -- or the seed's own
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


# Files whose AUDITED content must be in the main tree (sha256 of the text
# with CRLF normalized to LF, so git's autocrlf round trip does not matter).
# Written by `supervise_sve2.py --write-manifest` in the worktree when the
# audit loop closes; any later edit to one of them blocks the launch.
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sve2_manifest.json")
# Every repo module the run executes (SVE audit r1 L4), plus the tests
# and this supervisor.
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "experiments/ablation.py",
           "experiments/ray_actors.py", "tests/test_prioract.py",
           "tests/test_sve.py", "tests/test_searchact.py",
           "tests/test_factor.py", "tests/test_prioract2.py",
           "tests/test_tempscale.py", "tests/test_seqref.py",
           "tests/test_bnrecal.py", "tests/test_searchact8.py",
           "tests/test_sve2.py", "tests/test_vtree_communal.py",
           "tests/test_harness.py",
           "tests/test_ray_harness.py", "tests/test_checkpoint_resume.py",
           "tests/test_paper_mode.py", "tests/test_reanalyze_refresh.py",
           "tests/test_reanalysis.py", "tests/test_vtree.py", "pytest.ini",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_sve2.py"]


def file_hashes(root):
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


if "--write-manifest" in sys.argv:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(file_hashes(root), f, indent=1, sort_keys=True)
    print(f"wrote {MANIFEST} from {root}")
    sys.exit(0)


# The main tree must hold the AUDITED revision before anything launches (the
# change is merged there by hand). Otherwise argparse / import would kill
# every attempt and the supervisor would burn its crash-loop budget with the
# GPU idle -- or, worse, a stale revision would run.
QUICK = ("import sys; sys.path[:0] = ['.', 'experiments']; "
         "import muzero, ablation, redo; "
         f"assert {ARM!r} in ablation.ARMS and hasattr(muzero, 'BIRTH_SEARCH'); "
         "assert hasattr(redo, '_solve_bias') and hasattr(ablation, 'redo_due'); "
         "assert hasattr(muzero, 'EVAL_PRIOR_ONLY'); "
         "assert 'effzero-repro-faithful-prioract-noredo' in ablation.ARMS; "
         "assert hasattr(ablation, 'train_block') and "
         "hasattr(muzero, 'RAY_EAGER_COLLECT'); "
         f"assert ablation.ARMS[{ARM!r}]['VALUE_TARGET_MODE'] == 'mixed'; "
         f"assert ablation.ARMS[{ARM!r}]['BN_RECAL_ACTING'] is True")


def manifest_check():
    """(ok, reason): the main tree's AUDITED files hash to the manifest.
    Run before EVERY launch attempt, not just the first (SVE audit r1 L4):
    a relaunch must never run an unaudited edit."""
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


def preflight():
    """(ok, reason). The audited-revision manifest, the quick import probe,
    then the prioract + SVE + searchact + factor + prioract2 + tempscale +
    seqref + bnrecal + searchact8 + sve2 + reanalyze_refresh +
    vtree_communal suites run IN the main tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([PY, "-c", QUICK], cwd=REPO, capture_output=True,
                       text=True, env=env)
    if r.returncode:
        return False, "quick probe: " + (r.stderr.strip().splitlines() or ["?"])[-1]
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p",
                            "no:cacheprovider", "tests/test_prioract.py",
                            "tests/test_sve.py", "tests/test_searchact.py",
                            "tests/test_factor.py", "tests/test_prioract2.py",
                            "tests/test_tempscale.py", "tests/test_seqref.py",
                            "tests/test_bnrecal.py", "tests/test_searchact8.py",
                            "tests/test_sve2.py",
                            "tests/test_reanalyze_refresh.py",
                            "tests/test_vtree_communal.py"],
                           cwd=REPO, capture_output=True, text=True, env=env,
                           timeout=1800)
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


waited, told = 0, None
while True:
    states = [_seed_state(c) for c in WAIT_CKPTS]
    if all(s == "done" for s in states):
        break
    msg = "WAITING (%s)" % states
    if msg != told or waited % 60 == 0:
        print(f"{msg} [{waited // 60} h]", flush=True)
        told = msg
    time.sleep(60)
    waited += 1
print("nothing to wait for", flush=True)
polls, told = 0, None
while True:
    ok, why = preflight()
    if ok:
        break
    if why != told or polls % 6 == 0:
        print(f"PREFLIGHT not ready ({polls * 10} min): {why}", flush=True)
        told = why
    time.sleep(600)
    polls += 1
print(f"preflight ok (manifest + quick probe + 12 suites); starting {VARIANT} "
      f"({ARM})", flush=True)

os.makedirs(BASE, exist_ok=True)
replay_rows()
print(stop_rule_armed_line(), flush=True)

# CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
# would silently train on the CPU.
env = {**{k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
       "ABLATION_RESULTS": REL, "PYTHONFAULTHANDLER": "1"}
STALL_S = 2 * 3600    # the child logs at least every checkpoint (2 iters)
deaths = []
quick = 0
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print(f"RUN COMPLETE s{SEED} (checkpoint is a marker)", flush=True)
        sys.exit(0)
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
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, f"supv2_{attempt}.log"), "ab")
    born = time.time()
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED s{SEED} attempt {attempt} "
          f"({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
          flush=True)
    alerted = 0.0
    while p.poll() is None:
        emit_new_rows()
        try:
            idle = time.time() - max(os.path.getmtime(log.name), born)
        except OSError:
            idle = 0.0
        if idle > STALL_S and time.time() - alerted > 3600:
            print(f"ALERT: attempt {attempt} (pid {p.pid}) log silent for "
                  f"{idle / 3600:.1f} h -- possible hang", flush=True)
            alerted = time.time()
        time.sleep(30)
    log.close()
    emit_new_rows()
    rc = p.returncode
    lifetime = time.time() - born
    snapshot(attempt, rc, lifetime)
    if completed():
        print(f"RUN COMPLETE s{SEED}", flush=True)
        sys.exit(0)
    deaths.append(time.time())
    if len([t for t in deaths if time.time() - t < 3600]) >= 5:
        print("CRASH LOOP: 5 deaths within 60 min -- stopping for diagnosis",
              flush=True)
        sys.exit(1)
    if lifetime < 600:
        quick += 1
        wait = min(900, 60 * (2 ** quick))
        print(f"quick death #{quick}; backing off {wait}s", flush=True)
        time.sleep(wait)
    else:
        quick = 0
        time.sleep(15)
print("MAX ATTEMPTS REACHED", flush=True)
sys.exit(1)
