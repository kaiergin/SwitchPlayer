"""Supervisor for the communal3 extension (+50 iters from communal2's
completed weights). Attempt 1 warm-starts with --init-weights; every
relaunch after a checkpoint exists resumes (the two flags are mutually
exclusive by harness design). Prints eval rows and trainer exit codes;
auto-resumes; stops on completion or crash-loop."""
import json
import os
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
BASE = os.path.join(REPO, "benchmarks", "results", "pong-communal3-2026-08-29")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(
    BASE, "resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt")
PARENT = os.path.join(REPO, "benchmarks", "results", "pong-communal2-2026-08-27",
                      "weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth")
COMMON = [PY, "experiments/ablation.py", "--arms",
          "effzero-repro-refresh5x-communal", "--seed", "0", "--iterations",
          "50", "--train-steps", "2000", "--eval-every", "5", "--env",
          "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2"]
MAX_ATTEMPTS = 15

seen = set()


def rows():
    try:
        with open(JSONL, "rb") as f:
            return [l for l in f.read().decode("utf-8", "replace").splitlines() if l.strip()]
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
            print("EVAL iter %s  train %s  H(pi) %.3f  maxp %.3f  return %.1f" % (
                r.get("iteration"), r.get("train_steps"),
                r.get("policy_entropy", float("nan")),
                r.get("policy_maxp", float("nan")),
                r.get("return_mean", float("nan"))), flush=True)
        except Exception:
            pass


def completed():
    try:
        return os.path.getsize(CKPT) < 1_000_000
    except OSError:
        return False


if not os.path.exists(PARENT):
    print(f"PARENT WEIGHTS MISSING: {PARENT}", flush=True)
    sys.exit(1)
os.makedirs(BASE, exist_ok=True)
for line in rows():
    seen.add(line.strip())

env = {**os.environ, "ABLATION_RESULTS": "benchmarks/results/pong-communal3-2026-08-29"}
deaths = []
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print("RUN COMPLETE (checkpoint is a marker)", flush=True)
        sys.exit(0)
    if os.path.exists(CKPT):
        args = COMMON + ["--resume"]
    else:
        args = COMMON + ["--init-weights",
                         "benchmarks/results/pong-communal2-2026-08-27/"
                         "weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"]
    log = open(os.path.join(BASE, f"sup{attempt}.log"), "ab")
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED attempt {attempt} ({'resume' if '--resume' in args else 'warm start'}), pid {p.pid}",
          flush=True)
    while p.poll() is None:
        emit_new_rows()
        time.sleep(30)
    log.close()
    emit_new_rows()
    rc = p.returncode
    print(f"TRAINER EXITED rc={rc} (0x{rc & 0xFFFFFFFF:08X}) attempt {attempt}",
          flush=True)
    if completed():
        print("RUN COMPLETE", flush=True)
        sys.exit(0)
    deaths.append(time.time())
    if len([t for t in deaths if time.time() - t < 900]) >= 3:
        print("CRASH LOOP: 3 deaths within 15 min -- stopping for diagnosis",
              flush=True)
        sys.exit(1)
    time.sleep(15)
print("MAX ATTEMPTS REACHED", flush=True)
sys.exit(1)
