"""Supervisor for the post-audit from-scratch run: new architecture,
SEQUENTIAL 50-sim acting (batch-16 reverted -- flattened targets),
SEEDED communal chains, vl clamps, 31.25k transition buffer. The clean
matched control the 2026-08-30 re-audit said never existed. Fresh first
launch; --resume once a checkpoint exists; exit codes captured;
auto-resume with crash-loop guard; eval rows printed."""
import json
import os
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
BASE = os.path.join(REPO, "benchmarks", "results", "pong-archfix-2026-08-30")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(
    BASE, "resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt")
COMMON = [PY, "experiments/ablation.py", "--arms",
          "effzero-repro-refresh5x-communal", "--seed", "0", "--iterations",
          "90", "--train-steps", "2000", "--eval-every", "5", "--env",
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
            print("EVAL iter %s  train %s  H(pi) %.3f  maxp %.3f  freshH %s  return %.1f" % (
                r.get("iteration"), r.get("train_steps"),
                r.get("policy_entropy", float("nan")),
                r.get("policy_maxp", float("nan")),
                ("%.3f" % r["fresh_target_entropy"]
                 if isinstance(r.get("fresh_target_entropy"), float) else "n/a"),
                r.get("return_mean", float("nan"))), flush=True)
        except Exception:
            pass


def completed():
    try:
        return os.path.getsize(CKPT) < 1_000_000
    except OSError:
        return False


os.makedirs(BASE, exist_ok=True)
for line in rows():
    seen.add(line.strip())

env = {**os.environ, "ABLATION_RESULTS": "benchmarks/results/pong-archfix-2026-08-30"}
deaths = []
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print("RUN COMPLETE (checkpoint is a marker)", flush=True)
        sys.exit(0)
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, f"sup{attempt}.log"), "ab")
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED attempt {attempt} ({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
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
