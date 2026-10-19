"""P2a on Ms. Pac-Man with reward-clip growth (Kai, 2026-10-19): two arms at
SEED 0, run ONE AFTER THE OTHER (audit r1 B: two concurrent Ray runs press
host commit, and the twin trained alone), one supervisor each --
  tk: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow
      (first: Kai's question)
  rg: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow
      (waits for tk's completion marker before its preflight)
--ray --cpu-actor (pinned by the arms). Design and pre-registered reads:
CURRICULUM.md "Next experiment (pre-registered) -- P2a on Ms. Pac-Man, with
reward-clip growth".

    python benchmarks/supervisors/supervise_rgrow.py tk 0
    python benchmarks/supervisors/supervise_rgrow.py rg 0
    python benchmarks/supervisors/supervise_rgrow.py --write-manifest

Both may be launched at once (rg waits). Create each run folder before a
launch that redirects into it (`>> <folder>/supervisor.log`): the shell
opens the redirect before this file runs.

The runs execute from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits; development continues on branch
curriculum-p2 elsewhere), so REPO is this file's own tree. Supervisor v2
pattern: a relaunch auto-resumes from the last checkpoint; before EVERY
launch attempt the tree's AUDITED files must hash to the manifest written
from the audited commit."""
import json
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "rg": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow",
    "tk": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow",
}
RUNS = {"rg": "mspacman-rgrow", "tk": "mspacman-tokens-rgrow"}
# Run order: a variant starts only after every earlier one completed.
ORDER = ("tk", "rg")
MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
SEEDS = {0: "2026-10-19"}                 # the pre-registered seed and date
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# The shared interpreter lives in the main tree (worktrees have no .venv).
PY = os.environ.get("RGROW_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 40
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rgrow_manifest.json")
# Every repo module the runs execute, the probe the readout runs, the
# suites the preflight runs, and this supervisor.
SUITES = ["tests/test_rgrow.py", "tests/test_tokens.py",
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
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_rgrow.py"] + SUITES


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
    """The pre-registered recipe: the twin's (mspacman-s0-2026-10-16), the
    arm the only change."""
    return [PY, "experiments/ablation.py", "--arms", VARIANTS[variant],
            "--seed", str(seed),
            "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
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
    """The import probe for `variant`'s arm."""
    arm = VARIANTS[variant]
    tokens = 36 if variant == "tk" else None
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree; "
            f"spec = ablation.ARMS[{arm!r}]; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_UNIT'] == 10.0 and "
            "spec['REWARD_GROWTH_CEILING'] == 160.0; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            f"assert (len(spec['TOKEN_VOCAB']) if spec.get('TOKEN_VOCAB') "
            f"else None) == {tokens!r}; "
            "assert hasattr(ablation, 'reward_growth_gate') and "
            "hasattr(ablation.Run, 'apply_reward_ceiling'); "
            "assert hasattr(muzero, 'grown_reward')")


def run_done(variant, seed):
    """The run's completion marker: its checkpoint shrunk to a marker."""
    try:
        return os.path.getsize(paths(variant, seed)[3]) < 1_000_000
    except OSError:
        return False


def wait_turn(variant, seed):
    """Block until every variant before this one in ORDER has completed
    (one run on the GPU at a time). Logs a WAITING line hourly (the
    hand-off's stale-log ALERT reads the log's mtime) and ALERTs while an
    earlier run's supervisor has given up."""
    first, polls = True, 0
    while True:
        before = [u for u in ORDER[:ORDER.index(variant)]
                  if not run_done(u, seed)]
        if not before:
            if not first:
                print(f"{list(ORDER[:ORDER.index(variant)])} complete; "
                      f"{variant} s{seed} may start", flush=True)
            return
        if first or polls % 60 == 0:
            gave_up = []
            for u in before:
                try:
                    with open(os.path.join(paths(u, seed)[1],
                                           "supervisor.log"), "rb") as f:
                        lines = f.read().splitlines()
                except OSError:
                    lines = []
                fail = max((i for i, l in enumerate(lines) if l.startswith(
                    (b"CRASH LOOP", b"MAX ATTEMPTS"))), default=-1)
                live = max((i for i, l in enumerate(lines) if l.startswith(
                    (b"LAUNCHED", b"preflight ok", b"PREFLIGHT"))),
                    default=-1)
                if fail > live:
                    gave_up.append(u)
            print(f"WAITING for {before} to complete before {variant} s{seed} "
                  f"[{polls // 60} h]", flush=True)
            if gave_up:
                print(f"ALERT: the supervisor of {gave_up} gave up -- "
                      f"{variant} keeps waiting", flush=True)
        first = False
        time.sleep(60)
        polls += 1


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


def live_children(variant, seed):
    """PIDs of running ablation.py processes of this variant's arm and seed:
    an orphan of an earlier supervisor (the child is detached and survives a
    supervisor kill) or a duplicate launch. The arm is matched as a whole
    argument, so the other variant's run never counts."""
    import psutil
    arm = VARIANTS[variant]
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


class Rows:
    """Stream new jsonl rows as EVAL lines (the run's in-run eval, the
    reward-growth fields and, for tk, the token diagnostics)."""

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
            tok = ""
            if self.variant == "tk":
                # decisions per TRAJECTORY = per life on Ms. Pac-Man
                tok = (" | frames/decision %s  decisions/life %s  play share "
                       "%s  search share %s" % (
                           r.get("tok_frames_per_decision"),
                           r.get("tok_decisions_per_game"),
                           r.get("tok_duration_play_share"),
                           r.get("tok_duration_search_share")))
            print("EVAL %s s%d iter %s  train %s  env %s  return %s  H(pi) %s "
                  " erank %s | ceiling %s  kept %s  max %s%s | elapsed %s s, "
                  "%s min/iter" % (
                      self.variant, self.seed, r.get("iteration"),
                      r.get("train_steps"), r.get("env_steps"),
                      r.get("return_mean"), r.get("policy_entropy"),
                      r.get("proj_erank"), r.get("reward_ceiling"),
                      r.get("reward_kept"), r.get("reward_max"), tok, el,
                      per),
                  flush=True)


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


def main(argv):
    if "--write-manifest" in argv:
        return write_manifest()
    if (len(argv) != 2 or argv[0] not in VARIANTS or not argv[1].isdigit()
            or int(argv[1]) not in SEEDS):
        print(f"usage: supervise_rgrow.py <variant: {sorted(VARIANTS)}> "
              f"<seed: {sorted(SEEDS)}> | --write-manifest", flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    wait_turn(variant, seed)

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
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} s{seed} ({arm})", flush=True)

    os.makedirs(base, exist_ok=True)
    rows = Rows(jsonl, variant, seed)
    rows.prime()
    # CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
    # would silently train on the CPU.
    env = {**{k: v for k, v in os.environ.items()
              if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": rel, "PYTHONFAULTHANDLER": "1"}
    stall_s = 2 * 3600    # the child logs at least every checkpoint (2 iters)
    deaths, quick_deaths = [], 0

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
        polls, told = 0, None
        while True:
            ok, why = manifest_check()
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the tree is the "
                      f"audited revision [{polls} min]", flush=True)
                told = why
            time.sleep(60)
            polls += 1
        polls, told = 0, None
        while True:
            pids = [p for v in VARIANTS for p in live_children(v, seed)]
            if not pids:
                break
            msg = (f"ALERT: an ablation.py of this experiment at seed {seed} "
                   f"is already running (pids {pids}) -- one run on the GPU "
                   f"at a time, and never a second writer")
            if msg != told or polls % 60 == 0:
                print(f"{msg} [{polls} min]", flush=True)
                told = msg
            time.sleep(60)
            polls += 1
        args = command(variant, seed) + (["--resume"] if os.path.exists(ckpt)
                                         else [])
        log = open(os.path.join(base, f"supv2_{attempt}.log"), "ab")
        born = time.time()
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
