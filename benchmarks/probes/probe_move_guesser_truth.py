"""Test H, STAGE B: the truth (CURRICULUM "Next experiment (pre-registered) --
Test H", METHOD: TRUTH, COUNTERFACTUAL ITEMS, REACTION-FREE ITEMS). Scoring
only: nothing here reaches training.

At every held-out decision t of a game written by stage A (O0's games 8-9,
all of OF's), from the recorded emulator state and MaxAndSkip buffer, through
the training wrapper stack:
  - each of the 9 moves plays one base step; the played move's step must equal
    the record (frame, reward, raw digests): the one-step check;
  - moves leaving identical (emulator state, MaxAndSkip buffer, the step's raw
    digests) share everything after (exact merge);
  - each distinct post-move state continues W - 1 = 15 base steps two ways:
    CRN, under the snapshot's own acting rule at the game's uniforms for
    decision indices t+1 .. t+15 (what this player would have done after that
    first move), and NOOP (no player's later move at all); a branch stops
    where its life ends (a lost life or the game's end) or reaches the life
    cap, its later offsets then repeating its last stack (LIFE ENDS);
  - the played move's CRN continuation must reproduce the recorded line: the
    same moves, pooled frames and raw digests, byte for byte; any failure is a
    TRUTH FAILURE (exit 2), to be fixed before any training;
  - classes: moves are equivalent (primary) when their pooled observations at
    t+1 .. t+16 are identical, i.e. the same frames after steps t .. t+15 and
    the same end; raw-frame classes and end-state equality are kept beside,
    for CRN and NOOP alike.
Written per game: the classes, per distinct state its 16 per-step frame
digests and end for both continuations (read 4's per-k partitions), and the
future frames each scored item needs (every CRN class but the played one, and
every NOOP class, at contested decisions).

usage (from the run tree, under the lock, CPU):
  python benchmarks/probes/probe_move_guesser_truth.py [--data DIR] [--out DIR]
      [--games NAME,NAME] [--max-decisions N]
The optional --games / --max-decisions make a SMOKE run (written under
"truth-smoke"). A game already written with this source's stamp is skipped.
CPU, stage A's THREADS threads and batch-1 forwards, so the CRN continuations
repeat stage A's forwards byte for byte; the env settings happen only when it
runs as a script."""
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = "MsPacmanNoFrameskip-v4"
    os.environ["PROBE_THREADS"] = "2"
import argparse
import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ale_py
import numpy as np
import torch
import probe_move_guesser as mg
import muzero

W = 16                                      # base steps in the window, t .. t+15
SOURCES = mg.SOURCES + (pathlib.Path(__file__).resolve(),)
SOURCE = mg.source_hash(SOURCES)
END_NONE, END_LIFE, END_CAP = 0, 1, 2


def _d(x):
    return hashlib.blake2b(np.ascontiguousarray(x).tobytes(),
                           digest_size=mg.DIGEST).digest()


# --------------------------------------------------------------------------
# restore and step
# --------------------------------------------------------------------------
def restore(env, skip, blob, buf):
    """A saved state onto the training wrappers: the ALE state, MaxAndSkip's
    buffer (Python-side, outside the ALE state) and EpisodicLifeEnv's own
    life count, which restoreState does not touch (Test G's pin)."""
    ale = env.unwrapped.ale
    ale.restoreState(ale_py.ALEState(blob))
    skip._buffer[:] = buf
    env.lives = ale.lives()
    env.was_real_done = False


def snapshot_state(env, skip):
    return (env.unwrapped.ale.cloneState().serialize(), skip._buffer.copy())


def step(env, tap, move):
    """One base step: (pooled frame, reward, ended, raw digests)."""
    tap.take()
    obs, reward, term, trunc, _ = env.step(int(move))
    return mg.frame_of(obs), float(reward), bool(term or trunc), tap.take()


def continuation(env, tap, skip, start, stack, frame, steps_in_life, act, first_index,
                 n=W - 1, cap=mg.LIFE_CAP):
    """From a post-move state `start` (blob, buffer) whose step gave `frame`,
    with `stack` the decision's stack: n more base steps, each move from
    act(stack, decision_index) (None = NOOP). Stops where the life ends or
    reaches the cap. Returns (frames, moves, rewards, digests per step, end,
    end-state blob)."""
    restore(env, skip, *start)
    frames, moves, rewards, digests = [], [], [], []
    stack = muzero.push_frame(stack, frame)
    end = END_NONE
    steps = steps_in_life
    for j in range(n):
        if steps >= cap:
            end = END_CAP
            break
        m = 0 if act is None else act(stack, first_index + j)
        f, r, ended, dg = step(env, tap, m)
        frames.append(f)
        moves.append(m)
        rewards.append(r)
        digests.append(dg)
        steps += 1
        if ended:
            end = END_LIFE
            break
        stack = muzero.push_frame(stack, f)
    return frames, moves, rewards, digests, end, env.unwrapped.ale.cloneState().serialize()


# --------------------------------------------------------------------------
# classes
# --------------------------------------------------------------------------
def future_key(first_frame, first_end, frames, end):
    """What the guessers can see of a branch: the pooled frames after steps
    t .. t+15 as far as the life lasts, and where it ended (LIFE ENDS: later
    offsets repeat the last stack, so frames plus the end index fix them)."""
    fs = [first_frame] + list(frames)
    e = END_LIFE if first_end else end
    return (tuple(_d(f) for f in fs), len(fs), e)


def raw_key(first_digests, first_end, digests, end):
    ds = [tuple(first_digests)] + [tuple(d) for d in digests]
    return (tuple(ds), len(ds), END_LIFE if first_end else end)


def partition(keys):
    """Class ids by first appearance: equal keys share an id."""
    ids = {}
    return [ids.setdefault(k, len(ids)) for k in keys]


def step_digests(first_frame, frames, length=W):
    """The per-step frame digests t .. t+15, padded past the life's end with
    the last one (LIFE ENDS repeats the last stack): read 4's per-k shares."""
    fs = [first_frame] + list(frames)
    ds = [_d(f) for f in fs]
    ds += [ds[-1]] * (length - len(ds))
    return ds[:length]


# --------------------------------------------------------------------------
# one decision
# --------------------------------------------------------------------------
def decision_truth(env, tap, skip, state, buf, stack, steps_in_life, act, d,
                   record, cap=mg.LIFE_CAP):
    """All of one held-out decision's truth. `record` is the recorded line from
    t: dict(move, frame, reward, digests, then lists over t+1 .. for moves,
    frames, rewards, digests, and end) as stage A wrote it. Returns (result,
    failures)."""
    failures = []
    # one step per move
    first = []
    for m in range(mg.MOVES):
        restore(env, skip, state, buf)
        f, r, ended, dg = step(env, tap, m)
        post = snapshot_state(env, skip)
        first.append({"frame": f, "reward": r, "ended": ended, "digests": dg,
                      "post": post,
                      "key": (post[0], post[1].tobytes(), tuple(dg))})
    pm = int(record["move"])
    p1 = first[pm]
    if not (np.array_equal(p1["frame"], record["frame"])
            and p1["reward"] == record["reward"]
            and list(p1["digests"]) == list(record["step_digests"])
            and p1["ended"] == record["ended"]):
        failures.append(f"decision {d}: the played move's step differs from the record")
    groups = partition([x["key"] for x in first])
    n_groups = max(groups) + 1
    rep = [groups.index(g) for g in range(n_groups)]       # first move of each group
    out = {"groups": groups, "crn": [], "noop": []}
    for g, m in enumerate(rep):
        x = first[m]
        for kind, a in (("crn", act), ("noop", None)):
            if x["ended"]:
                c = ([], [], [], [], END_LIFE, x["post"][0])
            else:
                c = continuation(env, tap, skip, x["post"], stack, x["frame"],
                                 steps_in_life + 1, a, d + 1, cap=cap)
            frames, moves, rewards, digests, end, end_state = c
            out[kind].append({
                "frames": [x["frame"]] + frames, "moves": moves,
                "end": END_LIFE if x["ended"] else end,
                "fkey": future_key(x["frame"], x["ended"], frames, end),
                "rkey": raw_key(x["digests"], x["ended"], digests, end),
                "skey": _d(np.frombuffer(end_state, np.uint8)),
                "steps": step_digests(x["frame"], frames),
                "digests": [x["digests"]] + digests})
    # the played line reproduced
    pg = groups[pm]
    crn = out["crn"][pg]
    n = len(crn["moves"])
    rec_n = len(record["moves"])
    if (n != rec_n or crn["moves"] != list(record["moves"])
            or not all(np.array_equal(a, b) for a, b in
                       zip(crn["frames"][1:], record["frames"]))
            or [list(x) for x in crn["digests"][1:]] !=
            [list(x) for x in record["digests"]]
            or crn["end"] != record["end"]):
        failures.append(f"decision {d}: the played move's continuation differs "
                        f"from the recorded line (branch {n} steps, end "
                        f"{crn['end']}; record {rec_n} steps, end {record['end']})")
    by_move = {
        "crn": partition([out["crn"][groups[m]]["fkey"] for m in range(mg.MOVES)]),
        "crn_raw": partition([out["crn"][groups[m]]["rkey"] for m in range(mg.MOVES)]),
        "crn_strict": partition([(out["crn"][groups[m]]["rkey"],
                                  out["crn"][groups[m]]["skey"])
                                 for m in range(mg.MOVES)]),
        "noop": partition([out["noop"][groups[m]]["fkey"] for m in range(mg.MOVES)]),
        "noop_raw": partition([out["noop"][groups[m]]["rkey"] for m in range(mg.MOVES)]),
    }
    out["classes"] = by_move
    out["played"] = pm
    out["contested"] = len(set(by_move["crn"])) > 1
    return out, failures


def recorded_window(arr, i, n=W - 1):
    """The recorded line from event i (decision t's step): its move, frame,
    reward, raw digests (step_digests) and end, then up to n later steps of
    the same life as lists (moves, frames, rewards, digests), and their end:
    END_LIFE where the life ended, END_CAP where it was capped, END_NONE when
    the window ran out first."""
    def dg(k):
        return [bytes(x) for x in arr["digests"][k][:int(arr["ndigests"][k])]]

    rec = {"move": int(arr["move"][i]), "frame": arr["frames"][i],
           "reward": float(arr["reward"][i]), "step_digests": dg(i),
           "ended": int(arr["end"][i]) == 1,
           "moves": [], "frames": [], "rewards": [], "digests": []}
    end = END_LIFE if int(arr["end"][i]) == 1 else (END_CAP if int(arr["end"][i]) == 2
                                                      else END_NONE)
    k = i + 1
    while end == END_NONE and len(rec["moves"]) < n and k < len(arr["kind"]):
        if int(arr["kind"][k]) == mg.KIND_START:
            break
        rec["moves"].append(int(arr["move"][k]))
        rec["frames"].append(arr["frames"][k])
        rec["rewards"].append(float(arr["reward"][k]))
        rec["digests"].append(dg(k))
        e = int(arr["end"][k])
        if e == 1:
            end = END_LIFE
        elif e == 2:
            end = END_CAP
        k += 1
    if end == END_NONE and len(rec["moves"]) < n and not rec["ended"] \
            and int(arr["end"][i]) == 0:
        # every life's last event carries its end flag (stage A), so a window
        # that runs out without one is a malformed record
        raise ValueError(f"event {i}: the record's life ends without an end flag")
    rec["end"] = END_LIFE if rec["ended"] else end
    return rec


def steps_in_life_before(arr, i):
    """Base steps already played in event i's life before it."""
    k, n = i - 1, 0
    while k >= 0 and int(arr["kind"][k]) != mg.KIND_START:
        n += 1
        k -= 1
    return n


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------
def acting(run, key, t):
    """act(stack, decision index): the snapshot's acting rule at the game's
    uniforms (common random numbers with stage A's record)."""
    manager = run.searcher.network_manager

    def act(stack, index):
        policy, _ = manager.initial_inference(stack)
        return mg.sample(mg.tempered(policy, t), mg.uniform(key, index))

    return act


def load_game(data, name):
    meta = json.loads((data / f"{name}.json").read_text())
    with np.load(data / f"{name}.npz") as z:
        arr = {k: z[k] for k in z.files}
    with np.load(data / f"{name}-states.npz") as z:
        st = {k: z[k] for k in z.files}
    blob, offs = st["blob"].tobytes(), st["offsets"]
    states = {int(d): (blob[offs[i]:offs[i + 1]], st["buffer"][i])
              for i, d in enumerate(st["decision"])}
    return meta, arr, states


def game_truth(env, tap, skip, arr, states, act, max_decisions=None,
               cap=mg.LIFE_CAP):
    """Every held-out decision of one game. Returns (per-decision results,
    failures)."""
    stacks = mg.stacks_from_frames(arr)
    index = {int(arr["decision"][i]): i for i in range(len(arr["kind"]))
             if int(arr["kind"][i]) == mg.KIND_MOVE}
    results, failures = [], []
    for n, d in enumerate(sorted(index)):
        if max_decisions is not None and n >= max_decisions:
            break
        if d not in states:
            failures.append(f"decision {d}: no saved state")
            continue
        i = index[d]
        blob, buf = states[d]
        res, fail = decision_truth(env, tap, skip, blob, buf, stacks[d],
                                   steps_in_life_before(arr, i), act, d,
                                   recorded_window(arr, i), cap=cap)
        res["decision"] = d
        results.append(res)
        failures += fail
    return results, failures


def pack(results):
    """Arrays for the file: per decision the classes, per move its future's
    per-step digests, end and length for both continuations, and the future
    blocks each scored item needs (contested decisions: every CRN class but the
    played one, every NOOP class), frames padded to W."""
    n = len(results)
    out = {"decision": np.array([r["decision"] for r in results], np.int32),
           "played": np.array([r["played"] for r in results], np.int8),
           "contested": np.array([r["contested"] for r in results], bool),
           "n_states": np.array([max(r["groups"]) + 1 for r in results], np.int8)}
    for k in ("crn", "crn_raw", "crn_strict", "noop", "noop_raw"):
        out[f"class_{k}"] = np.array([r["classes"][k] for r in results],
                                     np.int8).reshape(n, mg.MOVES)
    for kind in ("crn", "noop"):
        steps = np.zeros((n, mg.MOVES, W, mg.DIGEST), np.uint8)
        end = np.zeros((n, mg.MOVES), np.int8)
        length = np.zeros((n, mg.MOVES), np.int8)
        for a, r in enumerate(results):
            for m in range(mg.MOVES):
                b = r[kind][r["groups"][m]]
                for s, dg in enumerate(b["steps"]):
                    steps[a, m, s] = np.frombuffer(dg, np.uint8)
                end[a, m] = b["end"]
                length[a, m] = len(b["frames"])
        out[f"{kind}_steps"], out[f"{kind}_end"], out[f"{kind}_len"] = steps, end, length
    blk = {"dec": [], "kind": [], "cls": [], "move": [], "len": [], "end": [],
           "frames": []}
    for a, r in enumerate(results):
        if not r["contested"]:
            continue
        for kind, code in (("crn", 0), ("noop", 1)):
            cls = r["classes"][kind]
            played_cls = cls[r["played"]]
            for c in sorted(set(cls)):
                if kind == "crn" and c == played_cls:
                    continue               # the recorded line, in stage A's file
                m = cls.index(c)
                b = r[kind][r["groups"][m]]
                fr = np.zeros((W, 3, 84, 84), np.uint8)
                fr[:len(b["frames"])] = np.stack(b["frames"])
                blk["dec"].append(a)
                blk["kind"].append(code)
                blk["cls"].append(c)
                blk["move"].append(m)
                blk["len"].append(len(b["frames"]))
                blk["end"].append(b["end"])
                blk["frames"].append(fr)
    out["blk_dec"] = np.array(blk["dec"], np.int32)
    out["blk_kind"] = np.array(blk["kind"], np.int8)
    out["blk_class"] = np.array(blk["cls"], np.int8)
    out["blk_move"] = np.array(blk["move"], np.int8)
    out["blk_len"] = np.array(blk["len"], np.int8)
    out["blk_end"] = np.array(blk["end"], np.int8)
    out["blk_frames"] = (np.stack(blk["frames"]) if blk["frames"]
                         else np.zeros((0, W, 3, 84, 84), np.uint8))
    return out


def held_games():
    names = []
    for p, spec in mg.RUNS.items():
        for it in spec["iterations"]:
            for g in spec["held"]:
                names.append((p, it, g, mg.game_name(p, it, g)))
    return names


def written(out, name):
    meta = out / f"{name}-truth.json"
    if not meta.exists() or not (out / f"{name}-truth.npz").exists():
        return None
    m = json.loads(meta.read_text())
    return m if m.get("source") == SOURCE else None


def stage_truth(data, out, only=None, max_decisions=None):
    out.mkdir(parents=True, exist_ok=True)
    mg.commit_gate()
    mg.pin_process()
    div = json.loads((data / "diversity.json").read_text()) \
        if (data / "diversity.json").exists() else None
    if only is None and not (div and div.get("passed") and div.get("source") == mg.SOURCE):
        raise SystemExit("stage A's diversity check has not passed for this data")
    games = held_games()
    if only is not None:
        games = [g for g in games if g[3] in only]
    by_snapshot = {}
    for p, it, g, name in games:
        by_snapshot.setdefault((p, it), []).append((g, name))
    total_fail = []
    for (p, it), members in by_snapshot.items():
        todo = [(g, name) for g, name in members if not written(out, name)]
        if not todo:
            continue
        table = mg.rows(p)
        path = mg.snapshot_path(p, it)
        env_steps = int(table[it]["env_steps"])
        with mg.player(path, env_steps) as (run, d):
            t = float(muzero.prior_acting_temperature(
                run.searcher.network_manager.training_steps))
            mg.temperature_gate(p, it, table, t)
            mg.check_threads()
            env, tap, noop, skip = mg.make_env()
            env.reset(seed=0)
            try:
                for g, name in todo:
                    mg.commit_gate()
                    meta, arr, states = load_game(data, name)
                    if meta.get("source") != mg.SOURCE or meta.get("threads") != mg.THREADS:
                        raise SystemExit(f"{name}: written by another stage-A source "
                                         f"or thread count")
                    if abs(meta["temperature"] - t) > 1e-12:
                        raise SystemExit(f"{name}: recorded T {meta['temperature']} "
                                         f"against {t}")
                    t0 = time.time()
                    results, failures = game_truth(
                        env, tap, skip, arr, states, acting(run, meta["key"], t),
                        max_decisions=max_decisions, cap=meta["life_cap"])
                    packed = pack(results)
                    tmp = out / f"{name}-truth.npz.tmp"
                    with open(tmp, "wb") as f:
                        np.savez_compressed(f, **packed)
                    os.replace(tmp, out / f"{name}-truth.npz")
                    info = {"name": name, "decisions": len(results),
                            "contested": int(packed["contested"].sum()),
                            "states_mean": float(packed["n_states"].mean())
                            if len(results) else None,
                            "blocks": int(len(packed["blk_dec"])),
                            "failures": failures, "source": SOURCE,
                            "data_source": meta["source"],
                            "max_decisions": max_decisions,
                            "secs": round(time.time() - t0, 1)}
                    tmp = out / f"{name}-truth.json.tmp"
                    tmp.write_text(json.dumps(info, indent=1))
                    os.replace(tmp, out / f"{name}-truth.json")
                    print(f"  {name}: {info['decisions']} decisions, "
                          f"{info['contested']} contested, states/decision "
                          f"{info['states_mean']}, {info['blocks']} blocks, "
                          f"{len(failures)} failures [{info['secs']}s]", flush=True)
                    total_fail += [f"{name}: {x}" for x in failures]
            finally:
                env.close()
    if total_fail:
        print("TRUTH FAILURE:", *total_fail[:20], sep="\n  ", flush=True)
        raise SystemExit(2)
    return games


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--games")
    ap.add_argument("--max-decisions", type=int)
    a = ap.parse_args(argv)
    smoke = bool(a.games or a.max_decisions)
    base = mg.ROOT / "benchmarks" / "probes" / "testh"
    data = pathlib.Path(a.data) if a.data else base / "data"
    out = pathlib.Path(a.out) if a.out else base / ("truth-smoke" if smoke else "truth")
    only = set(a.games.split(",")) if a.games else None
    stage_truth(data, out, only, a.max_decisions)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
