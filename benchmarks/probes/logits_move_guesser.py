"""Test H, STAGE D1: every model's logits on every scored item (CURRICULUM "Next
experiment (pre-registered) -- Test H", READS), plus the GPU-side reads: the
near-duplicate search (read 1) and O0's acting argmax on OF's held-out states
(read 5). Stage D2 (score_move_guesser.py) turns these into the reads and the
verdict; nothing here decides anything.

Items, per held-out game (O0's games 8-9, all of OF's), from stage A's record
and stage B's truth:
  - OWN: every held-out decision, obs_t and the recorded future (read 2);
  - CRN (the counterfactual items, read 1): at each contested decision one item
    per CRN class, the played class's future the recorded line, every other
    class's its stage-B block;
  - NOOP (the reaction-free items): one item per NOOP class, every future a
    stage-B block, the played class's included.
A block's future: obs_{t+k} is obs_t with the block's first k frames pushed,
the last stack repeated past the block's end (LIFE ENDS). Every item keeps its
decision, class, member moves and weight (class size / 9).
Models: plan.json's read models (arm-seed, the best point within the model's
cap), the logits of each on each item; for SIGMA also its cut profile (weights
renormalised over offsets j <= k, k = 1 .. 16) on the CRN and NOOP items, and
its learned offset weights. Games are processed one at a time (their blocks
are about 15 GB in all).
Near-duplicates: for each held-out decision, the fewest differing pixels (of
84 x 84; a pixel differs when any channel does) between the newest frame of its
obs_t and every frame of the FIT games and of the other held-out games, exact,
on the GPU in chunks. Also every decision's obs_t stack hash and whether it
occurs in FIT or in another held-out game (the READS exclusion), and whether a
whole item (obs_t and its 16 future stacks) occurs in FIT (read 7).

usage (from the run tree, under the lock, GPU):
  python benchmarks/probes/logits_move_guesser.py [--data DIR] [--truth DIR]
      [--train DIR] [--out DIR] [--players O0,OF] [--games NAME,NAME]"""
import os
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # before any import (see train_move_guesser)
import argparse
import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np
import torch
import probe_move_guesser as mg
import train_move_guesser as tg
import testh_windows as tw

W = 16
SOURCES = (pathlib.Path(__file__).resolve(), HERE / "testh_windows.py")
SOURCE = mg.source_hash(SOURCES)
SET_OWN, SET_CRN, SET_NOOP = 0, 1, 2
CUTS = tuple(range(1, tg.K_MAX + 1))


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------
def block_window(row0, first_index, length):
    """obs_t's 4 frame indices, then the block's frames at first_index ..
    first_index + length - 1 pushed one by one; past the block, the last stack
    repeated (LIFE ENDS). (K_MAX + 1, 4)."""
    rows = [list(row0)]
    stack = list(row0)
    for k in range(1, tg.K_MAX + 1):
        if k <= length:
            stack = stack[1:] + [first_index + k - 1]
        rows.append(list(stack))
    return rows


def game_items(arr, truth, frame_offset=0):
    """(frames, items) for one game: frames = stage A's frames then every
    block's frames; items = dict of arrays, one row per item."""
    d, ev, mv, win = tg.decision_windows(arr, frame_offset)
    by_dec = {int(x): n for n, x in enumerate(d)}
    n_a = len(arr["kind"])
    blk_frames = truth["blk_frames"]
    nb = len(blk_frames)
    frames = np.concatenate([arr["frames"], blk_frames.reshape(nb * W, 3, 84, 84)]) \
        if nb else arr["frames"]
    blk_index = {(int(truth["blk_dec"][b]), int(truth["blk_kind"][b]),
                  int(truth["blk_class"][b])): b for b in range(nb)}
    rows = {k: [] for k in ("set", "decision", "cls", "members", "weight", "played",
                            "window", "length", "end", "a")}

    def add(s, dec, cls, members, played, window, length, end, a):
        rows["set"].append(s)
        rows["decision"].append(dec)
        rows["cls"].append(cls)
        rows["members"].append(members)
        rows["weight"].append(sum(members) / mg.MOVES)
        rows["played"].append(played)
        rows["window"].append(window)
        rows["length"].append(length)
        rows["end"].append(end)
        rows["a"].append(a)

    for a, dec in enumerate(truth["decision"]):
        dec = int(dec)
        n = by_dec[dec]
        pm = int(truth["played"][a])
        rec_win = win[n]
        crn = [int(x) for x in truth["class_crn"][a]]
        add(SET_OWN, dec, crn[pm], [c == crn[pm] for c in crn], True, rec_win,
            int(truth["crn_len"][a, pm]), int(truth["crn_end"][a, pm]), a)
        if not truth["contested"][a]:
            continue
        for kind, s, cls in ((0, SET_CRN, crn),
                             (1, SET_NOOP, [int(x) for x in truth["class_noop"][a]])):
            for c in sorted(set(cls)):
                members = [x == c for x in cls]
                played = cls[pm] == c
                if kind == 0 and played:
                    add(s, dec, c, members, True, rec_win,
                        int(truth["crn_len"][a, pm]), int(truth["crn_end"][a, pm]), a)
                    continue
                b = blk_index[(a, kind, c)]
                first = frame_offset + n_a + b * W
                add(s, dec, c, members, played,
                    block_window(rec_win[0], first, int(truth["blk_len"][b])),
                    int(truth["blk_len"][b]), int(truth["blk_end"][b]), a)
    items = {"set": np.array(rows["set"], np.int8),
             "decision": np.array(rows["decision"], np.int32),
             "cls": np.array(rows["cls"], np.int8),
             "members": np.array(rows["members"], bool).reshape(-1, mg.MOVES),
             "weight": np.array(rows["weight"], np.float64),
             "played": np.array(rows["played"], bool),
             "window": np.array(rows["window"], np.int64).reshape(-1, tg.K_MAX + 1, 4),
             "length": np.array(rows["length"], np.int8),
             "end": np.array(rows["end"], np.int8),
             "a": np.array(rows["a"], np.int32)}
    return frames, items


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def load_models(train, device, smoke=False):
    """plan.json's read models; each must carry a completion stamp from this
    trainer and this stage-A data (a smoke's plan from older code is let
    through, flagged in logits.json)."""
    plan = json.loads((train / "plan.json").read_text())
    if plan.get("source") != tg.SOURCE and not smoke:
        raise SystemExit("plan.json is from other trainer code")
    models = {}
    for key, m in plan["models"].items():
        arm = key.split("-s")[0]
        mdir = train / "models" / m["dir"]
        stamp = json.loads((mdir / "stamp.json").read_text())
        if not smoke and (stamp.get("source") != tg.SOURCE
                          or stamp.get("data_source") != mg.SOURCE):
            raise SystemExit(f"{key}: model from other trainer code or data")
        state, step = tg.best_state(mdir, m["cap"])
        if step != m["best_step"]:
            raise SystemExit(f"{key}: best step {step} against plan's {m['best_step']}")
        net = tg.Guesser(tg.ARMS[arm]).to(device)
        net.load_state_dict(state)
        net.eval()
        models[key] = net
        if "final_state" in m:
            # read (1)'s final-step model: the state at the model's used cap
            fin = tg.Guesser(tg.ARMS[arm]).to(device)
            fin.load_state_dict(torch.load(mdir / m["final_state"], map_location="cpu",
                                           weights_only=True))
            fin.eval()
            models[key + "@final"] = fin
    return plan, models


def sigma_features(net, x):
    """SIGMA's per-pair 512-vectors on gather()'s layout, (B, 16, 512) float."""
    b = x.shape[0]
    o0 = x[:, 0]
    rows = x[:, 1:]
    p = rows.shape[1]
    pairs = torch.cat([o0.unsqueeze(1).expand(-1, p, -1, -1, -1), rows], dim=2)
    z = pairs.flatten(0, 1)
    z = z.to(torch.float16 if z.is_cuda else torch.float32) / 255.0
    return net.enc(z).view(b, p, -1).float()


def cut_logits(net, z):
    """Logits under every cut j <= k, (len(CUTS), B, 9)."""
    out = []
    for k in CUTS:
        w = net.w.float()[:k]
        a = torch.softmax(w, 0)
        h = (z[:, :k] * a[None, :, None]).sum(1)
        out.append(net.head(h))
    return torch.stack(out)


def model_logits(nets, frames_t, windows, sets, device, chunk_pairs=tg.EVAL_PAIRS):
    """{key: (n, 9) float32}, and for SIGMA keys {key + '/cut': (16, n, 9)} on
    the CRN and NOOP items (zeros elsewhere)."""
    out = {}
    win_t = torch.from_numpy(windows).to(device)
    n = len(windows)
    for key, net in nets.items():
        p = len(net.offsets)
        chunk = max(1, chunk_pairs // p)
        lg = np.zeros((n, mg.MOVES), np.float32)
        cuts = np.zeros((len(CUTS), n, mg.MOVES), np.float32) if key.startswith("SIGMA") else None
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            for s in range(0, n, chunk):
                x = tg.gather(frames_t, win_t[s:s + chunk], net.offsets)
                if cuts is not None:
                    z = sigma_features(net, x)
                    a = torch.softmax(net.w.float(), 0)
                    lg[s:s + chunk] = net.head((z * a[None, :, None]).sum(1)).float().cpu().numpy()
                    c = cut_logits(net, z).float().cpu().numpy()
                    m = sets[s:s + chunk] != SET_OWN
                    cuts[:, s:s + chunk][:, m] = c[:, m]
                else:
                    lg[s:s + chunk] = net(x).float().cpu().numpy()
        out[key] = lg
        if cuts is not None:
            out[key + "/cut"] = cuts
    return out


# --------------------------------------------------------------------------
# hashes and near-duplicates
# --------------------------------------------------------------------------
stack_hash = tw.stack_digest                # the exclusion's key, one definition
D1_COMMIT_GB = 14.0


def packed(frames):
    """(n, 7056) int32: a pixel's three channels in one integer, so a pixel
    differs exactly when its packed value does."""
    f = frames.astype(np.int32)
    return (f[:, 0] << 16 | f[:, 1] << 8 | f[:, 2]).reshape(len(frames), -1)


def min_pixel_diff(queries, refs, device, q_chunk=32, r_chunk=2048, exclude=None):
    """For each query frame, the fewest differing pixels against any ref frame
    (exact). exclude[i]: (lo, hi) of refs to skip for query i (its own game)."""
    q = torch.from_numpy(packed(queries)).to(device)
    best = torch.full((len(queries),), 10 ** 9, dtype=torch.int64, device=device)
    for r0 in range(0, len(refs), r_chunk):
        r = torch.from_numpy(packed(refs[r0:r0 + r_chunk])).to(device)
        for q0 in range(0, len(q), q_chunk):
            qq = q[q0:q0 + q_chunk]
            diff = (qq[:, None, :] != r[None, :, :]).sum(-1)          # (qc, rc)
            if exclude is not None:
                for i in range(len(qq)):
                    lo, hi = exclude[q0 + i]
                    a, b = max(lo - r0, 0), min(hi - r0, len(r))
                    if a < b:
                        diff[i, a:b] = 10 ** 9
            best[q0:q0 + q_chunk] = torch.minimum(best[q0:q0 + q_chunk], diff.min(1).values)
    return best.cpu().numpy()


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------
def held_names(player, only=None, iterations=None):
    spec = mg.RUNS[player]
    its = [it for it in spec["iterations"] if iterations is None or it in iterations]
    names = [mg.game_name(player, it, g) for it in its for g in spec["held"]]
    return [n for n in names if only is None or n in only]


def fit_names(iterations=None):
    spec = mg.RUNS["O0"]
    its = [it for it in spec["iterations"] if iterations is None or it in iterations]
    return [mg.game_name("O0", it, g) for it in its for g in mg.SPLIT["fit"]]


def load_stage_a(data, name):
    meta = json.loads((data / f"{name}.json").read_text())
    if meta.get("source") != mg.SOURCE:
        raise SystemExit(f"{name}: not this stage A's data")
    with np.load(data / f"{name}.npz") as z:
        arr = {k: z[k] for k in z.files}
    return meta, arr


def load_truth(truth, name, smoke=False):
    """A game's truth: this source, failure-free, and whole (no max_decisions)
    unless this is a smoke."""
    import probe_move_guesser_truth as tr
    info = json.loads((truth / f"{name}-truth.json").read_text())
    if info.get("source") != tr.SOURCE or info["failures"]:
        raise SystemExit(f"{name}: truth from another source, or with failures")
    if info.get("max_decisions") is not None and not smoke:
        raise SystemExit(f"{name}: a smoke's partial truth")
    with np.load(truth / f"{name}-truth.npz") as z:
        return info, {k: z[k] for k in z.files}


def fit_hashes(data, iterations=None):
    """Every FIT window stack's hash, and every FIT window's whole-item key
    (obs_t and its 16 future stacks), plus the FIT frames for the
    near-duplicate search (stage B's floors use the same window_hashes)."""
    stacks, items, frames = set(), set(), []
    for name in fit_names(iterations):
        _, arr = load_stage_a(data, name)
        _, hs = tw.window_hashes(arr)
        for row in hs:
            stacks.update(row)
            items.add(b"".join(row))
        frames.append(arr["frames"])
    return stacks, items, np.concatenate(frames)


def stage_logits(data, truth, train, out, players=("O0", "OF"), only=None,
                 iterations=None, smoke=False):
    import probe_move_guesser_truth as tr
    out.mkdir(parents=True, exist_ok=True)
    # a run killed inside its write loop must not leave new per-game files
    # under the old summary: the summary goes first, comes back last
    (out / "logits.json").unlink(missing_ok=True)
    commit = mg.free_commit_gb() if sys.platform == "win32" else float("inf")
    if commit < D1_COMMIT_GB:
        raise SystemExit(f"free commit {commit:.1f} GB < {D1_COMMIT_GB} GB")
    if tr.of_void(truth):
        players = tuple(p for p in players if p != "OF")   # (5) VOID, recorded
    device = torch.device("cuda")
    plan, nets = load_models(train, device, smoke)
    fit_stacks, fit_items, fit_frames = fit_hashes(data, iterations)
    weights = {k: torch.softmax(v.w.detach().float(), 0).cpu().numpy().tolist()
               for k, v in nets.items() if k.startswith("SIGMA")}
    # held-out obs_t hashes across both players, for the cross-game exclusion
    games = [(p, n) for p in players for n in held_names(p, only, iterations)]
    t0 = time.time()
    per_game, held_frames, spans = {}, [], {}
    for p, name in games:
        meta, arr = load_stage_a(data, name)
        info, tru = load_truth(truth, name, smoke)
        frames, items = game_items(arr, tru)
        lg = model_logits(nets, torch.from_numpy(frames).to(device), items["window"],
                          items["set"], device)
        h0 = [stack_hash(frames, w[0]) for w in items["window"]]
        whole = [b"".join(stack_hash(frames, r) for r in w) for w in items["window"]]
        newest = np.array([int(w[0][-1]) for w in items["window"]], np.int64)
        lo = sum(len(f) for f in held_frames)
        held_frames.append(arr["frames"])
        spans[name] = (lo, lo + len(arr["frames"]))
        per_game[name] = {"player": p, "meta": meta, "items": items, "logits": lg,
                          "h0": h0, "whole_in_fit": np.array([x in fit_items for x in whole]),
                          "h0_in_fit": np.array([x in fit_stacks for x in h0]),
                          "newest": newest + lo}
        print(f"  {name}: {len(items['set'])} items [{time.time() - t0:.0f}s]", flush=True)
    # an obs_t that recurs in another held-out game (either player)
    owners = {}
    for name, g in per_game.items():
        for h in set(g["h0"]):
            owners.setdefault(h, set()).add(name)
    held_all = np.concatenate(held_frames)
    for name, g in per_game.items():
        g["h0_in_other"] = np.array([len(owners[h] - {name}) > 0 for h in g["h0"]])
    # near-duplicates: newest frames of the held-out decisions (own items)
    for name, g in per_game.items():
        own = g["items"]["set"] == SET_OWN
        q = held_all[g["newest"][own]]
        lo, hi = spans[name]
        d_fit = min_pixel_diff(q, fit_frames, device)
        d_held = min_pixel_diff(q, held_all, device, exclude=[(lo, hi)] * len(q))
        nd = np.full(len(g["items"]["set"]), -1, np.int64)
        nd[own] = np.minimum(d_fit, d_held)
        g["near_dup"] = nd
    # read 5: O0's acting argmax at the same iteration on OF's held-out states
    agree = of_agreement(data, per_game, plan) if "OF" in players else {}
    for name, g in per_game.items():
        rec = {k: v for k, v in g["items"].items()}
        rec.update({"h0_in_fit": g["h0_in_fit"], "h0_in_other": g["h0_in_other"],
                    "whole_in_fit": g["whole_in_fit"], "near_dup": g["near_dup"]})
        for k, v in g["logits"].items():
            rec["logits/" + k] = v
        if name in agree:
            rec["o0_argmax"] = agree[name]
        tmp = out / f"{name}-logits.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **rec)
        os.replace(tmp, out / f"{name}-logits.npz")
    summary = {"games": list(per_game), "sigma_weights": weights, "plan_cap": plan["cap"],
               "source": SOURCE, "train_source": plan["source"],
               "of_void": tr.of_void(truth), "smoke": bool(smoke),
               "secs": round(time.time() - t0)}
    tmp = out / "logits.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, out / "logits.json")
    return summary


def of_agreement(data, per_game, plan):
    """For each OF held-out game: O0's acting argmax (iteration 160 for OF's
    159, else the same iteration) on each own item's obs_t, -1 elsewhere."""
    out = {}
    o0_rows = mg.rows("O0")
    by_it = {}
    for name, g in per_game.items():
        if g["player"] != "OF":
            continue
        it = g["meta"]["iteration"]
        by_it.setdefault(160 if it == 159 else it, []).append(name)
    for it, names in by_it.items():
        path = mg.snapshot_path("O0", it)
        with mg.player(path, o0_rows[it]["env_steps"]) as (run, d):
            manager = run.searcher.network_manager
            for name in names:
                g = per_game[name]
                _, arr = load_stage_a(data, name)
                stacks = mg.stacks_from_frames(arr)
                am = np.full(len(g["items"]["set"]), -1, np.int8)
                for i in np.flatnonzero(g["items"]["set"] == SET_OWN):
                    policy, _ = manager.initial_inference(stacks[int(g["items"]["decision"][i])])
                    am[i] = int(np.argmax(policy))
                out[name] = am
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--truth")
    ap.add_argument("--train")
    ap.add_argument("--out")
    ap.add_argument("--players")
    ap.add_argument("--games")
    ap.add_argument("--iterations")
    a = ap.parse_args(argv)
    base = mg.ROOT / "benchmarks" / "probes" / "testh"
    smoke = bool(a.games or a.players or a.iterations)
    data = pathlib.Path(a.data) if a.data else base / "data"
    truth = pathlib.Path(a.truth) if a.truth else base / "truth"
    train = pathlib.Path(a.train) if a.train else base / "train"
    out = pathlib.Path(a.out) if a.out else base / ("logits-smoke" if smoke else "logits")
    if not torch.cuda.is_available():
        raise SystemExit("stage D1 runs on the GPU; none is visible")
    players = tuple(a.players.split(",")) if a.players else ("O0", "OF")
    only = set(a.games.split(",")) if a.games else None
    its = set(int(x) for x in a.iterations.split(",")) if a.iterations else None
    stage_logits(data, truth, train, out, players, only, its, smoke)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
