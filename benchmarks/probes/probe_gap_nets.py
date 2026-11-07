"""Test G, THE VALUE NETS (CURRICULUM "Next experiment (pre-registered) --
Test G: does the master-student value gap point at real weakness?", its "THE
VALUE NETS" paragraph and G2's first clause, Kai 2026-11-07). No RL training
and no emulator: each net learns measured returns from one own-play log and is
scored on held-out lives of that log, nothing else. Nothing here reads the
pilot's states or any later stage's, and nothing here forms g: the nets meet
the sampled states in a later stage, after the gates are read (load / predict
below are what it calls).

THE LOGS, Test B2's lineage, seed 0, read only from Y:\\SwitchPlayer-testb2-run
(probe_gap_check.RESULTS and PLAYERS):
  - T2's own-play log in full, iterations 1-42: 126 lives, 33,021 base steps,
    raw reward 66,850 (asserted on a full run).
  - T0's own-play log through iteration 45, T0's snapshot (the registration's
    cut): the whole log, iterations 1-47, holds 141 lives, 33,616 base steps,
    raw 34,430 (asserted), and its 6 lives of iterations 46-47 postdate the
    snapshot: counted (asserted 6), never rebuilt. The clip terms are checked
    against the T1 prep's record of its master, T0 (reward_unit 10, master
    ceiling 1).
THE STACKS, at base-step resolution, each life rebuilt from its records and
inner frames (ablation.OwnPlayLog: play_token logs done - 1 inner frames a
decision, the next record's stack brings the last), so a life of n base steps
holds n + 4 frames and the stack before base step t is frames t..t+3 (12 x 84
x 84 uint8, the network input). Base steps are counted by the reward tuples'
lengths, since a life's last token is often cut short, and must equal the
life's "steps"; every logged stack must equal its rebuilt stack byte for byte,
or the log is refused. Frames are held, not stacks. A stack's hash is
probe_action_census._digest of its bytes: the pilot's stack digest.
THE TARGETS, eight a stack, HEADS order, the first primary: the measured
discounted return to the end of the life, no bootstrap (a life cut short by
the cap is not bootstrapped either; such lives are counted), base step j after
t weighing gamma ** (4 (j - t)) for gamma 0.997, 0.99, 0.98, 0.95 per frame
(probe_gap_check.discounted's convention, the real gap's), on the student's
clip (probe_e3a_sequences.clipped, i.e. muzero.grown_reward per base step, at
probe_gap_check's CLIP_UNIT 10 and CLIP_CEILING 1) and raw.
THE NETS, all alike: a Nature-DQN torso of their own (convs 32 8x8/4, 64 4x4/2,
64 3x3/1, fc 512, ReLUs; inputs / 255), eight linear outputs. Targets are
standardised per head by the training stacks' mean and SD, the loss is the
mean over heads of each head's squared error (heads weighted equally), and
outputs are de-standardised. Adam 3e-4, batch 64 (E1's IDM), at most
MAX_EPOCHS epochs; after each, the primary head's mean squared error
(standardised) on the SCORED held-out stacks; the best epoch's weights are
kept (strictly lower wins, the first on ties) and training stops PATIENCE
epochs after it. Three seeds a net (the seed sets the init and the batch
order); their de-standardised outputs are averaged.
  - master: T2's log.  twin: T0's log through iteration 45.  late: the last
    third of those T0 lives (floor(n / 3); 45 of 135), read (3)'s late-only
    twin.
THE SPLIT, one a net, shared by its three seeds (so their average has held-out
lives): inside each life index (the life's place in its game, 0 at a game's
first life), floor(n / 5 + 1/2) of its n lives held out, drawn at random with a
fixed seed. A held-out stack whose hash is in the net's training stacks is left
out of the held-out score (early stopping and G2): the first 75-175 base steps
of first lives repeat byte for byte. The SCORED stacks are the rest.
G2, its first clause: each net, seed-averaged, beats a constant at the primary
head on its scored held-out stacks: R2 = 1 - SSE / SST about their own mean
(so above every constant) must be above 0, for the master and the twin, or
G2_VOID. The late twin's is printed beside them and gates nothing (read (3) is
descriptive). R2 against the training mean, R2 per head and per seed print
beside it. Written to g2.json.

THE PASSES, each a fresh process (`nets` runs the missing ones in order, then
g2); every file is stamped with the knobs and a hash of the source its numbers
depend on, written whole or not at all, and a file made otherwise is refused,
never mixed in:
  train --net N --seed S: net-N-sS.pt: the weights, the standardisation
      constants, the training stacks' hashes, the split, the history, the
      scored held-out hashes and predictions, timings and memory.
  g2: g2.json (never overwritten), through load / predict, the path a later
      stage takes; the saved held-out predictions must come back.
For a later stage: load(net, out) gives the net's seeds (stamp-checked);
.predict(stacks) takes uint8 (n, 12, 84, 84) stacks and returns (n, 8)
float64, de-standardised, averaged over seeds, columns in HEADS order;
.train_digests is the set of its training stacks' hashes (the registration
leaves out states whose stack is in either net's training data);
stack_digest(stack) is the hash to test against it.

usage: python probe_gap_nets.py nets [--out DIR]
       python probe_gap_nets.py train --net master|twin|late --seed 0|1|2
                                [--out DIR]
       python probe_gap_nets.py g2 [--out DIR]
DIR defaults to probe_gap_nets-out next to this file. Smoke knobs (any
non-default marks the stamp smoke, skips the log totals, and moves the default
DIR to probe_gap_nets-out-smoke): GAP_NETS_LIVES (read only the first N lives
of each log; 0, all), GAP_NETS_EPOCHS (MAX_EPOCHS). CPU only, THREADS threads.
Imported as a module (the tests, a later stage) it changes nothing
process-wide."""
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import argparse
import collections
import copy
import json
import math
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import torch
import torch.nn as nn
import probe_action_census as pc
import probe_e1_equivalence as e1
import probe_e3a_sequences as e3
import probe_gap_check as gc

# the logs: the last iteration file each holds, the cut (the snapshot's
# iteration), and the registration's totals over the whole log
LOGS = {"T2": {"last": 42, "cut": gc.PLAYERS["T2"]["iteration"],
               "lives": 126, "base_steps": 33021, "raw": 66850.0,
               "later_lives": 0},
        "T0": {"last": 47, "cut": gc.PLAYERS["T0"]["iteration"],
               "lives": 141, "base_steps": 33616, "raw": 34430.0,
               "later_lives": 6}}
T0_PREP = "two-prep-t1-s0-2026-10-29"          # its master is T0
NETS = {"master": {"player": "T2", "late": False, "stream": 0},
        "twin": {"player": "T0", "late": False, "stream": 1},
        "late": {"player": "T0", "late": True, "stream": 2}}
GATED = ("master", "twin")                     # G2's nets; late is read (3)'s
SEEDS = (0, 1, 2)
HEADS = tuple((scale, gamma) for scale in gc.SCALES for gamma in gc.DISCOUNTS)
PRIMARY = 0                                    # ("clip", 0.997)
STACK_FRAMES = 4
STACK_SHAPE = (12, 84, 84)
HELD_SHARE = 0.2
BATCH = 64
LR = 3e-4
MAX_EPOCHS = 30
PATIENCE = 5
EVAL_BATCH = 256
THREADS = 4
LIVES = int(os.environ.get("GAP_NETS_LIVES") or 0)
EPOCHS = int(os.environ.get("GAP_NETS_EPOCHS") or MAX_EPOCHS)
SMOKE = LIVES != 0 or EPOCHS != MAX_EPOCHS
# What the numbers depend on: this file and the code it reads the logs,
# hashes, clips and discounts with.
SOURCES = (pathlib.Path(__file__).resolve(), HERE / "probe_gap_check.py",
           HERE / "probe_e3a_sequences.py", HERE / "probe_e1_equivalence.py",
           HERE / "probe_action_census.py", ROOT / "muzero.py")
SOURCE = gc.source_hash(SOURCES)               # as this process imported them
# where this probe never writes: the read-only run tree and the experiment's
# own state directories
NO_WRITE_PARTS = ("probe_gap_check-pilot", "probe_gap_stage2")


# --------------------------------------------------------------------------
# the logs, rebuilt
# --------------------------------------------------------------------------
def log_folder(player, results=None):
    """The player's own-play log folder (exactly one under its run)."""
    results = gc.RESULTS if results is None else pathlib.Path(results)
    found = sorted((results / gc.PLAYERS[player]["run"]).glob("own-play-*"))
    if len(found) != 1:
        raise SystemExit(f"{player}: {len(found)} own-play logs under "
                         f"{results / gc.PLAYERS[player]['run']}")
    return found[0]


def check_files(folder, last):
    """The log holds iterations 1..last, no more and no fewer."""
    names = sorted(p.name for p in pathlib.Path(folder).glob("iter-*.pkl"))
    want = [f"iter-{i:03d}.pkl" for i in range(1, int(last) + 1)]
    if names != want:
        raise SystemExit(f"{folder} holds {names[:2]}..{names[-2:]} "
                         f"({len(names)} files), not iterations 1-{last}")


def reward_tuples(life):
    """Each record's raw per-base-step rewards (the first record has none)."""
    out = [r[2] for r in life["traj"]]
    if not all(isinstance(r, (tuple, list)) for r in out):
        raise ValueError("a record carries no per-base-step rewards")
    if len(out[0]):
        raise ValueError("the first record carries rewards")
    return out


def life_frames(life):
    """One logged life at base-step resolution: (frames (n + 4, 3, 84, 84)
    uint8, raw reward per base step (n,)). The first record's stack gives
    four frames; each decision i then gives its inner frames and the next
    record's newest frame, len(rewards of record i + 1) frames in all.
    ValueError when the counts or any logged stack disagree."""
    traj, inner = life["traj"], life["inner"]
    m = len(traj) - 1
    rewards = reward_tuples(life)
    if len(inner) != m:
        raise ValueError(f"{len(inner)} inner lists for {m} decisions")
    steps = [len(r) for r in rewards[1:]]
    n = sum(steps)
    if n != int(life["steps"]):
        raise ValueError(f"{n} base steps by the rewards, {life['steps']} "
                         f"logged")
    first = traj[0][0]
    if first.shape != STACK_SHAPE or first.dtype != np.uint8:
        raise ValueError(f"a stack of {first.shape} {first.dtype}")
    c = STACK_SHAPE[0] // STACK_FRAMES
    frames = np.empty((n + STACK_FRAMES, c) + STACK_SHAPE[1:], np.uint8)
    frames[:STACK_FRAMES] = first.reshape((STACK_FRAMES,) + frames.shape[1:])
    k = STACK_FRAMES
    for i in range(m):
        if steps[i] < 1 or len(inner[i]) != steps[i] - 1:
            raise ValueError(f"decision {i}: {len(inner[i])} inner frames "
                             f"for {steps[i]} base steps")
        nxt = traj[i + 1][0]
        for x in list(inner[i]) + [nxt[-c:]]:
            if x.shape != frames.shape[1:] or x.dtype != np.uint8:
                raise ValueError(f"decision {i}: a frame of {x.shape} "
                                 f"{x.dtype}")
            frames[k] = x
            k += 1
        if nxt.shape != STACK_SHAPE or not np.array_equal(
                frames[k - STACK_FRAMES:k].reshape(STACK_SHAPE), nxt):
            raise ValueError(f"decision {i}: the next logged stack is not "
                             f"the rebuilt one")
    return frames, np.array([float(x) for r in rewards for x in r])


def read_log(player, results=None, max_lives=None):
    """The player's own-play log: its lives through the cut, rebuilt
    ({"frames", "rewards", "index" (the life's place in its game),
    "iteration", "real_done", "terminated"}), and totals over every life
    read ("lives", "base_steps", "raw", "later_lives": lives after the cut,
    counted, not rebuilt). `max_lives` > 0 reads only the first that many
    (smoke runs; LIVES by default)."""
    spec = LOGS[player]
    folder = log_folder(player, results)
    check_files(folder, spec["last"])
    lives, index = [], 0
    totals = {"lives": 0, "base_steps": 0, "raw": 0.0, "later_lives": 0}
    max_lives = LIVES if max_lives is None else max_lives
    for it, life in e1.iter_own_play(folder, spec["last"], max_lives):
        where = f"{player}'s log, iteration {it}, life {totals['lives'] + 1}"
        if int(life["iteration"]) != it:
            raise SystemExit(f"{where}: logged as iteration "
                             f"{life['iteration']}")
        try:
            rewards = reward_tuples(life)
            if it <= spec["cut"]:
                frames, flat = life_frames(life)
        except ValueError as err:
            raise SystemExit(f"{where}: {err}")
        totals["lives"] += 1
        totals["base_steps"] += sum(len(r) for r in rewards)
        totals["raw"] += float(sum(sum(r) for r in rewards))
        if it <= spec["cut"]:
            lives.append({"frames": frames, "rewards": flat, "index": index,
                          "iteration": it,
                          "real_done": bool(life["real_done"]),
                          "terminated": bool(life["terminated"])})
        else:
            totals["later_lives"] += 1
        index = 0 if life["real_done"] else index + 1
    return lives, totals


def check_totals(player, lives, totals, results=None):
    """A full run's log against the registration's totals; for T0 also its
    clip terms against the T1 prep's record of its master (T0)."""
    spec = LOGS[player]
    got = (totals["lives"], totals["base_steps"], totals["later_lives"],
           len(lives))
    want = (spec["lives"], spec["base_steps"], spec["later_lives"],
            spec["lives"] - spec["later_lives"])
    if got != want or abs(totals["raw"] - spec["raw"]) > 1e-6:
        raise SystemExit(f"{player}'s log: (lives, base steps, later lives, "
                         f"rebuilt) {got}, raw {totals['raw']}; the "
                         f"registration's {want}, raw {spec['raw']}")
    if player == "T0":
        results = gc.RESULTS if results is None else pathlib.Path(results)
        prep = json.loads((results / T0_PREP / "vocab.json").read_text())
        if gc.PLAYERS["T0"]["run"] not in prep["master_snapshot"] or (
                prep["reward_unit"], prep["master_ceiling"]) != (
                gc.CLIP_UNIT, gc.CLIP_CEILING):
            raise SystemExit(f"T0's clip per its prep: unit "
                             f"{prep['reward_unit']}, ceiling "
                             f"{prep['master_ceiling']}; the targets use "
                             f"{gc.CLIP_UNIT}, {gc.CLIP_CEILING}")


# --------------------------------------------------------------------------
# targets, stacks, the split
# --------------------------------------------------------------------------
def life_returns(rewards):
    """(n, len(HEADS)): every base step's discounted return to the life's
    end on each head, from its raw rewards per base step: base step j after
    t weighs gamma ** (BASE (j - t)); the clip per base step; no
    bootstrap."""
    raw = np.asarray(rewards, dtype=np.float64)
    clip = np.array([e3.clipped(r, gc.CLIP_UNIT, gc.CLIP_CEILING)
                     for r in raw], dtype=np.float64)
    out = np.zeros((len(raw), len(HEADS)))
    for h, (scale, gamma) in enumerate(HEADS):
        x, g, acc = (clip if scale == "clip" else raw), gamma ** gc.BASE, 0.0
        for t in range(len(x) - 1, -1, -1):
            acc = x[t] + g * acc
            out[t, h] = acc
    return out


def stack_digest(stack):
    """A stack's hash: the pilot's (probe_action_census._digest of its
    bytes). uint8 only: another dtype's bytes would never match."""
    stack = np.asarray(stack)
    if stack.dtype != np.uint8 or stack.shape != STACK_SHAPE:
        raise ValueError(f"a stack of {stack.shape} {stack.dtype}")
    return pc._digest(stack.tobytes())


def select(net, lives):
    """A net's lives: all, or for the late twin the last third."""
    if NETS[net]["late"]:
        return lives[len(lives) - len(lives) // 3:]
    return lives


def split_lives(indices, stream):
    """The held-out lives (positions in the net's lives, sorted): inside each
    life index, floor(n * HELD_SHARE + 1/2) of its n lives at random, with a
    fixed seed."""
    rng = np.random.default_rng((gc.SEED, 10, int(stream)))
    groups = {}
    for pos, k in enumerate(indices):
        groups.setdefault(int(k), []).append(pos)
    held = []
    for k in sorted(groups):
        members = groups[k]
        n = int(math.floor(len(members) * HELD_SHARE + 0.5))
        held += [members[int(j)]
                 for j in rng.choice(len(members), n, replace=False)]
    return sorted(held)


class Data:
    """The stacks and targets of some rebuilt lives. Frames are held; a
    stack is built when asked for."""

    def __init__(self, lives):
        self.frames = [life["frames"] for life in lives]
        n = [len(life["rewards"]) for life in lives]
        self.life = np.repeat(np.arange(len(lives)), n)
        self.t = (np.concatenate([np.arange(k) for k in n]) if lives
                  else np.zeros(0, dtype=np.int64))
        self.y = (np.concatenate([life_returns(life["rewards"])
                                  for life in lives]) if lives
                  else np.zeros((0, len(HEADS))))
        self.digests = [stack_digest(self.stack(i))
                        for i in range(len(self.t))]

    def __len__(self):
        return len(self.t)

    def stack(self, i):
        t = int(self.t[i])
        return self.frames[int(self.life[i])][t:t + STACK_FRAMES].reshape(
            STACK_SHAPE)

    def stacks(self, idx):
        out = np.empty((len(idx),) + STACK_SHAPE, np.uint8)
        for k, i in enumerate(idx):
            out[k] = self.stack(i)
        return out

    def partition(self, held):
        """(training samples, held-out samples) for held-out life
        positions."""
        mask = np.isin(self.life, np.asarray(list(held), dtype=np.int64))
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    def scored(self, train, held):
        """The held-out samples whose stack is not among the training
        stacks', and the training hashes."""
        seen = {self.digests[i] for i in train}
        return (np.array([i for i in held if self.digests[i] not in seen],
                         dtype=np.int64), seen)


# --------------------------------------------------------------------------
# the net and its training
# --------------------------------------------------------------------------
class Torso(nn.Module):
    """A Nature-DQN torso of its own and eight linear outputs."""

    def __init__(self, in_channels=STACK_SHAPE[0], outputs=len(HEADS),
                 hidden=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
                                nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.fc(self.conv(x))


def standardiser(y):
    """Per-head mean and SD (population) of the training targets."""
    y = np.asarray(y, dtype=np.float64)
    mean, sd = y.mean(axis=0), y.std(axis=0)
    if not (len(y) and np.all(np.isfinite(sd)) and np.all(sd > 0)):
        raise SystemExit(f"a head without spread in the training targets "
                         f"(SD {sd})")
    return mean, sd


@torch.no_grad()
def forward_z(model, stacks):
    """The model's standardised outputs on uint8 stacks, (n, heads)."""
    model.eval()
    out = [model(e1._floats(stacks[i:i + EVAL_BATCH])).double()
           for i in range(0, len(stacks), EVAL_BATCH)]
    return (torch.cat(out).numpy() if out
            else np.zeros((0, len(HEADS))))


def data_z(model, data, idx):
    """forward_z over data's samples `idx`, built EVAL_BATCH at a time."""
    out = [forward_z(model, data.stacks(idx[i:i + EVAL_BATCH]))
           for i in range(0, len(idx), EVAL_BATCH)]
    return np.concatenate(out) if out else np.zeros((0, len(HEADS)))


def fit(data, train, scored, seed, stream, epochs=None, patience=PATIENCE,
        batch=BATCH, lr=LR, log=None):
    """One seed: standardise on the training targets, train on the mean over
    heads of the squared standardised error, early stop on the primary
    head's error on the scored held-out samples. The init under a forked
    RNG (the process's is untouched), the order from a seeded generator.
    Returns (model at the best epoch, mean, sd, history, best epoch)."""
    epochs = EPOCHS if epochs is None else epochs
    mean, sd = standardiser(data.y[train])
    z = ((data.y - mean) / sd).astype(np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = Torso()
    rng = np.random.default_rng((gc.SEED, 11, int(stream), int(seed)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_epoch, history = math.inf, None, 0, []
    for ep in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        losses = []
        perm = rng.permutation(np.asarray(train))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = ((model(e1._floats(data.stacks(idx)))
                     - torch.from_numpy(z[idx])) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        t_train = time.time() - t0
        held = ((data_z(model, data, scored) - z[scored]) ** 2).mean(axis=0)
        h = {"epoch": ep, "train_loss": float(np.mean(losses)),
             "held_mse": [float(v) for v in held],
             "secs_train": round(t_train, 2),
             "secs": round(time.time() - t0, 2)}
        history.append(h)
        if held[PRIMARY] < best:
            best, best_epoch = float(held[PRIMARY]), ep
            best_state = copy.deepcopy(model.state_dict())
        if log is not None:
            log(f"  epoch {ep}: train {h['train_loss']:.4f}, held primary "
                f"{held[PRIMARY]:.4f} (best {best:.4f} @{best_epoch}) "
                f"[{h['secs']:.0f}s]")
        if ep - best_epoch >= patience:
            break
    if best_state is None:
        raise SystemExit("no epoch gave a finite held-out loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, sd, history, best_epoch


def r2(pred, y, constant=None):
    """1 - SSE / SST, SST about `constant` (y's own mean by default, the
    best constant); None without spread."""
    pred, y = np.asarray(pred, np.float64), np.asarray(y, np.float64)
    c = y.mean() if constant is None else constant
    sst = float(((y - c) ** 2).sum())
    return None if sst <= 0 else 1.0 - float(((y - pred) ** 2).sum()) / sst


def g2_verdict(reads):
    """("PASS" | "G2_VOID", reasons): each gated net's seed-averaged R2 at
    the primary head above 0."""
    reasons = [f"{n}: R2 {reads[n]['r2']} at the primary head, not above 0"
               for n in GATED
               if reads[n]["r2"] is None or not reads[n]["r2"] > 0]
    return ("G2_VOID", reasons) if reasons else ("PASS", [])


# --------------------------------------------------------------------------
# files: stamped, whole or not at all, refused when made otherwise
# --------------------------------------------------------------------------
def knobs():
    return {"lives": LIVES, "epochs": EPOCHS}


def stamp():
    return {"knobs": knobs(), "smoke": SMOKE, "source": SOURCE}


def check_out(out):
    """Refuses an output directory in the read-only run tree or in the
    experiment's own state directories."""
    out = pathlib.Path(out).resolve()
    tree = gc.RESULTS.resolve().parents[1]
    if out == tree or tree in out.parents or any(
            part.startswith(NO_WRITE_PARTS) for part in out.parts):
        raise SystemExit(f"{out}: never written by this probe")
    return out


def _digest_tensor(digests):
    """Hashes as a (n, 16) uint8 tensor, sorted (torch.load's weights_only
    reads tensors, not bytes)."""
    blob = b"".join(sorted(digests))
    return torch.from_numpy(np.frombuffer(blob, dtype=np.uint8)
                            .reshape(-1, 16).copy())


def _digest_set(tensor):
    a = np.ascontiguousarray(tensor.numpy())
    return {a[i].tobytes() for i in range(len(a))}


def write(path, obj):
    """Whole or not at all; the stamp added."""
    path = pathlib.Path(path)
    obj = {**obj, "stamp": stamp()}
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def read(path):
    """A file of this probe, refused unless this run would have written it."""
    path = pathlib.Path(path)
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("stamp") != stamp():
        raise SystemExit(f"{path} was made by {obj.get('stamp')}, this run is "
                         f"{stamp()}: move it aside, files are never mixed")
    return obj


def net_file(out, net, seed):
    return pathlib.Path(out) / f"net-{net}-s{int(seed)}.pt"


def default_out():
    return HERE / f"probe_gap_nets-out{'-smoke' if SMOKE else ''}"


class Nets:
    """One net's seeds from their stamped files: predict(stacks) averages
    their de-standardised outputs; train_digests is their training stacks'
    hashes (one split a net, so the same for every seed)."""

    def __init__(self, net, out=None):
        out = default_out() if out is None else pathlib.Path(out)
        self.net, self.models, self.terms, digests = net, [], [], None
        for seed in SEEDS:
            obj = read(net_file(out, net, seed))
            if obj["net"] != net or obj["seed"] != seed:
                raise SystemExit(f"{net_file(out, net, seed)} holds "
                                 f"{obj['net']} seed {obj['seed']}")
            model = Torso()
            model.load_state_dict(obj["state_dict"])
            model.eval()
            self.models.append(model)
            self.terms.append((np.asarray(obj["mean"], np.float64),
                               np.asarray(obj["sd"], np.float64)))
            mine = _digest_set(obj["train_digests"])
            if digests is not None and mine != digests:
                raise SystemExit(f"{net}: its seeds trained on different "
                                 f"stacks")
            digests = mine
            self.held = obj["split"]["held"]
        self.train_digests = frozenset(digests)

    def predict_seeds(self, stacks):
        """(seeds, n, heads) de-standardised outputs on uint8 stacks."""
        stacks = np.asarray(stacks)
        if stacks.dtype != np.uint8 or stacks.shape[1:] != STACK_SHAPE:
            raise ValueError(f"stacks of {stacks.shape} {stacks.dtype}, not "
                             f"uint8 (n,) + {STACK_SHAPE}")
        return np.stack([forward_z(m, stacks) * sd + mean
                         for m, (mean, sd) in zip(self.models, self.terms)])

    def predict(self, stacks):
        return self.predict_seeds(stacks).mean(axis=0)


def load(net, out=None):
    """A net's three seeds, for a later stage."""
    return Nets(net, out)


def predict(stacks, net, out=None):
    """The net's 8 heads on uint8 (n, 12, 84, 84) stacks: (n, 8) float64,
    de-standardised, averaged over the seeds, columns in HEADS order."""
    return load(net, out).predict(stacks)


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------
def net_data(net, lives):
    """A net's lives, split, as Data: (data, split, training samples,
    held-out samples, scored samples, training hashes)."""
    chosen = select(net, lives)
    split = split_lives([life["index"] for life in chosen],
                        NETS[net]["stream"])
    data = Data(chosen)
    train, held = data.partition(split)
    scored, seen = data.scored(train, held)
    if not len(scored):
        raise SystemExit(f"{net}: no held-out stack to score")
    info = {"lives": len(chosen), "held": split,
            "iterations": [chosen[0]["iteration"], chosen[-1]["iteration"]],
            "by_index": {str(k): [sum(1 for c in chosen if c["index"] == k),
                                  sum(1 for p in split
                                      if chosen[p]["index"] == k)]
                         for k in sorted({c["index"] for c in chosen})},
            "truncated_lives": sum(not c["terminated"] for c in chosen)}
    return data, info, train, held, scored, seen


def stage_train(out, net, seed):
    path = net_file(out, net, seed)
    if path.exists():
        read(path)
        print(f"{path.name}: done already, reused", flush=True)
        return
    t0 = time.time()
    player = NETS[net]["player"]
    lives, totals = read_log(player)
    if not SMOKE:
        check_totals(player, lives, totals)
    data, info, train, held, scored, seen = net_data(net, lives)
    del lives
    t_read = time.time() - t0
    print(f"{net} seed {seed}: {player}'s log, {totals['lives']} lives "
          f"read, {info['lives']} used ({len(info['held'])} held out), "
          f"{len(data)} stacks: {len(train)} training, {len(held)} held "
          f"out, {len(scored)} scored ({len(held) - len(scored)} share a "
          f"training stack) [{t_read:.0f}s]", flush=True)
    model, mean, sd, history, best = fit(
        data, train, scored, seed, NETS[net]["stream"],
        log=lambda s: print(s, flush=True))
    pred = data_z(model, data, scored) * sd + mean
    n = len(gc.DISCOUNTS)
    rec = {"net": net, "seed": int(seed), "player": player,
           "heads": [[s, g] for s, g in HEADS],
           "state_dict": model.state_dict(),
           "mean": [float(v) for v in mean], "sd": [float(v) for v in sd],
           "train_digests": _digest_tensor(seen),
           "split": info, "log_totals": totals,
           "counts": {"stacks": len(data), "train": len(train),
                      "held": len(held), "scored": len(scored),
                      "held_in_training": len(held) - len(scored)},
           "raw_over_clip_sd": [float(sd[n + j] / sd[j])
                                for j in range(n)],
           "history": history, "best_epoch": best,
           "scored_digests": torch.from_numpy(np.frombuffer(
               b"".join(data.digests[i] for i in scored), dtype=np.uint8)
               .reshape(-1, 16).copy()),
           "held_pred": torch.from_numpy(pred),
           "threads": torch.get_num_threads(),
           "secs": {"read": round(t_read), "total":
                    round(time.time() - t0)},
           "memory": e1._memory()}
    write(path, rec)
    print(f"{net} seed {seed}: best epoch {best} of {len(history)}, held "
          f"primary {history[best - 1]['held_mse'][PRIMARY]:.4f}; raw / clip "
          f"SD {[round(v, 1) for v in rec['raw_over_clip_sd']]}; memory "
          f"{rec['memory']} [{rec['secs']['total']}s]", flush=True)


def read_net(net, data, train, held, scored, seen, out):
    """G2's numbers for one net through load / predict."""
    nets = load(net, out)
    if nets.train_digests != frozenset(seen):
        raise SystemExit(f"{net}: its files trained on other stacks")
    per_seed = np.concatenate(
        [nets.predict_seeds(data.stacks(scored[i:i + EVAL_BATCH]))
         for i in range(0, len(scored), EVAL_BATCH)], axis=1)
    for s, seed in enumerate(SEEDS):
        obj = read(net_file(out, net, seed))
        saved = obj["held_pred"].numpy()
        same = np.frombuffer(b"".join(data.digests[i] for i in scored),
                             dtype=np.uint8).reshape(-1, 16)
        if not np.array_equal(obj["scored_digests"].numpy(), same) or \
                np.max(np.abs(saved - per_seed[s]) / nets.terms[s][1]) > 1e-4:
            raise SystemExit(f"{net} seed {seed}: its saved held-out "
                             f"predictions do not come back")
    pred, y = per_seed.mean(axis=0), data.y[scored]
    train_mean = data.y[train].mean(axis=0)
    return {"r2": r2(pred[:, PRIMARY], y[:, PRIMARY]),
            "r2_vs_train_mean": r2(pred[:, PRIMARY], y[:, PRIMARY],
                                   train_mean[PRIMARY]),
            "r2_per_seed": [r2(p[:, PRIMARY], y[:, PRIMARY])
                            for p in per_seed],
            "r2_per_head": [r2(pred[:, h], y[:, h])
                            for h in range(len(HEADS))],
            "held_stacks": len(held),
            "scored": len(scored), "held_in_training":
                len(held) - len(scored), "train_stacks": len(train)}


def stage_g2(out):
    path = pathlib.Path(out) / "g2.json"
    if path.exists():
        raise SystemExit(f"{path} exists: G2 is never overwritten (move it "
                         f"aside first)")
    for net in NETS:
        for seed in SEEDS:
            read(net_file(out, net, seed))
    t0, reads = time.time(), {}
    for player in ("T2", "T0"):
        lives, totals = read_log(player)
        if not SMOKE:
            check_totals(player, lives, totals)
        for net in (n for n in NETS if NETS[n]["player"] == player):
            data, info, train, held, scored, seen = net_data(net, lives)
            for seed in SEEDS:
                if read(net_file(out, net, seed))["split"]["held"] != \
                        info["held"]:
                    raise SystemExit(f"{net} seed {seed}: another split")
            reads[net] = read_net(net, data, train, held, scored, seen, out)
            reads[net]["held_lives"] = len(info["held"])
            reads[net]["gated"] = net in GATED
            reads[net]["pass"] = (reads[net]["r2"] is not None
                                  and reads[net]["r2"] > 0)
            print(f"G2 {net}: R2 {reads[net]['r2']} at the primary head on "
                  f"{len(scored)} held-out stacks of {len(info['held'])} "
                  f"lives (against the training mean "
                  f"{reads[net]['r2_vs_train_mean']}; per seed "
                  f"{reads[net]['r2_per_seed']})"
                  + ("" if net in GATED else "; descriptive"), flush=True)
            del data
        del lives
    verdict, reasons = g2_verdict(reads)
    write(path, {"verdict": verdict, "reasons": reasons, "reads": reads,
                 "heads": [[s, g] for s, g in HEADS], "gated": list(GATED),
                 "secs": round(time.time() - t0), "memory": e1._memory()})
    print(("SMOKE " if SMOKE else "") + f"G2 (first clause): {verdict}"
          + (f" ({'; '.join(reasons)})" if reasons else ""), flush=True)


def run_stage(out, stage, *args):
    """A pass in its own process; a failed pass stops the run."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), stage,
           "--out", str(out), *args]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"pass {stage} {' '.join(args)}: rc {r.returncode}")


def stage_nets(out):
    if (pathlib.Path(out) / "g2.json").exists():
        raise SystemExit(f"{pathlib.Path(out) / 'g2.json'} exists: the nets "
                         f"are done")
    for net in NETS:
        for seed in SEEDS:
            if net_file(out, net, seed).exists():
                read(net_file(out, net, seed))
            else:
                run_stage(out, "train", "--net", net, "--seed", str(seed))
    run_stage(out, "g2")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("nets", "train", "g2"))
    ap.add_argument("--net", choices=tuple(NETS))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if LIVES < 0 or EPOCHS < 1:
        raise SystemExit(f"GAP_NETS_LIVES {LIVES} must be >= 0 and "
                         f"GAP_NETS_EPOCHS {EPOCHS} >= 1")
    out = check_out(args.out if args.out else default_out())
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    if args.stage == "nets":
        stage_nets(out)
    elif args.stage == "g2":
        stage_g2(out)
    else:
        if args.net is None or args.seed is None:
            raise SystemExit("train needs --net and --seed")
        stage_train(out, args.net, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
