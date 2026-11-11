"""Test H, STAGE C: the guessers (CURRICULUM "Next experiment (pre-registered) --
Test H", METHOD: GUESSERS). Trained only on O0's own play (FIT games 0-5 of each
snapshot), early-stopped on VALIDATION (games 6-7); held-out games and every
truth file stay untouched here.

One network for every arm: a pair encoder over obs_t and one other observation
stack, stacked by channel (24 channels of 84x84, scaled to [0, 1]): conv 32
8x8/4, conv 64 4x4/2, conv 64 3x3/1, linear 512, ReLU throughout, no BatchNorm;
a learned softmax-weighted sum of the pairs' 512-vectors; a linear head of 9
move logits. Arms differ only in the pairs: S (obs_t, obs_t); F_k (obs_t,
obs_{t+k}), k = 1, 2, 4, 8, 16; SIGMA (obs_t, obs_{t+j}) for j = 1 .. 16, the
16 weights zero at the start. LIFE ENDS: an offset past a life's last base step
takes that life's last stack.

Training: Adam, no weight decay, cross-entropy on the move played, one batch
for every arm (256, or the largest SIGMA fits), mixed precision. VALIDATION
move NLL every EVAL_EVERY steps, step 0 included, on one common set (validation
decisions none of whose 17 observations occurs byte for byte in FIT); each
model keeps its best point. The registered rules, in order (plan()):
  1. S and SIGMA, seed 0, at 3e-4, 1e-3, 3e-3 under a provisional cap of 40,000
     (each with AT THE CAP); a best rate at a grid edge adds one rate beyond it.
  2. The cap: 3x the latest best step of the two chosen-rate models (after their
     extensions), rounded up to EVAL_EVERY.
  3. Every other arm type's seed 0 at the three rates (edge rule) under the cap;
     seeds 1-4 of S, F_1 and SIGMA at their chosen rates.
  4. AT THE CAP: a model whose best point falls in the last 10% of its cap gets
     one extension to twice that cap; still there, flagged.
  5. S's sanity check per seed (validation move accuracy above the majority
     move's) and TRAINING FAILED per future-arm seed (best validation NLL not
     below S's, same seed): one rerun at a third of the rate and twice the cap;
     still failing, VOID.
A model trains in its own directory; only a completion stamp makes it count,
so a model killed by a Switch session reruns from scratch (not counted as a
rerun). plan.json lists, per arm and seed, the model and point the read uses.

A real run refuses before any GPU work unless stage A's diversity check
passed, the truth is whole and failure-free, and stage B's floors.json passed
(before_gpu). The batch is pinned for the run in batch.json at its first fit;
a relaunch reuses it, or refuses when it no longer fits (never re-decided).

usage (from the run tree, under the lock, GPU):
  python benchmarks/probes/train_move_guesser.py [--data DIR] [--truth DIR]
      [--out DIR] [--smoke-cap N] [--smoke-iterations 80,160]"""
import os
if __name__ == "__main__":
    # the modules imported below query CUDA once and cache the answer, so the
    # GPU stage names its device before any import; imported as a module (the
    # tests, stage D1) it changes nothing process-wide
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse
import hashlib
import json
import math
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np
import probe_move_guesser as mg
import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX = 16
ARMS = {"S": (0,), "F1": (1,), "F2": (2,), "F4": (4,), "F8": (8,), "F16": (16,),
        "SIGMA": tuple(range(1, K_MAX + 1))}
SEEDED = ("S", "F1", "SIGMA")               # 5 seeds each
SEEDS = (0, 1, 2, 3, 4)
RATES = (3e-4, 1e-3, 3e-3)
EDGE = {3e-4: 1e-4, 3e-3: 1e-2}
PROVISIONAL_CAP = 40_000
EVAL_EVERY = 500
BATCH = 256
EVAL_PAIRS = 4096                           # pairs per evaluation chunk
TRAIN_COMMIT_GB = 14.0                      # its frames, CUDA context and GPU reservations
GPU_HEADROOM = 0.85                         # of the memory other processes leave
CAP_FACTOR = 3
LATE = 0.9                                  # "the last 10% of its cap"
SOURCES = (pathlib.Path(__file__).resolve(),)
SOURCE = mg.source_hash(SOURCES)


# --------------------------------------------------------------------------
# indices: every decision's 17 stacks as frame indices (LIFE ENDS)
# --------------------------------------------------------------------------
# the windows: one copy, in stage A's module (the truth's floors and the
# scorer read the same rule)
event_stacks = mg.event_stacks
decision_windows = mg.decision_windows


class Corpus:
    """Games' frames in one uint8 array and their decisions' windows."""

    def __init__(self, data, names):
        frames, rows = [], {"game": [], "decision": [], "move": [], "window": [],
                            "event": []}
        self.meta, off = [], 0
        for gi, name in enumerate(names):
            meta = json.loads((data / f"{name}.json").read_text())
            if meta.get("source") != mg.SOURCE:
                raise SystemExit(f"{name}: not this stage A's data")
            with np.load(data / f"{name}.npz") as z:
                arr = {k: z[k] for k in ("kind", "end", "move", "decision", "frames")}
            d, e, m, w = decision_windows(arr, off)
            frames.append(arr["frames"])
            rows["game"].append(np.full(len(d), gi, np.int32))
            rows["decision"].append(d)
            rows["event"].append(e)
            rows["move"].append(m)
            rows["window"].append(w)
            self.meta.append(meta)
            off += len(arr["kind"])
        self.names = list(names)
        self.frames = np.concatenate(frames)
        for k, v in rows.items():
            setattr(self, k, np.concatenate(v) if v else np.zeros(0))

    def stack_hashes(self, idx):
        """blake2b of each window row's stack bytes: (n, K_MAX + 1) digests
        (a stack is hashed once however many windows hold it)."""
        cache = getattr(self, "_hash_cache", None)
        if cache is None:
            cache = self._hash_cache = {}
        out = []
        for w in self.window[idx]:
            row = []
            for r in w:
                t = tuple(int(x) for x in r)
                h = cache.get(t)
                if h is None:
                    h = cache[t] = hashlib.blake2b(self.frames[list(t)].tobytes(),
                                                   digest_size=16).digest()
                row.append(h)
            out.append(row)
        return out


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------
class PairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(24, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class Guesser(nn.Module):
    def __init__(self, offsets):
        super().__init__()
        self.offsets = tuple(offsets)
        self.enc = PairEncoder()
        self.w = nn.Parameter(torch.zeros(len(self.offsets)))
        self.head = nn.Linear(512, mg.MOVES)

    def forward(self, obs, keep=None):
        """obs: gather()'s layout, (B, 1 + P, 12, 84, 84) uint8: row 0 obs_t,
        then the rows of self.offsets in order. `keep`, a subset of the
        offsets, renormalises the weights over it (read 3's cut profile; no
        retraining)."""
        b = obs.shape[0]
        o0 = obs[:, 0]
        rows = obs[:, 1:]
        p = rows.shape[1]
        if p != len(self.offsets):
            raise ValueError(f"{p} rows for {len(self.offsets)} offsets")
        pairs = torch.cat([o0.unsqueeze(1).expand(-1, p, -1, -1, -1), rows], dim=2)
        # half precision on the GPU (autocast runs the convolutions in it
        # anyway): a float32 copy of SIGMA's 16 pairs would double the input
        x = pairs.flatten(0, 1)
        half = x.is_cuda and torch.is_autocast_enabled("cuda")
        x = x.to(torch.float16 if half else torch.float32) / 255.0
        z = self.enc(x).view(b, p, -1)
        logits_w = self.w
        if keep is not None:
            mask = torch.tensor([o in keep for o in self.offsets], device=self.w.device)
            logits_w = self.w.masked_fill(~mask, float("-inf"))
        a = torch.softmax(logits_w.float(), 0)
        h = (z.float() * a[None, :, None]).sum(1)
        return self.head(h)


def gather(frames_t, windows, offsets):
    """The batch's stacks on the device: row 0 then the arm's offsets of each
    window, (B, 1 + len(offsets), 12, 84, 84) uint8 (S reads row 0 twice)."""
    rows = windows[:, [0] + list(offsets)]          # (B, 1 + P, 4)
    b, r = rows.shape[:2]
    x = frames_t[rows.reshape(-1)]                   # (B*r*4, 3, 84, 84)
    return x.view(b, r, 4 * 3, 84, 84)


# --------------------------------------------------------------------------
# one model
# --------------------------------------------------------------------------
class Trainer:
    """FIT / VALIDATION on one device: frames and windows as tensors."""

    def __init__(self, corpus, fit, val, device, batch=BATCH):
        self.device = torch.device(device)
        self.frames = torch.from_numpy(corpus.frames).to(self.device)
        self.win = torch.from_numpy(corpus.window).to(self.device)
        self.move = torch.from_numpy(corpus.move).to(self.device)
        self.fit = np.asarray(fit)
        self.val = np.asarray(val)
        self.batch = batch
        fit_moves = corpus.move[self.fit]
        self.majority = int(np.bincount(fit_moves, minlength=mg.MOVES).argmax())
        self.majority_acc = float((corpus.move[self.val] == self.majority).mean()) \
            if len(self.val) else float("nan")

    def logits(self, model, idx, keep=None, chunk=None):
        """Logits on decisions `idx`, in chunks of about EVAL_PAIRS pairs
        (SIGMA's 16 pairs a decision would otherwise overflow the GPU)."""
        chunk = chunk or max(1, EVAL_PAIRS // len(model.offsets))
        out = []
        model.eval()
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16,
                                             enabled=self.device.type == "cuda"):
            for s in range(0, len(idx), chunk):
                i = torch.as_tensor(idx[s:s + chunk], device=self.device)
                out.append(model(gather(self.frames, self.win[i], model.offsets),
                                 keep=keep).float())
        return torch.cat(out) if out else torch.zeros(0, mg.MOVES, device=self.device)

    def evaluate(self, model):
        lg = self.logits(model, self.val)
        y = self.move[torch.as_tensor(self.val, device=self.device)]
        nll = float(F.cross_entropy(lg, y))
        acc = float((lg.argmax(1) == y).float().mean())
        return nll, acc

    def train(self, arm, seed, lr, steps, mdir, resume=None, save_all=False):
        """Train (or continue) to `steps`. mdir holds history.json, bests/,
        last.pt and stamp.json, each written whole (tmp, then replace; the
        stamp last). save_all also keeps the state at every evaluation in
        evals/ (read 1's final-step R for seed 0's S and SIGMA). Returns the
        stamp."""
        torch.manual_seed(seed)
        model = Guesser(ARMS[arm]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(self.device.type,
                                      enabled=self.device.type == "cuda")
        rng = np.random.default_rng(seed)
        step, history, perm, pos = 0, [], rng.permutation(self.fit), 0
        if resume is not None:
            st = torch.load(mdir / "last.pt", map_location=self.device,
                            weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"])
            rng = st["rng"]
            step, perm, pos = st["step"], st["perm"], st["pos"]
            history = json.loads((mdir / "history.json").read_text())
        (mdir / "bests").mkdir(parents=True, exist_ok=True)
        if save_all:
            (mdir / "evals").mkdir(exist_ok=True)
        best = min((h["nll"] for h in history), default=math.inf)
        if step == 0:
            nll, acc = self.evaluate(model)
            history.append({"step": 0, "nll": nll, "acc": acc})
            best = nll
            torch.save(model.state_dict(), mdir / "bests" / "step000000.pt")
            if save_all:
                torch.save(model.state_dict(), mdir / "evals" / "step000000.pt")
        while step < steps:
            model.train()
            if pos + self.batch > len(perm):
                perm, pos = rng.permutation(self.fit), 0
            i = torch.as_tensor(perm[pos:pos + self.batch], device=self.device)
            pos += self.batch
            x = gather(self.frames, self.win[i], model.offsets)
            with torch.autocast(self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), self.move[i])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % EVAL_EVERY == 0 or step == steps:
                nll, acc = self.evaluate(model)
                history.append({"step": step, "nll": nll, "acc": acc})
                if nll < best:
                    best = nll
                    torch.save(model.state_dict(), mdir / "bests" / f"step{step:06d}.pt")
                if save_all:
                    torch.save(model.state_dict(), mdir / "evals" / f"step{step:06d}.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng, "step": step,
                    "perm": perm, "pos": pos}, mdir / "last.pt.tmp")
        (mdir / "history.json.tmp").write_text(json.dumps(history))
        os.replace(mdir / "last.pt.tmp", mdir / "last.pt")
        os.replace(mdir / "history.json.tmp", mdir / "history.json")
        stamp = {"arm": arm, "seed": seed, "lr": lr, "steps": step,
                 "batch": self.batch, "source": SOURCE, "data_source": mg.SOURCE,
                 "save_all": bool(save_all)}
        (mdir / "stamp.json.tmp").write_text(json.dumps(stamp))
        os.replace(mdir / "stamp.json.tmp", mdir / "stamp.json")
        return stamp


def best_within(mdir, cap):
    """(step, nll, acc) of the best evaluation at or before `cap`."""
    hist = json.loads((mdir / "history.json").read_text())
    pts = [h for h in hist if h["step"] <= cap]
    h = min(pts, key=lambda h: (h["nll"], h["step"]))
    return h["step"], h["nll"], h["acc"]


def best_state(mdir, cap):
    """The state dict saved at the best point within `cap` (every new best is
    saved, so the best within any cap is on disk)."""
    step, _, _ = best_within(mdir, cap)
    return torch.load(mdir / "bests" / f"step{step:06d}.pt", map_location="cpu",
                      weights_only=True), step


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
class Plan:
    def __init__(self, trainer, out, provisional=PROVISIONAL_CAP, log=print):
        self.t, self.out, self.provisional, self.log = trainer, out, provisional, log
        (out / "models").mkdir(parents=True, exist_ok=True)
        self.record = {"batch": trainer.batch, "majority": trainer.majority,
                       "majority_acc": trainer.majority_acc, "rates": {},
                       "models": {}, "flags": [], "void": [], "source": SOURCE,
                       "data_source": mg.SOURCE}

    def mdir(self, arm, seed, lr, tag=""):
        return self.out / "models" / f"{arm}-s{seed}-lr{lr:g}{tag}"

    def save_all(self, arm, seed, tag):
        """Seed 0's S and SIGMA keep every evaluation's state (read 1's
        final-step R)."""
        return arm in ("S", "SIGMA") and seed == 0 and not tag

    def ensure(self, arm, seed, lr, steps, tag=""):
        """The model trained to at least `steps`. A dir without a stamp (a
        killed run), or whose stamp, history and last.pt disagree (killed
        between their writes), is cleared and retrained; one written by other
        trainer code or other stage-A data likewise. One stamped short of
        `steps` is continued from last.pt (states past the stamp, from a
        killed extension, cleared first). A stamp at another batch refuses:
        the batch is pinned for the run (batch.json), never re-decided."""
        d = self.mdir(arm, seed, lr, tag)
        stamp = None
        if (d / "stamp.json").exists():
            stamp = json.loads((d / "stamp.json").read_text())
            if stamp.get("batch") != self.t.batch:
                raise SystemExit(f"{d.name}: trained at batch {stamp.get('batch')}, "
                                 f"the run's is {self.t.batch}")
            if (stamp.get("source") != SOURCE or stamp.get("data_source") != mg.SOURCE
                    or not self._consistent(d, stamp)):
                shutil.rmtree(d)
                stamp = None
        elif d.exists():
            shutil.rmtree(d)
        if stamp is not None and stamp["steps"] >= steps:
            return d
        if stamp is not None:
            for sub in ("bests", "evals"):
                for f in (d / sub).glob("step*.pt"):
                    if int(f.stem[4:]) > stamp["steps"]:
                        f.unlink()
        t0 = time.time()
        self.t.train(arm, seed, lr, steps, d, resume=stamp,
                     save_all=self.save_all(arm, seed, tag))
        self.log(f"  trained {d.name} to {steps} [{time.time() - t0:.0f}s]")
        return d

    @staticmethod
    def _consistent(d, stamp):
        try:
            hist = json.loads((d / "history.json").read_text())
            last = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        except (OSError, ValueError, RuntimeError):
            return False
        return hist and hist[-1]["step"] == stamp["steps"] == last["step"]

    def at_cap(self, arm, seed, lr, cap, tag=""):
        """AT THE CAP: trained to cap; a best in the last 10% gets one extension
        to twice the cap; still there, flagged. Returns (dir, used cap, best
        step, nll, acc, flagged)."""
        d = self.ensure(arm, seed, lr, cap, tag)
        step, nll, acc = best_within(d, cap)
        if step > LATE * cap:
            cap2 = 2 * cap
            d = self.ensure(arm, seed, lr, cap2, tag)
            step, nll, acc = best_within(d, cap2)
            return d, cap2, step, nll, acc, step > LATE * cap2
        return d, cap, step, nll, acc, False

    def choose_rate(self, arm, cap):
        """The best of RATES by seed 0's best validation NLL; a best rate at
        an edge adds the rate beyond it, once."""
        res = {lr: self.at_cap(arm, 0, lr, cap) for lr in RATES}
        lr = min(res, key=lambda r: res[r][3])
        if lr in EDGE:
            extra = EDGE[lr]
            res[extra] = self.at_cap(arm, 0, extra, cap)
            lr = min(res, key=lambda r: res[r][3])
        self.record["rates"][arm] = {"chosen": lr, "grid": {
            f"{r:g}": {"best_step": v[2], "nll": v[3], "cap": v[1], "flag": v[5]}
            for r, v in res.items()}}
        return lr, res

    def use(self, arm, seed, lr, cap, tag=""):
        d, used, step, nll, acc, flag = self.at_cap(arm, seed, lr, cap, tag)
        key = f"{arm}-s{seed}"
        rec = {"dir": d.name, "lr": lr, "cap": used, "best_step": step, "nll": nll,
               "acc": acc, "flagged": flag, "rerun": bool(tag)}
        if self.save_all(arm, seed, tag):
            # read (1)'s final-step model: the state at the model's used cap
            final = d / "evals" / f"step{used:06d}.pt"
            if not final.exists():
                raise SystemExit(f"{key}: no state at its final step {used}")
            rec["final_state"] = f"evals/step{used:06d}.pt"
        self.record["models"][key] = rec
        return rec

    def run(self):
        # 1-2: S and SIGMA under the provisional cap, then the cap
        lr_s, res_s = self.choose_rate("S", self.provisional)
        lr_g, res_g = self.choose_rate("SIGMA", self.provisional)
        latest = max(res_s[lr_s][2], res_g[lr_g][2])
        cap = int(math.ceil(max(EVAL_EVERY, CAP_FACTOR * latest) / EVAL_EVERY) * EVAL_EVERY)
        self.record["cap"] = cap
        self.record["cap_from"] = {"S": res_s[lr_s][2], "SIGMA": res_g[lr_g][2]}
        self.log(f"rates S {lr_s:g}, SIGMA {lr_g:g}; cap {cap}")
        rates = {"S": lr_s, "SIGMA": lr_g}
        # 3: the other arm types' grids under the cap
        for arm in ("F1", "F2", "F4", "F8", "F16"):
            rates[arm], _ = self.choose_rate(arm, cap)
        # the read's models: seed 0 of every arm, seeds 1-4 of S, F_1, SIGMA
        for arm in ARMS:
            for seed in (SEEDS if arm in SEEDED else (0,)):
                self.use(arm, seed, rates[arm], cap)
        # 5: S's sanity check, then TRAINING FAILED, per seed
        for seed in SEEDS:
            m = self.record["models"][f"S-s{seed}"]
            if m["acc"] <= self.t.majority_acc:
                m = self.use("S", seed, rates["S"] / 3, 2 * cap, tag="-rerun")
                if m["acc"] <= self.t.majority_acc:
                    self.record["void"].append(f"S-s{seed}: sanity")
        for arm in ARMS:
            if arm == "S":
                continue
            for seed in (SEEDS if arm in SEEDED else (0,)):
                s_nll = self.record["models"][f"S-s{seed}"]["nll"]
                m = self.record["models"][f"{arm}-s{seed}"]
                if m["nll"] >= s_nll:
                    m = self.use(arm, seed, rates[arm] / 3, 2 * cap, tag="-rerun")
                    if m["nll"] >= s_nll:
                        self.record["void"].append(f"{arm}-s{seed}: training failed")
        self.record["rates_chosen"] = {a: float(r) for a, r in rates.items()}
        # the flags of the read's models as they stand (a rerun replaces the
        # model it reran, flag and all)
        self.record["flags"] = sorted(k for k, m in self.record["models"].items()
                                      if m["flagged"])
        (self.out / "plan.json").write_text(json.dumps(self.record, indent=1))
        return self.record


# --------------------------------------------------------------------------
# data and sets
# --------------------------------------------------------------------------
def o0_names(split, iterations=None, games=None):
    spec = mg.RUNS["O0"]
    its = spec["iterations"] if iterations is None else iterations
    gs = mg.SPLIT[split] if games is None else [g for g in mg.SPLIT[split] if g in games]
    return [mg.game_name("O0", it, g) for it in its for g in gs]


def common_validation(corpus, fit_idx, val_idx):
    """Validation decisions none of whose 17 stacks occurs byte for byte in
    FIT's windows."""
    fit_h = set()
    for row in corpus.stack_hashes(fit_idx):
        fit_h.update(row)
    keep = [i for i, row in zip(val_idx, corpus.stack_hashes(val_idx))
            if not any(h in fit_h for h in row)]
    return np.array(keep, np.int64)


def _sigma_step(trainer, b):
    model = Guesser(ARMS["SIGMA"]).to(trainer.device)
    opt = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler(trainer.device.type,
                                  enabled=trainer.device.type == "cuda")
    i = torch.as_tensor(trainer.fit[:b], device=trainer.device)
    with torch.autocast(trainer.device.type, dtype=torch.float16,
                        enabled=trainer.device.type == "cuda"):
        loss = F.cross_entropy(model(gather(trainer.frames, trainer.win[i],
                                            model.offsets)).float(), trainer.move[i])
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def fit_batch(trainer, start=BATCH):
    """The largest batch, from `start` halving, at which a SIGMA training step
    and an evaluation chunk fit the GPU (the registration: every arm then uses
    it). On Windows the driver spills into shared system memory instead of
    raising out-of-memory, which runs many times slower and is charged to host
    commit, so the peak this process reserves must stay within GPU_HEADROOM
    of what other processes leave free."""
    if trainer.device.type != "cuda":
        return start
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    others = total - free - torch.cuda.memory_reserved()
    budget = GPU_HEADROOM * (total - others)
    b = start
    while b >= 8:
        ok = True
        torch.cuda.reset_peak_memory_stats()
        try:
            _sigma_step(trainer, b)
            # one whole evaluation chunk, as evaluate() runs it
            trainer.logits(Guesser(ARMS["SIGMA"]).to(trainer.device),
                           trainer.fit[:EVAL_PAIRS // len(ARMS["SIGMA"])])
            torch.cuda.synchronize()
            ok = torch.cuda.max_memory_reserved() <= budget
        except torch.cuda.OutOfMemoryError:
            ok = False
        torch.cuda.empty_cache()
        if ok:
            return b
        b //= 2
    raise SystemExit("SIGMA does not fit a batch of 8 within the GPU's free memory")


def before_gpu(data, truth):
    """The registration's order: the truth stage complete before any guesser
    trains, and the floors (decisions, effective games) read from it before
    any GPU work. Refuses unless stage A's diversity check passed, every held
    game's truth is present, from the current stage-B source, failure-free and
    whole (no max_decisions), and stage B's floors.json passed for O0."""
    import probe_move_guesser_truth as tr
    div = data / "diversity.json"
    if not div.exists() or not json.loads(div.read_text()).get("passed"):
        raise SystemExit("stage A's diversity check has not passed")
    for p, it, g, name in tr.held_games():
        info = tr.written(truth, name, None)
        if info is None:
            raise SystemExit(f"{name}: no whole, failure-free truth from this source")
    floors = truth / "floors.json"
    if not floors.exists():
        raise SystemExit("no floors.json: run stage B's floors step")
    f = json.loads(floors.read_text())
    if f.get("source") != tr.SOURCE or not f["O0"]["passed"]:
        raise SystemExit(f"floors not passed for O0: {f.get('O0')}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--truth")
    ap.add_argument("--out")
    ap.add_argument("--smoke-cap", type=int)
    ap.add_argument("--smoke-iterations")
    a = ap.parse_args(argv)
    smoke = bool(a.smoke_cap or a.smoke_iterations)
    base = mg.ROOT / "benchmarks" / "probes" / "testh"
    data = pathlib.Path(a.data) if a.data else base / "data"
    truth = pathlib.Path(a.truth) if a.truth else base / "truth"
    out = pathlib.Path(a.out) if a.out else base / ("train-smoke" if smoke else "train")
    if not smoke:
        before_gpu(data, truth)
    out.mkdir(parents=True, exist_ok=True)
    commit = mg.free_commit_gb() if sys.platform == "win32" else math.inf
    if commit < TRAIN_COMMIT_GB:
        raise SystemExit(f"free commit {commit:.1f} GB < {TRAIN_COMMIT_GB} GB")
    if not torch.cuda.is_available():
        raise SystemExit("stage C trains on the GPU; none is visible")
    its = (tuple(int(x) for x in a.smoke_iterations.split(","))
           if a.smoke_iterations else None)
    names = o0_names("fit", its) + o0_names("val", its)
    corpus = Corpus(data, names)
    is_fit = np.array([corpus.meta[g]["split"] == "fit" for g in corpus.game])
    fit = np.flatnonzero(is_fit)
    val_all = np.flatnonzero(~is_fit)
    val = common_validation(corpus, fit, val_all)
    print(f"FIT {len(fit)} decisions, VALIDATION {len(val_all)} ({len(val)} after "
          f"the FIT-overlap exclusion), frames {corpus.frames.shape}", flush=True)
    trainer = Trainer(corpus, fit, val, "cuda")
    pinned = out / "batch.json"
    if pinned.exists():
        b = json.loads(pinned.read_text())["batch"]
        if fit_batch(trainer, start=b) != b:
            raise SystemExit(f"the run's batch {b} no longer fits the GPU's free "
                             f"memory: free the GPU (close what holds it) and relaunch")
        trainer.batch = b
    else:
        trainer.batch = fit_batch(trainer)
        pinned.write_text(json.dumps({"batch": trainer.batch}))
    print(f"batch {trainer.batch}", flush=True)
    plan = Plan(trainer, out, provisional=a.smoke_cap or PROVISIONAL_CAP,
                log=lambda s: print(s, flush=True))
    rec = plan.run()
    rec.update({"fit": int(len(fit)), "val_all": int(len(val_all)),
                "val": int(len(val)), "smoke": smoke})
    (out / "plan.json").write_text(json.dumps(rec, indent=1))
    print("plan:", json.dumps({k: rec[k] for k in ("cap", "rates_chosen", "flags",
                                                   "void")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
