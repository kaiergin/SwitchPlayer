"""E1: can a model trained on the agent's own play recover E0's outcome
classes? (CURRICULUM "Next experiment (pre-registered) -- E1", Kai
2026-11-02.) At the decisions of T2's own exact-eval play (the E0 census's
main line, replayed exactly), two candidates group T2's tokens of the SAME
length, and are scored against the census's frame classes restricted to the
group (the truth: probe_action_census-<run>-decisions.jsonl):
  - the 4-frame group (PRIMARY): T2's nine 4-frame tokens, one per action;
  - the 32-frame group (5 tokens) and the 16-frame group (3) (secondary).
Kai's rule holds: the emulator's branches only score the models and never
train them (no census line and no emulator branch reaches a model here).

CANDIDATES
  (A) FORWARD, no new training: T2's own dynamics g and consistency head
      (the snapshot's projector). At a state s, for each token i of a
      group, z_i = projector(g(h(s), i), with_prediction=True), h with the
      trainer's (raw) BN statistics -- the birth search's and the
      consistency loss's -- while the main line plays with the deployed
      acting states. Tokens i, j are predicted equivalent when 1 - cos(z_i,
      z_j) <= tau; the classes are the connected components. tau is the
      median one-step consistency error of the same nets over T2's own-play
      log (every logged decision of every life): 1 - cos(projector(g(h(s_t),
      a_t), True), projector(h(s_t+1), False)), as train_step's loss pairs
      them, on the un-augmented stacks (augmentation is a training-time
      regularizer; the census states are not augmented either).
  (B) INVERSE, a new model trained on own play only: an IDM q(action | s_t,
      s_t+1), a small conv net (3 conv + fc 256 -> 9) on the 4-frame
      decisions of the lineage's own-play logs (T0, T1, T2; Test B2's run
      tree). Input: s_t's last two frames + s_t+1's new frame (9 x 84 x 84;
      for a 4-frame token s_t+1 is s_t's last three frames + one new frame,
      asserted). Label: the token's ALE action. Held out: the last 10% of
      T2's lives (ceil), for the training read only. At a census decision
      whose chosen token is 4 frames long, the predicted class of the
      chosen action is {a : q(a | s, s') >= 0.1 x max_b q(b | s, s')}, s'
      the main line's next stack.

READS (pre-registered)
  (1) PRIMARY, (A) on the 4-frame group: pair-counting precision, recall
      and F1 over the 36 token pairs per decision, pooled over the
      decisions, against the census truth; the floor is the better F1 of
      the two trivial rules (all one class; all distinct), scored the same
      way. Also: the mean predicted class count against the truth's (E0:
      2.09).
  (2) (B) on the chosen 4-frame actions: precision and recall of the
      predicted class's members against the census class of the chosen
      action (the chosen action counts as a member of both; pooled member
      counts); the same floor rules (all nine actions; the chosen action
      alone); the IDM's held-out accuracy. Also reported, not used: the
      same read with the chosen action left out of both sets.
  (3) secondary: (A) on the 32- and 16-frame groups; (A) and (B) split by
      token play frequency (amendment (A1), the lineage's 1% vocabulary
      rule: a token is RARE when its plays are under 1% of all T2's
      decisions in its own-play log over iterations 23-42 (its last 20),
      strictly; (A) pairs involving a rare token against the others; (B)
      decisions whose census class of the chosen action contains a rare
      action against the others);
      per game family (FAMILY: the game's return -- the 30 noop starts fall
      into 8 near-duplicate families of one return each, whose base steps
      differ by a few).
  (4) robustness: (A) at tau x 0.5 and tau x 2, (B) at 0.05 and 0.2;
      reported, not used for the decision.
F1 = 2tp / (2tp + fp + fn); when that denominator is 0 (neither the truth
nor the prediction has a positive pair) F1 = 1.0, the two agree exactly.
Precision (recall) is None when tp + fp (tp + fn) is 0.
DECISION (on the better F1 of (1) and (2), each counted only where it beats
its floor, unrounded): >= 0.8 -> "E2"; < 0.5, or neither beats its floor ->
"KAI_CALL_LOW"; in between -> "KAI_CALL_MID". VOID when the main line is not
the census's: the census must be valid (its main_line_ok), every game must
equal the exact eval's game of the same noop start and the census's decision
count, every replayed decision's token must equal the census's (and the
hook's prior argmax), every census decision must be found.

usage: python probe_e1_equivalence.py SNAPSHOT.pth   (T2 iteration 42)
Writes probe_e1_equivalence-<run>.jsonl (the summary: tau, reads (1)-(4),
floors, IDM accuracies, verdict, main_line_ok, void_reasons, timings) and
probe_e1_equivalence-<run>-decisions.jsonl (one line per decision: per group
the truth and predicted labels and the pair distances, (B)'s fields) next to
itself. Smoke knobs (any of them non-default writes "-smoke" files):
E1_GAMES (default 30), E1_MAX_DECISIONS (score each game's first N
decisions; the main line plays on), E1_LOG_LIVES (read only each log's first
N lives), E1_IDM_EPOCHS (default 10), E1_IDM_CAP (default 20000: the IDM
training set's cap, a uniform sample with a fixed seed above it). CPU only,
2 threads (probe_exhaustive_eval's PROBE_THREADS default), as the census.
Imported as a module (the tests) it changes nothing process-wide: the env
settings and probe_exhaustive_eval's import happen only when it runs as a
script."""
import os
ENV_ID = "MsPacmanNoFrameskip-v4"
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = ENV_ID
    os.environ["PROBE_ACTING_STATES"] = "1"     # the deployed policy
import collections
import contextlib
import copy
import itertools
import json
import pathlib
import pickle
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
import torch.nn.functional as F
import muzero
import ablation
import probe_action_census as pc

DURATIONS = pc.DURATIONS
GRID = pc.GRID                               # TK's 36 order: a * 4 + j
N_ACTIONS = 9
GROUPS = (4, 16, 32)                         # same-length groups; 4 primary
PRIMARY = "4"
TAU_MULTS = {"0.5": 0.5, "1": 1.0, "2": 2.0}  # (A): tau x; "1" is the read
B_RATIOS = {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2}   # (B): "0.1" is the read
B_RATIO = "0.1"
E2_AT, LOW_BELOW = 0.8, 0.5                  # the decision's thresholds
LATE_ITERS = 20                              # the frequency split's window
RARE_PER = 100                               # (A1): rare = under 1 in 100
HELD_DIV = 10                                # T2's last 10% of lives (ceil)
CALIB_BATCH = 64
# the generation preps (Test B2's run tree): T1's vocabulary and T0's log
# (the T1 prep's master); T2's vocabulary and T1's log (the T2 prep's)
PREPS = ("two-prep-t1-s0-2026-10-29", "two-prep-t2-s0-2026-10-29")
# the IDM (fixed a priori; the held-out lives select nothing)
IDM_SEED = 0
IDM_BATCH = 64
IDM_LR = 3e-4
IDM_EPOCHS_DEFAULT = 10
IDM_CAP_DEFAULT = 20000
GAMES = int(os.environ.get("E1_GAMES") or 30)
MAX_DECISIONS = int(os.environ.get("E1_MAX_DECISIONS") or 0)
LOG_LIVES = int(os.environ.get("E1_LOG_LIVES") or 0)
IDM_EPOCHS = int(os.environ.get("E1_IDM_EPOCHS") or IDM_EPOCHS_DEFAULT)
IDM_CAP = int(os.environ.get("E1_IDM_CAP") or IDM_CAP_DEFAULT)
SMOKE = (GAMES != 30 or MAX_DECISIONS > 0 or LOG_LIVES > 0
         or IDM_EPOCHS != IDM_EPOCHS_DEFAULT or IDM_CAP != IDM_CAP_DEFAULT)


# --------------------------------------------------------------------------
# partitions and pair counting
# --------------------------------------------------------------------------
def canonical(labels):
    """Class ids by first appearance (equal labels share an id)."""
    return pc.partition(labels)


def restrict(partition, grid_indices):
    """A 36-token census partition restricted to some tokens (their GRID
    indices, in the group's order), relabelled canonically."""
    return canonical([partition[g] for g in grid_indices])


def components(n, edges):
    """The connected components of n nodes under undirected `edges`, as
    canonical labels."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return canonical([find(i) for i in range(n)])


def all_pairs(n):
    return list(itertools.combinations(range(n), 2))


def pair_counts(truth, pred, pairs=None):
    """(tp, fp, fn, tn) over unordered pairs (all of them, or `pairs`): a
    pair is positive when its two members share a class."""
    if len(truth) != len(pred):
        raise ValueError(f"{len(truth)} truth labels, {len(pred)} predicted")
    tp = fp = fn = tn = 0
    for i, j in (all_pairs(len(truth)) if pairs is None else pairs):
        t, p = truth[i] == truth[j], pred[i] == pred[j]
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif t:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    """Pooled precision, recall, F1. F1 = 2tp / (2tp + fp + fn), and 1.0
    when that is 0/0 (no positive anywhere in truth or prediction: they
    agree); precision / recall None on a zero denominator."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    den = 2 * tp + fp + fn
    return precision, recall, (2 * tp / den if den else 1.0)


def trivial_partitions(n):
    """The two trivial rules: all one class, all distinct."""
    return {"one": [0] * n, "distinct": list(range(n))}


def floor_of(f1s):
    """The floor: the better F1 of the trivial rules."""
    return max(f1s.values())


def predicted_class(q, ratio):
    """(B)'s class: every action whose posterior is >= ratio x the max."""
    q = np.asarray(q, dtype=np.float64)
    top = float(q.max())
    return [a for a, v in enumerate(q) if float(v) >= ratio * top]


def membership_counts(truth_class, pred_class):
    """(tp, fp, fn) of a predicted class's members against the true
    class's."""
    t, p = set(truth_class), set(pred_class)
    return len(t & p), len(p - t), len(t - p)


def verdict(f1_a, floor_a, f1_b, floor_b):
    """The pre-registered decision on the unrounded F1s: (verdict, the F1
    it rests on or None). Each F1 counts only where it beats its floor."""
    eligible = [f for f, fl in ((f1_a, floor_a), (f1_b, floor_b))
                if f is not None and fl is not None and f > fl]
    if not eligible:
        return "KAI_CALL_LOW", None
    best = max(eligible)
    return ("E2" if best >= E2_AT else
            "KAI_CALL_LOW" if best < LOW_BELOW else "KAI_CALL_MID"), best


# --------------------------------------------------------------------------
# the token groups
# --------------------------------------------------------------------------
def token_groups(vocab):
    """{"4": slots, "16": slots, "32": slots} of a vocabulary (slot order).
    The 4-frame group must hold one token per action, in action order
    (the IDM's outputs are actions)."""
    out = {}
    for f in GROUPS:
        out[str(f)] = [s for s, t in enumerate(vocab) if int(t[1]) == f]
    acts = [int(vocab[s][0]) for s in out[PRIMARY]]
    if acts != list(range(N_ACTIONS)):
        raise SystemExit(f"the 4-frame tokens' actions are {acts}, not one "
                         f"per action in order")
    return out


# --------------------------------------------------------------------------
# own-play logs
# --------------------------------------------------------------------------
def iter_own_play(folder, last_iteration, max_lives=0):
    """ablation.read_own_play's lives (iterations 1..last_iteration, each
    file's lives in write order; a missing iteration raises), streamed one
    life at a time as (iteration, life); `max_lives` > 0 stops after that
    many (smoke runs)."""
    n = 0
    for it in range(1, int(last_iteration) + 1):
        p = pathlib.Path(folder) / f"iter-{it:03d}.pkl"
        if not p.exists():
            raise SystemExit(f"own-play log {folder}: iteration {it} is "
                             f"missing")
        with open(p, "rb") as f:
            while True:
                try:
                    life = pickle.load(f)
                except EOFError:
                    break
                yield it, life
                n += 1
                if max_lives and n >= max_lives:
                    return


def idm_input(s, s_next, frames=4):
    """The IDM's input for a 4-frame decision: s's last two frames and
    s_next's new frame (uint8, 3 x C channels). Asserts the 4-frame shift:
    s_next is s's last frames-1 frames plus one new frame."""
    c = s.shape[0] // frames
    assert s.shape == s_next.shape and s.shape[0] == c * frames, \
        (s.shape, s_next.shape)
    assert np.array_equal(s_next[:-c], s[c:]), "not a one-frame shift"
    return np.concatenate([s[-2 * c:], s_next[-c:]], axis=0)


def life_idm_samples(traj, vocab):
    """(input, ALE action) for every 4-frame decision of one life (the last
    record is the final-stack placeholder: never a decision)."""
    out = []
    for i in range(len(traj) - 1):
        a, f = vocab[int(traj[i][1])]
        if int(f) == muzero.TOKEN_BASE_FRAMES:
            out.append((idm_input(traj[i][0], traj[i + 1][0]), int(a)))
    return out


def held_out_count(n_lives, div=HELD_DIV):
    """The last ceil(n / div) lives are held out (integer arithmetic: a
    float 0.1 x 30 would ceil to 4)."""
    return -(-int(n_lives) // int(div))


class Reservoir:
    """A uniform sample of at most `cap` items of a stream (Algorithm R),
    deterministic for a seed; every item while the stream is <= cap."""

    def __init__(self, cap, seed):
        self.cap, self.items, self.seen = int(cap), [], 0
        self.rng = np.random.RandomState(seed)

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.seen)
            if j < self.cap:
                self.items[j] = item


# --------------------------------------------------------------------------
# (A): the dynamics and consistency head
# --------------------------------------------------------------------------
Nets = collections.namedtuple("Nets", "rep dyn proj")


def _floats(stacks):
    return torch.as_tensor(np.asarray(stacks), dtype=torch.float32) / 255.0


@torch.no_grad()
def life_errors(nets, traj, batch=CALIB_BATCH):
    """One life's one-step consistency errors, one per acted decision:
    1 - cos(proj(g(h(s_t), a_t), True), proj(h(s_t+1), False)) -- train_
    step's pairing and muzero.consistency_loss's cosine. Each stack goes
    through h once (s_t+1 of one decision is s_t of the next)."""
    m = len(traj) - 1
    if m < 1:
        return np.zeros(0)
    lat = torch.cat([nets.rep(_floats([r[0] for r in traj[i:i + batch]]))
                     for i in range(0, len(traj), batch)])
    slots = torch.as_tensor([int(r[1]) for r in traj[:-1]], dtype=torch.long)
    errs = []
    for i in range(0, m, batch):
        j = min(i + batch, m)
        nxt, _, _ = nets.dyn(lat[i:j], slots[i:j])
        online = nets.proj(nxt, with_prediction=True)
        target = nets.proj(lat[i + 1:j + 1], with_prediction=False)
        errs.append(1.0 - F.cosine_similarity(online, target, dim=1))
    return torch.cat(errs).double().numpy()


@torch.no_grad()
def token_projections(nets, stack, n_slots):
    """z_i = proj(g(h(stack), i), True) for every slot i: (n_slots, d)."""
    lat = nets.rep(_floats(stack[None]))
    nxt, _, _ = nets.dyn(lat.repeat(n_slots, 1, 1, 1),
                         torch.arange(n_slots, dtype=torch.long))
    return nets.proj(nxt, with_prediction=True)


@torch.no_grad()
def cos_distances(z):
    """1 - cos between every two rows of z (float64 numpy, (n, n))."""
    cos = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=2)
    return (1.0 - cos).double().numpy()


def group_prediction(dist, slots, tau):
    """(A)'s classes of a group: components of the pairs within tau."""
    edges = [(a, b) for a, b in all_pairs(len(slots))
             if dist[slots[a], slots[b]] <= tau]
    return components(len(slots), edges)


# --------------------------------------------------------------------------
# (B): the inverse dynamics model
# --------------------------------------------------------------------------
class IDM(nn.Module):
    """q(action | s_t, s_t+1) from 9 x 84 x 84 uint8 frames (Nature-DQN
    convs, fc 256)."""

    def __init__(self, in_channels=9, n_actions=N_ACTIONS, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
                                nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x))


@torch.no_grad()
def idm_probs(net, x, batch=256):
    """q for uint8 inputs (n, 9, 84, 84): (n, 9) float64."""
    net.eval()
    out = [F.softmax(net(_floats(x[i:i + batch])), dim=1)
           for i in range(0, len(x), batch)]
    return torch.cat(out).double().numpy() if out else np.zeros((0,
                                                                 N_ACTIONS))


def idm_scores(net, x, y):
    """(accuracy, mean NLL) on a labelled set (None, None when empty)."""
    if len(x) == 0:
        return None, None
    q = idm_probs(net, x)
    y = np.asarray(y)
    acc = float(np.mean(q.argmax(1) == y))
    nll = float(-np.mean(np.log(np.maximum(q[np.arange(len(y)), y],
                                           1e-12))))
    return acc, nll


def train_idm(x, y, epochs, seed=IDM_SEED, batch=IDM_BATCH, lr=IDM_LR,
              held=None):
    """The IDM trained by cross-entropy (Adam, fixed epochs, a fixed seed:
    the init under a forked RNG, so the process's RNG is untouched; the
    order from a seeded RandomState). Returns (net, per-epoch history: the
    mean training loss and, given `held` (x, y), the held-out accuracy and
    NLL -- reported, never used to choose anything)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = IDM()
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    history = []
    for ep in range(int(epochs)):
        net.train()
        perm = rng.permutation(len(x))
        losses = []
        for i in range(0, len(x), batch):
            idx = np.sort(perm[i:i + batch])
            loss = F.cross_entropy(net(_floats(x[idx])), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        h = {"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 5)}
        if held is not None and len(held[0]):
            acc, nll = idm_scores(net, *held)
            h.update(held_acc=round(acc, 4), held_nll=round(nll, 4))
        history.append(h)
    net.eval()
    return net, history


# --------------------------------------------------------------------------
# the own-play passes
# --------------------------------------------------------------------------
def t2_pass(lives, vocab, nets, last_iteration):
    """T2's log, one pass: the calibration errors (nets None: skipped), the
    4-frame IDM samples per life, and each slot's plays in the last
    LATE_ITERS iterations."""
    errors, per_life = [], []
    late = np.zeros(len(vocab), dtype=np.int64)
    first_late = int(last_iteration) - LATE_ITERS + 1
    for it, life in lives:
        traj = life["traj"]
        if nets is not None:
            errors.append(life_errors(nets, traj))
        per_life.append(life_idm_samples(traj, vocab))
        if it >= first_late:
            for rec in traj[:-1]:
                late[int(rec[1])] += 1
    errors = np.concatenate(errors) if errors else np.zeros(0)
    return errors, per_life, late


def rare_slots(plays, per=RARE_PER):
    """(A1): the slots played under 1 / per of all the window's decisions
    (every slot's plays in the denominator), strictly; integer arithmetic.
    An empty window has no rare slot."""
    total = int(np.sum(plays))
    return [s for s, n in enumerate(plays) if int(n) * int(per) < total]


def calibration_read(errors):
    e = np.asarray(errors, dtype=np.float64)
    if not len(e):
        return {"decisions": 0}
    q = np.percentile(e, [10, 25, 50, 75, 90])
    return {"decisions": int(len(e)), "mean": round(float(e.mean()), 6),
            **{f"p{p}": round(float(v), 6)
               for p, v in zip((10, 25, 50, 75, 90), q)}}


# --------------------------------------------------------------------------
# the main line's hook
# --------------------------------------------------------------------------
class Scorer:
    """Wraps ablation.play_token: after the agent's token plays, (A) and
    (B) score the decision against the census's truth for it."""

    def __init__(self, run, nets, vocab, truth, tau, idm, out):
        self.run, self.nets, self.vocab = run, nets, tuple(vocab)
        self.truth, self.tau, self.idm, self.out = truth, tau, idm, out
        self.groups = token_groups(vocab)
        self.grid_idx = {g: [GRID.index(tuple(int(v) for v in vocab[s]))
                             for s in slots]
                         for g, slots in self.groups.items()}
        self.four_pos = {s: p for p, s in enumerate(self.groups[PRIMARY])}
        self.game, self.index, self.recs = 0, 0, []
        self.token_mismatches = 0   # the replayed token against the census's
        self.argmax_mismatches = 0  # the hook's prior argmax against it
        self.missing = 0            # decisions the census has no line for

    def decision(self, token, stack, nxt):
        token = int(token)
        with torch.no_grad():
            prior, _ = self.run.searcher.network_manager.initial_inference(
                stack)
        prior = np.asarray(prior.detach().cpu().numpy()
                           if hasattr(prior, "detach") else prior,
                           dtype=np.float64).reshape(-1)
        self.argmax_mismatches += int(int(np.argmax(prior)) != token)
        rec = {"game": self.game, "d": self.index, "token": token}
        truth = self.truth.get((self.game, self.index))
        if truth is None:
            self.missing += 1
            rec["missing"] = True
        else:
            self.token_mismatches += int(int(truth[0]) != token)
            dist = cos_distances(token_projections(self.nets, stack,
                                                   len(self.vocab)))
            rec["groups"] = {}
            for g, slots in self.groups.items():
                rec["groups"][g] = {
                    "truth": restrict(truth[1], self.grid_idx[g]),
                    "dist": [round(float(dist[slots[a], slots[b]]), 6)
                             for a, b in all_pairs(len(slots))],
                    "pred": {k: group_prediction(dist, slots, self.tau * m)
                             for k, m in TAU_MULTS.items()}}
            if token in self.four_pos:
                rec["b"] = self.inverse(token, stack, nxt,
                                        rec["groups"][PRIMARY]["truth"])
        self.recs.append(rec)
        if self.out is not None:
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()
        self.index += 1

    def inverse(self, token, stack, nxt, four_truth):
        """(B) at a 4-frame decision: the chosen action's census class and
        the IDM's predicted classes (positions in the 4-frame group are
        actions: token_groups)."""
        pos = self.four_pos[token]
        q = idm_probs(self.idm, idm_input(stack, nxt)[None])[0]
        return {"action": pos,
                "q": [round(float(v), 6) for v in q],
                "truth_class": [a for a in range(N_ACTIONS)
                                if four_truth[a] == four_truth[pos]],
                "pred_class": {k: predicted_class(q, r)
                               for k, r in B_RATIOS.items()}}

    @contextlib.contextmanager
    def hooked(self):
        orig = ablation.play_token

        def play_token(env, token, stack, inner=None):
            out = orig(env, token, stack, inner)
            if not MAX_DECISIONS or self.index < MAX_DECISIONS:
                # the next decision's stack, as play_episode builds it
                # (also its final stack at a life's end)
                nxt = ablation.push_frame(out[4], ablation.preprocess_frame(
                    out[0], ablation.IMAGE_RESHAPE))
                self.decision(token, stack, nxt)
            return out

        ablation.play_token = play_token
        try:
            yield
        finally:
            ablation.play_token = orig


# --------------------------------------------------------------------------
# the reads
# --------------------------------------------------------------------------
def _read(counts, floors=None):
    tp, fp, fn = counts[:3]
    p, r, f1 = prf(tp, fp, fn)
    out = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
           "precision": None if p is None else round(p, 4),
           "recall": None if r is None else round(r, 4),
           "f1": round(f1, 4), "_f1": f1}
    if floors is not None:
        f1s = {k: prf(*c[:3])[2] for k, c in floors.items()}
        out["floor_rules"] = {k: round(v, 4) for k, v in f1s.items()}
        out["floor"] = round(floor_of(f1s), 4)
        out["_floor"] = floor_of(f1s)
        out["beats_floor"] = bool(f1 > floor_of(f1s))
    return out


def group_read(recs, g, mult="1", positions=None, rare=None):
    """(A) on group g at tau x mult, pooled over the scored decisions; with
    `rare` (`positions`: the group's rare tokens, (A1)) only the pairs
    involving one (rare=True) or none (rare=False)."""
    rs = [r for r in recs if "groups" in r]
    counts = np.zeros(4, dtype=np.int64)
    floors = {k: np.zeros(4, dtype=np.int64) for k in ("one", "distinct")}
    n_pred, n_truth = [], []
    pairs = None
    for r in rs:
        t = r["groups"][g]["truth"]
        p = r["groups"][g]["pred"][mult]
        if pairs is None:
            pairs = all_pairs(len(t))
            if rare is not None:
                pairs = [(a, b) for a, b in pairs
                         if (a in positions or b in positions) == rare]
        counts += pair_counts(t, p, pairs)
        for k, triv in trivial_partitions(len(t)).items():
            floors[k] += pair_counts(t, triv, pairs)
        n_pred.append(len(set(p)))
        n_truth.append(len(set(t)))
    out = {"decisions": len(rs), "pairs_per_decision": len(pairs or [])}
    if not rs or not pairs:
        return out
    out.update(_read(counts, floors))
    if rare is None:
        out["classes_pred_mean"] = round(float(np.mean(n_pred)), 4)
        out["classes_truth_mean"] = round(float(np.mean(n_truth)), 4)
    return out


def b_read(recs, ratio=B_RATIO, keep=None, exclude_chosen=False):
    """(B) at `ratio`, pooled member counts over the scored 4-frame
    decisions (`keep`: a filter on the (B) records)."""
    bs = [r["b"] for r in recs if "b" in r and (keep is None or keep(r["b"]))]
    counts = np.zeros(3, dtype=np.int64)
    floors = {k: np.zeros(3, dtype=np.int64) for k in ("one", "distinct")}
    top1 = top1_in = 0
    for b in bs:
        a, t, p = b["action"], set(b["truth_class"]), set(b["pred_class"][
            ratio])
        triv = {"one": set(range(N_ACTIONS)), "distinct": {a}}
        if exclude_chosen:
            t, p = t - {a}, p - {a}
            triv = {k: v - {a} for k, v in triv.items()}
        counts += membership_counts(t, p)
        for k, v in triv.items():
            floors[k] += membership_counts(t, v)
        top = int(np.argmax(b["q"]))
        top1 += int(top == a)
        top1_in += int(top in b["truth_class"])
    out = {"decisions": len(bs)}
    if not bs:
        return out
    out.update(_read(counts, floors))
    if not exclude_chosen:
        out["pred_class_mean"] = round(float(np.mean(
            [len(b["pred_class"][ratio]) for b in bs])), 4)
        out["truth_class_mean"] = round(float(np.mean(
            [len(b["truth_class"]) for b in bs])), 4)
        out["top1"] = round(top1 / len(bs), 4)
        out["top1_in_class"] = round(top1_in / len(bs), 4)
    return out


def _public(d):
    """A read without its unrounded internals (the "_" keys)."""
    if isinstance(d, dict):
        return {k: _public(v) for k, v in d.items() if not k.startswith("_")}
    return d


def summarize(recs, groups, rare, families, held_acc=None):
    """Reads (1)-(4) and the verdict (main_line_ok aside) from the decision
    records (also from the decisions file read back); `rare` the rare
    slots (A1)."""
    rare = set(int(s) for s in rare)
    r1 = group_read(recs, PRIMARY)
    r2 = b_read(recs)
    r2["idm_held_acc"] = held_acc
    r2["excluding_chosen"] = b_read(recs, exclude_chosen=True)
    rare_pos = {g: {p for p, s in enumerate(slots) if s in rare}
                for g, slots in groups.items()}
    rare_actions = rare_pos[PRIMARY]          # 4-frame positions = actions
    fams = {}
    for n, fam in sorted(families.items()):
        fams.setdefault(str(fam), []).append(int(n))
    r3 = {"groups": {g: group_read(recs, g) for g in groups if g != PRIMARY},
          "frequency": {
              "rare_slots": sorted(rare),
              "a": {g: {"rare": group_read(recs, g, positions=rare_pos[g],
                                           rare=True),
                        "rest": group_read(recs, g, positions=rare_pos[g],
                                           rare=False)}
                    for g in groups},
              "b": {"rare": b_read(recs, keep=lambda b: bool(
                        set(b["truth_class"]) & rare_actions)),
                    "rest": b_read(recs, keep=lambda b: not (
                        set(b["truth_class"]) & rare_actions))}},
          "families": {}}
    for fam, games in fams.items():
        sub = [r for r in recs if r["game"] in games]
        r3["families"][fam] = {"games": games,
                               "a": group_read(sub, PRIMARY),
                               "b": b_read(sub)}
    r4 = {"a": {k: {g: group_read(recs, g, k) for g in groups}
                for k in TAU_MULTS if k != "1"},
          "b": {k: b_read(recs, k) for k in B_RATIOS if k != B_RATIO}}
    v, best = verdict(r1.get("_f1"), r1.get("_floor"), r2.get("_f1"),
                      r2.get("_floor"))
    return {"read_1": _public(r1), "read_2": _public(r2),
            "read_3": _public(r3), "read_4": _public(r4),
            "verdict": v, "verdict_f1": best}


# --------------------------------------------------------------------------
# validity, inputs, files
# --------------------------------------------------------------------------
def validity(games, ref, truth_games, scored, token_mismatches,
             argmax_mismatches, missing, census_ok, n_games=None):
    """(ok, reasons): the main line is the census's. `games` {noop: played
    tuple}; `ref` the exact-eval row or None (a smoke run without one: ok
    None); `truth_games` {game: the census's decisions}; `scored` {noop:
    decisions scored} against the game's decisions (at most
    MAX_DECISIONS); `n_games` the games a full run must play."""
    reasons = []
    if census_ok is not True:
        reasons.append(f"the census's main_line_ok is {census_ok!r}")
    if n_games is not None and len(games) < n_games:
        reasons.append(f"stopped after {len(games)} of {n_games} games")
    if ref is not None:
        for n, g in games.items():
            r = ref["games"].get(str(n))
            if r is None or not pc.same_game(g, r):
                reasons.append(f"game {n} {list(g)} != exact eval {r}")
    for n, g in games.items():
        if truth_games.get(n) != int(g[3]):
            reasons.append(f"game {n}: {int(g[3])} decisions played, the "
                           f"census has {truth_games.get(n)}")
        want = (int(g[3]) if not MAX_DECISIONS
                else min(MAX_DECISIONS, int(g[3])))
        if scored.get(n) != want:
            reasons.append(f"game {n}: {scored.get(n)} decisions scored of "
                           f"{want}")
    if token_mismatches:
        reasons.append(f"{token_mismatches} decisions whose token is not the "
                       f"census's")
    if argmax_mismatches:
        reasons.append(f"{argmax_mismatches} decisions whose prior argmax is "
                       f"not the agent's token")
    if missing:
        reasons.append(f"{missing} decisions the census has no line for")
    if reasons:
        return False, reasons
    return (None if ref is None else True), []


def finalize(rec, ok, reasons):
    """A main line that is not the census's makes the verdict VOID (the
    valid run's verdict kept beside it)."""
    if ok is False:
        rec["verdict_if_valid"] = rec["verdict"]
        rec["verdict"] = "VOID (" + "; ".join(reasons)[:300] + ")"
    return rec


def census_files(run_name):
    return (HERE / f"probe_action_census-{run_name}.jsonl",
            HERE / f"probe_action_census-{run_name}-decisions.jsonl")


def census_row(path, run_name, iteration):
    """The full census's summary row for the snapshot (the last matching),
    or None."""
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (r.get("run") == run_name and r.get("iteration") == iteration
                and r.get("n_games") == 30 and r.get("max_decisions") is None
                and r.get("env") == ENV_ID):
            found = r
    return found


def read_truth(path):
    """{(game, d): (token, partition)} and {game: decisions} from the
    census's decisions file."""
    truth, per_game = {}, collections.Counter()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["game"]), int(r["d"]))
            if key in truth:
                raise SystemExit(f"{path.name}: decision {key} twice")
            truth[key] = (int(r["token"]), list(r["partition"]))
            per_game[key[0]] += 1
    return truth, dict(per_game)


def _vocab(v):
    return tuple(tuple(int(a) for a in t) for t in v)


def lineage(path, vocab, iteration):
    """The three own-play logs (T2 first): folder, last iteration and
    vocabulary each, from the snapshot's run tree and the generation preps
    (cross-checked: the T2 prep's vocabulary is the snapshot's, its master
    vocabulary the T1 prep's)."""
    results, tree = path.parents[2], path.parents[4]
    p1, p2 = (json.loads((results / p / "vocab.json").read_text())
              for p in PREPS)
    if _vocab(p2["vocab"]) != _vocab(vocab):
        raise SystemExit(f"{PREPS[1]}'s vocabulary is not the snapshot's")
    if _vocab(p2["master_vocab"]) != _vocab(p1["vocab"]):
        raise SystemExit(f"{PREPS[1]}'s master vocabulary is not "
                         f"{PREPS[0]}'s vocabulary")
    t2 = path.parent.parent / path.parent.name.replace("snapshots-",
                                                       "own-play-", 1)
    logs = [{"name": "T2", "folder": t2, "last": int(iteration),
             "vocab": _vocab(vocab)},
            {"name": "T0", "folder": tree / p1["log_dir"],
             "last": int(p1["log_iterations"]),
             "vocab": _vocab(p1["master_vocab"])},
            {"name": "T1", "folder": tree / p2["log_dir"],
             "last": int(p2["log_iterations"]),
             "vocab": _vocab(p2["master_vocab"])}]
    for log in logs:
        if not log["folder"].is_dir():
            raise SystemExit(f"{log['name']}'s own-play log {log['folder']} "
                             f"is missing")
    return logs


def out_files(run_name, smoke):
    """(summary, decisions); a full run's decisions are never
    overwritten."""
    tag = f"probe_e1_equivalence-{run_name}{'-smoke' if smoke else ''}"
    dec = HERE / f"{tag}-decisions.jsonl"
    if dec.exists() and not smoke:
        raise SystemExit(f"{dec} exists: a full E1 is never overwritten "
                         f"(move it aside first)")
    return HERE / f"{tag}.jsonl", dec


def _memory():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"rss_mb": round(mi.rss / 2 ** 20),
                "peak_mb": round(getattr(mi, "peak_wset", 0) / 2 ** 20)
                or None}
    except Exception:                       # memory is a report, not a read
        return None


def main(argv):
    import probe_exhaustive_eval as pe    # sets ablation's env globals
    if pe.SEARCH_SIMS:
        raise SystemExit("E1 replays the prior argmax: unset "
                         "PROBE_SEARCH_SIMS")
    if not pe.ACTING:
        raise SystemExit("E1 replays the deployed policy: "
                         "PROBE_ACTING_STATES=1")
    if (GAMES < 1 or GAMES > 30 or MAX_DECISIONS < 0 or LOG_LIVES < 0
            or IDM_EPOCHS < 1 or IDM_CAP < 1):
        raise SystemExit(f"E1_GAMES {GAMES} must be 1..30, E1_MAX_DECISIONS "
                         f"{MAX_DECISIONS} and E1_LOG_LIVES {LOG_LIVES} >= 0, "
                         f"E1_IDM_EPOCHS {IDM_EPOCHS} and E1_IDM_CAP "
                         f"{IDM_CAP} >= 1")
    t_start = time.time()
    path = pathlib.Path(argv[0]).resolve()
    vocab = pe.snapshot_vocab(path)
    pc.check_snapshot(vocab, pe.ENV)
    groups = token_groups(vocab)
    run_name = path.parent.parent.name
    iteration = int(path.stem[-3:]) if path.stem[-3:].isdigit() else None
    summary_path, truth_path = census_files(run_name)
    census = census_row(summary_path, run_name, iteration)
    if census is None or not truth_path.exists():
        raise SystemExit(f"no full census of {run_name} @{iteration} next to "
                         f"the probe (E1's truth)")
    truth, truth_games = read_truth(truth_path)
    if len(truth) != census.get("decisions"):
        raise SystemExit(f"{truth_path.name}: {len(truth)} decisions, the "
                         f"census's summary says {census.get('decisions')}")
    ref = pc.exact_eval_row(path)
    if ref is None and not SMOKE:
        raise SystemExit(f"no deployed exact-eval row for {path} (a full E1 "
                         f"needs one to check its main line)")
    logs = lineage(path, vocab, iteration)
    pbn = pe.snapshot_prefix_bn(path)
    summary_file, dec = out_files(run_name, SMOKE)
    secs = {}
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
            pe.snapshot_tokens(vocab), pe.snapshot_prefix_layout(pbn):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
        d = pe.load(run, path)
        # the trainer's (raw) representation, kept before the acting states
        # replace the live one's BN statistics
        raw_rep = copy.deepcopy(run.trainer.representation).eval()
        pe.apply_acting_states(run, d)
        extra = set(d["acting_states"]) - {"representation", "prediction"}
        if extra:
            raise SystemExit(f"acting states for {sorted(extra)}: (A)'s "
                             f"dynamics / projector would not be the "
                             f"trainer's")
        raw_differs = any(
            not torch.equal(a, b) for a, b in zip(
                raw_rep.state_dict().values(),
                run.trainer.representation.state_dict().values()))
        nets = Nets(raw_rep, run.trainer.dynamics, run.trainer.projector)
        assert not any(n.training for n in nets)
        del d
        # 1. own play: tau, the IDM's set, the play frequencies
        t0 = time.time()
        t2 = logs[0]
        errors, t2_lives, late = t2_pass(
            iter_own_play(t2["folder"], t2["last"], LOG_LIVES), t2["vocab"],
            nets, t2["last"])
        tau = float(np.median(errors))
        hold = held_out_count(len(t2_lives))
        held = [s for life in t2_lives[len(t2_lives) - hold:] for s in life]
        pool = Reservoir(IDM_CAP, IDM_SEED)
        four = {"T2": sum(len(life) for life in t2_lives)}
        lives_n = {"T2": len(t2_lives)}
        for life in t2_lives[:len(t2_lives) - hold]:
            for s in life:
                pool.add(s)
        del t2_lives
        secs["t2_pass"] = round(time.time() - t0)
        print(f"T2 log: {lives_n['T2']} lives, tau {tau:.6f} over "
              f"{len(errors)} decisions, {four['T2']} 4-frame decisions "
              f"({hold} lives held out) [{secs['t2_pass']}s]", flush=True)
        t0 = time.time()
        for log in logs[1:]:
            n_l = n_f = 0
            for _, life in iter_own_play(log["folder"], log["last"],
                                         LOG_LIVES):
                n_l += 1
                for s in life_idm_samples(life["traj"], log["vocab"]):
                    pool.add(s)
                    n_f += 1
            lives_n[log["name"]], four[log["name"]] = n_l, n_f
        secs["t0_t1_pass"] = round(time.time() - t0)
        if not pool.items:
            raise SystemExit("no 4-frame decisions in the own-play logs")
        x = np.stack([s[0] for s in pool.items])
        y = np.array([s[1] for s in pool.items], dtype=np.int64)
        hx = (np.stack([s[0] for s in held]) if held
              else np.zeros((0,) + x.shape[1:], dtype=np.uint8))
        hy = np.array([s[1] for s in held], dtype=np.int64)
        capped = pool.seen > len(pool.items)
        del pool
        print(f"IDM set: {len(y)} training, {len(hy)} held out "
              f"{four} [{secs['t0_t1_pass']}s]", flush=True)
        t0 = time.time()
        idm, history = train_idm(x, y, IDM_EPOCHS, held=(hx, hy))
        train_acc, train_nll = idm_scores(idm, x, y)
        held_acc, held_nll = idm_scores(idm, hx, hy)
        secs["idm"] = round(time.time() - t0)
        mem_own_play = _memory()
        idm_rec = {"train": int(len(y)), "held": int(len(hy)),
                   "lives": lives_n, "held_lives": hold,
                   "four_frame_decisions": four,
                   "cap": IDM_CAP, "capped": bool(capped),
                   "label_counts_train": np.bincount(
                       y, minlength=N_ACTIONS).tolist(),
                   "label_counts_held": np.bincount(
                       hy, minlength=N_ACTIONS).tolist(),
                   "epochs": IDM_EPOCHS, "lr": IDM_LR, "batch": IDM_BATCH,
                   "seed": IDM_SEED,
                   "train_acc": train_acc, "train_nll": train_nll,
                   "held_acc": held_acc, "held_nll": held_nll,
                   "history": history}
        print(f"IDM: train acc {train_acc:.4f}, held-out acc "
              f"{held_acc if held_acc is None else round(held_acc, 4)} "
              f"[{secs['idm']}s]", flush=True)
        del x, y, hx, hy
        # 2. the census's main line, replayed and scored
        t0 = time.time()
        games, scored = {}, {}
        with open(dec, "w") as out:
            scorer = Scorer(run, nets, vocab, truth, tau, idm, out)
            with scorer.hooked():
                for n in range(1, GAMES + 1):
                    scorer.game, scorer.index = n, 0
                    games[n] = pe.play(run, n)
                    scored[n] = scorer.index
                    same = (None if ref is None else pc.same_game(
                        games[n], ref["games"].get(str(n), ())))
                    print(f"game {n}: return {games[n][0]:.0f}, "
                          f"{games[n][3]} decisions, {len(scorer.recs)} "
                          f"scored [{time.time() - t0:.0f}s]"
                          + ("" if same is None else " MATCH" if same else
                             f" MISMATCH (exact eval "
                             f"{ref['games'].get(str(n))})"), flush=True)
                    if same is False and not SMOKE:
                        break                  # VOID whatever follows
        secs["replay"] = round(time.time() - t0)
        ok, reasons = validity(games, ref, truth_games, scored,
                               scorer.token_mismatches,
                               scorer.argmax_mismatches, scorer.missing,
                               census.get("main_line_ok"),
                               n_games=None if SMOKE else GAMES)
        rare = rare_slots(late)
        families = {n: int(round(g[0])) for n, g in games.items()}
        reads = summarize(scorer.recs, groups, rare, families, held_acc)
        secs["total"] = round(time.time() - t_start)
        rec = {"snapshot": str(path), "run": run_name,
               "iteration": iteration, "env": ENV_ID, "smoke": SMOKE,
               "knobs": {"games": GAMES,
                         "max_decisions": MAX_DECISIONS or None,
                         "log_lives": LOG_LIVES or None,
                         "idm_epochs": IDM_EPOCHS, "idm_cap": IDM_CAP},
               "games": {n: list(g) for n, g in games.items()},
               "families": {str(k): v for k, v in families.items()},
               "exact_mean": round(float(np.mean([g[0] for g in
                                                  games.values()])), 3),
               "exact_eval_mean": None if ref is None else ref["exact_mean"],
               "census": {"file": truth_path.name,
                          "main_line_ok": census.get("main_line_ok"),
                          "decisions": len(truth)},
               "groups": groups,
               "tau": tau, "calibration": calibration_read(errors),
               "raw_rep_differs_from_acting": raw_differs,
               "frequency": {"iterations": [t2["last"] - LATE_ITERS + 1,
                                            t2["last"]],
                             "slot_plays": late.tolist(),
                             "decisions": int(late.sum()),
                             "rare_under": 1 / RARE_PER,
                             "rare_slots": rare,
                             "rare_tokens": [list(vocab[s]) for s in rare]},
               "idm": idm_rec,
               **reads,
               "main_line_ok": ok, "void_reasons": reasons,
               "secs": secs, "threads": torch.get_num_threads(),
               "memory": {"after_own_play": mem_own_play,
                          "end": _memory()}}
        finalize(rec, ok, reasons)
        with open(summary_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("snapshot", "games", "idm",
                                       "read_3", "read_4")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
