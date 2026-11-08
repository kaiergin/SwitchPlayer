"""Test O, stage 3: the option arms' TRAINING data -- jump windows, head
targets, loss masks and the PMAX rewrite (CURRICULUM.md "Test O": DESIGN
"Head targets", "Jump training", "Prefix BN"; DEPARTURES 1, 3, 4, 6, 7).
OptionZero = rlglab/optionzero, branch iclr2025, a383d66 (data only, never
run):
  minizero/environment/base/base_env.h   getStepLength 306-316,
      getOptionPolicy 318-372, getOptionLossScale 374-416, getMaxActionID
      419-425, the OP1 tag read into Action::option_ 207-216
  minizero/environment/atari/atari.cpp   getActionFeatures 243-268 (a
      jump's per-position action planes), getReward 279-290 (a jump's
      reward: OptionZero DISCOUNTS inside it; DEPARTURE 4 sums undiscounted)
  minizero/learner/data_loader.cpp       setMuZeroTrainingData 157-213 (the
      window: pos += getStepLength(pos), targets at every position),
      updateMax 282-298 (the PMAX rewrite)
  minizero/learner/train.py              calculate_loss 158-165, the
      learner's argmax 223 / 244, update_max 274, the shared loss 275
  minizero/actor/zero_actor.cpp          PMAX 164 / 306 and OP1 165-170 at
      record time

Pure host-side (numpy / Python): TrainingHelper's paper batch builder calls
these only when its nets carry option heads (option_length L > 1); an
OPTION_LENGTH 1 run never imports this module.

THE RECORDS. A life's records 0..n, n = len(traj) - 1 the CLOSING record.
Records 0..n-1 are the acted positions -- OptionZero's action pairs, every
one "DIS" (first-move mode: zero_disable_option_ratio 1, DEPARTURE 1). The
closing record is OptionZero's state past the last action pair (pos >=
action_pairs_.size(): terminal), whether the life ended or the per-life cap
cut it (DEPARTURE 6; "the closing record counts as terminal"), and its
action is the actor's placeholder; a batch position past n (padding) is
terminal too. Trajectories are per life (DEPARTURE 6). An acted record of
an option arm carries two fields beside the usual ones (written by
ablation.Run.paper_targets from the birth search):
  "root_option"  the birth search root's dominant option, an L-tuple of
                 moves, -1 padded (all -1: none) -- OptionZero's OP1 tag,
                 the root's option child actions, read back into the
                 action's option (zero_actor.cpp 165-170, base_env.h
                 207-216). Interned (OPTION_TUPLES): one shared tuple per
                 distinct option. The refresh never writes it (it keeps
                 primitive edges only, DEPARTURE 3).
  "pmax"         the per-record prior argmax, OptionZero's PMAX tag:
                 written at birth from the birth search root's head-0
                 argmax (zero_actor.cpp 164 / 306: p_max_id_, the root
                 policy's argmax) and REWRITTEN by the learner at every
                 sample from its own policy logits, root and unrolled
                 latents alike (train.py 223 / 244 / 274, updateMax). A
                 record without one (a 5-field prefill, a pre-option
                 checkpoint) reads the argmax of its stored policy, as
                 getMaxActionID's fallback (419-425), first maximum on ties
                 (std::max_element; np.argmax).
The closing record carries neither: nothing reads them there.

THE ORDER (PMAX). OptionZero builds a batch (its data loader reads PMAX in
getOptionPolicy / getOptionLossScale), runs the forward, then writes PMAX
at every window position from that forward's argmax (update_max, before
the backward); the next batch is built after. The trainer mirrors it: the
batch builder reads PMAX (head_targets), train_step keeps each position's
argmax of its forward's policy logits and writes it back after the step
(write_pmax), before the next batch is built. The written values are the
forward's, so writing after the optimizer step rather than before the
backward changes nothing. A non-finite step that is skipped writes no PMAX,
as it writes no priority (the trainer's SKIP_NONFINITE_GRAD_STEPS; OptionZero
has no skip). With TRAIN_PREFETCH the next batches are built ahead, so they
read PMAX one to TRAIN_PREFETCH_DEPTH + 1 steps late, as priorities do
(muzero's TRAIN_PREFETCH note); O0's recipe runs it off (the default), and
the option arms' gate (stage 4) is to refuse it.
Positions written: the acted records the window visits (OptionZero's
setActionPairInfo refuses pos >= size), rows in order then positions in
order, so a position two rows share takes the later row's value. The
written argmax is the TRAINING forward's (train-mode BN; the root's on the
augmented observation under USE_AUGMENTATION, where OptionZero's Atari
learner has no augmentation -- a departure recorded in CURRICULUM).
"""
import numpy as np

# Interned option tuples: every record holding the same option shares one
# tuple object (9 moves at L 3: at most 9 ** 3 + 1 of them).
OPTION_TUPLES = {}


def intern_option(moves, length):
    """The record's "root_option": an L-tuple of the option's moves, -1
    padded past `length` (0 or 1 = no option: all -1), interned."""
    moves = [int(m) for m in moves]
    L = len(moves)
    length = int(length)
    key = tuple(moves[:length] + [-1] * (L - length)) if length >= 2 \
        else (-1,) * L
    return OPTION_TUPLES.setdefault(key, key)


def option_moves(rec):
    """The stored suggestion's moves (the -1 padding dropped); () when the
    record has none (or carries no field)."""
    opt = rec.get("root_option")
    if not opt:
        return ()
    return tuple(m for m in opt if m >= 0)


def record_pmax(rec):
    """getMaxActionID (base_env.h 419-425): the record's PMAX, else the
    argmax of its stored policy (first maximum)."""
    pm = rec.get("pmax")
    if pm is None or pm < 0:
        return int(np.argmax(rec["target_policy"]))
    return int(pm)


def step_length(traj, pos):
    """getStepLength (base_env.h 306-316) in first-move mode (every action
    pair "DIS"): the number of positions the unroll step from `pos` spans
    -- len(option) where the moves played at pos, pos+1, ... all match the
    stored suggestion at pos, else 1. Every move of the option must sit at
    an ACTED position (pos + j < n, OptionZero's pos + length <
    action_pairs_.size()), so a match stops before the closing record,
    whose action is a placeholder; the landing itself may be the closing
    record. Past the acted positions (the closing record, padding): 1."""
    n = len(traj) - 1
    if pos >= n:
        return 1
    opt = option_moves(traj[pos])
    if len(opt) < 2:
        return 1
    for j, m in enumerate(opt):
        if pos + j >= n or int(traj[pos + j]["action"]) != m:
            return 1
    return len(opt)


def window(traj, start, unroll_steps):
    """setMuZeroTrainingData's walk (data_loader.cpp 168-210): positions
    [K + 1] (pos_0 = start, pos_{i+1} = pos_i + step_length(pos_i), past the
    trajectory's end one per step, as OptionZero's) and spans [K] (base
    steps each unroll edge covers)."""
    pos, spans = [int(start)], []
    p = int(start)
    for _ in range(int(unroll_steps)):
        s = step_length(traj, p)
        p += s
        pos.append(p)
        spans.append(s)
    return pos, spans


def span_of(traj, start, unroll_steps):
    """Base steps a window from `start` really spans (pos_K - pos_0)."""
    p = int(start)
    for _ in range(int(unroll_steps)):
        p += step_length(traj, p)
    return p - int(start)


def truncated_roots(traj, unroll_steps, option_length):
    """The window roots of a TRUNCATED life longer than the unroll: the
    plain builder keeps windows off the cut tail (roots 0..len-K-1, so the
    last position is at most the closing record). With jumps a window spans
    span_of(start) positions, up to K x L (15 at K 5, L 3), so a root is
    legal iff start + span_of(start) <= n = len - 1 -- the positions it
    really spans, counted per root. Roots below len - K x L always fit;
    only the last K x L - K candidates are walked. At L 1 (every span K)
    this is range(len - K) exactly.
    A cut life with K < len <= K x L can have NO fitting root (every window
    jumps past the cut: len 7, K 5, length-3 jumps played at 0 and 1). It
    then keeps root 0 under the plain fallback (its window padded past the
    cut), as a short cut life (len <= K) keeps its roots (audit r1 of stage
    3: a replay of such lives alone had no root, and sampling crashed)."""
    n = len(traj) - 1
    K, L = int(unroll_steps), int(option_length)
    safe = max(0, len(traj) - K * L)
    out = list(range(min(safe, len(traj) - K)))
    for idx in range(safe, len(traj) - K):
        if idx + span_of(traj, idx, K) <= n:
            out.append(idx)
    return out or [0]


def jump_reward(traj, prev, pos):
    """The reward target of the edge from position `prev` to `pos` (both
    inside the trajectory): the rewards of the positions it enters, summed
    UNDISCOUNTED (DEPARTURE 4; OptionZero's getReward discounts them,
    atari.cpp 279-290). A one-step edge is the record's own reward,
    unchanged."""
    if pos - prev == 1:
        return traj[pos]["reward"]
    total = traj[prev + 1]["reward"]
    for q in range(prev + 2, pos + 1):
        total = total + traj[q]["reward"]
    return total


def jump_row(traj, pos, span, option_length):
    """The unroll edge's dynamics input, an L-list -1 padded: the moves
    played at pos .. pos + span - 1 (a jump: the option's moves; one step:
    [a, -1, ...]) -- OptionZero's per-position action planes
    (getActionFeatures, atari.cpp 243-268)."""
    row = [int(traj[pos + j]["action"]) for j in range(span)]
    return row + [-1] * (int(option_length) - span)


def pad_row(option_length):
    """A padded edge past the trajectory's end: move 0, as the plain
    builder's padding action (OptionZero draws a random move there)."""
    return [0] + [-1] * (int(option_length) - 1)


def head_targets(traj, pos, option_length, action_space, tgt, scale):
    """getOptionPolicy (base_env.h 318-372) and getOptionLossScale
    (374-416) at `pos`, first-move mode, for the EXTRA heads 1..L-1 (head 0
    is the policy, trained by the policy loss), with DEPARTURE 7's explicit
    product. Writes tgt [L-1, A+1] (class A = stop) and scale [L-1] in
    place (both arrive zeroed / any; every entry is set).

    Counting the policy as head 0, m_j = PMAX at pos + j (record_pmax: the
    NETWORK's top move, not pi's) and pi(q) the record's stored policy
    (birth or refreshed, read now: built at batch time):
      * pos at or past the closing record (terminal): every head stop,
        scale 1 (325-330; 380-384);
      * the move played at pos misses m_0: every head stop and scale 0, the
        extra heads masked (341-345 targets; 397 the mask);
      * else, for i = 1..L-1: pos + i terminal -> heads i.. stop, scale 1;
        else head i puts prod_{j=1..i} pi(pos+j)[m_j] on m_i and the rest
        on stop, scale 1 (OptionZero puts pi(pos+i)[m_i] alone, 340-343;
        DEPARTURE 7), and a miss at pos + i (its played move is not m_i)
        keeps head i's target and puts stop on the heads after it, scale 1
        (341-347; 398-399)."""
    L, A = int(option_length), int(action_space)
    n = len(traj) - 1
    tgt[:] = 0.0
    tgt[:, A] = 1.0
    if pos >= n:
        scale[:] = 1.0
        return
    rec = traj[pos]
    if int(rec["action"]) != record_pmax(rec):
        scale[:] = 0.0
        return
    scale[:] = 1.0
    prod = 1.0
    for i in range(1, L):
        q = pos + i
        if q >= n:
            return
        r = traj[q]
        m = record_pmax(r)
        prod *= float(r["target_policy"][m])
        tgt[i - 1, m] = prod
        tgt[i - 1, A] = 1.0 - prod
        if int(r["action"]) != m:
            return


def pmax_ref(traj, pos):
    """The record a window position's argmax is written to: an acted
    record, else None (the closing record, padding)."""
    return traj[pos] if pos < len(traj) - 1 else None


def validate_jump_rows(actions, action_space):
    """Host-side check of a batch's [B, K, L] edge rows before the forward
    (the training dynamics call skips its own, check_jump=False): every row
    holds a first move in [0, A), its other entries are moves in [0, A) or
    -1, and the -1 padding is a suffix. Raises ValueError."""
    a = np.asarray(actions)
    if a.ndim != 3:
        raise ValueError(f"option batch actions must be [B, K, L]; got "
                         f"shape {a.shape}")
    A = int(action_space)
    if not (a[..., 0] >= 0).all():
        raise ValueError("a jump row without a first move (-1 at position 1)")
    if (a >= A).any() or (a < -1).any():
        raise ValueError(f"a jump row with a move outside [0, {A}) or -1")
    valid = a >= 0
    if a.shape[2] > 1 and (valid[..., 1:] & ~valid[..., :-1]).any():
        raise ValueError("a jump row with a move after its -1 padding (the "
                         "padding must be a suffix)")


def write_pmax(refs, argmax):
    """updateMax (data_loader.cpp 282-298): refs [B][K+1] records or None,
    argmax [K+1, B] the forward's policy argmax per position; rows in
    order, then positions in order (a later write wins)."""
    for b, row in enumerate(refs):
        for step, rec in enumerate(row):
            if rec is not None:
                rec["pmax"] = int(argmax[step, b])
