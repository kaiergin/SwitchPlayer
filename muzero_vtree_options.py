"""Test O, stage 2: OptionZero's option edge in the fresh-root vectorized
PUCT search (CURRICULUM.md "Test O": DESIGN "Dominant option" and "Tree",
DEPARTURES 1, 2 and 4). OptionZero = rlglab/optionzero, branch iclr2025,
a383d66 (data only, never run): minizero/actor/mcts.cpp, mcts.h,
zero_actor.cpp. v1 (muzero_vtree.VectorizedPUCT) is untouched and stays
parity-pinned; VectorizedOptionPUCT subclasses it and is a drop-in in the
birth search: the same search() arguments (plus option_edge, return_info,
keep_tree) and the same (policies, values, visit_counts) return.

THE OPTION (zero_actor.cpp 368-381, 395-396; dominant_option below). From
the prediction net's option heads [B, L, A+1] (head 0 = the policy logits
plus a fixed stop logit of -100; networks.PredictionNetwork), softmaxed per
head in float32 as muzero_atari_network.py 179/197: each head's argmax in
turn (std::max_element: the FIRST maximum on ties), ending at the first
stop (class A); length at least 2, else none. No 0.5 threshold (the code
has none). p(o) = the last head's probability of its own move;
1 - p(o) is the primitive's prior in the option-against-primitive choice
only. A node's option is read off the heads at the node's evaluation.

THE TREE. Every node is a primitive node; an option edge is a pointer from
its holder to the node reached by following the option's moves through the
primitive children (mcts.cpp 248-268: OptionZero's option child IS that
deep primitive node). One option edge per node beside the A primitives.

  * Statistics are ROOT-RELATIVE (mcts.cpp 271-288, 54-56): every node
    keeps the root-relative total reward of its first evaluation (tot =
    tot(parent of that evaluation) + discount^depth(parent) * r) and its
    depth in base steps; a backup adds the leaf's root-relative total
    tot(leaf) + discount^depth(leaf) * value(leaf) to EVERY node on the
    path, placeholders included. A child's Q seen from node n is
    (mean(child) - tot(n)) / discount^depth(n) -- for a primitive child
    exactly v1's r + d * q, so with no option v1's search comes back to
    float tolerance (not bit for bit: the sums are formed differently).
  * Primitive statistics INCLUDE the option's: a backup through an option
    edge passes the option's first move's primitive child and every node
    along the option (mcts.cpp 203-208 push them on the path). The
    landing's own count / mean are the option's (mcts.cpp 311-312).
  * Placeholders: nodes on the option's path that have no evaluation yet.
    They start at zero visits and take visits and value from backups like
    any node; when their tree parent expands they are FILLED (the edge
    fields set, the evaluation written if they have none) and keep their
    statistics (mcts.cpp 239-247 keeps existing children; 255-265 creates
    them). Only the path nodes are created here, not all A siblings as
    OptionZero allocates (257-264): a sibling placeholder carries nothing
    until its parent is evaluated, and selection never scores the children
    of an unexpanded node, so the two are indistinguishable.

SELECTION (mcts.cpp 192-220, 300-336). v1's primitive PUCT among the A
children (v1's score: min-max normalised composite + c_puct * prior *
sqrt(sum of child visits) / (1 + visits); primitive priors are the
policy row's, mcts.cpp 336). If the pick is the holder's option's first
move, option against primitive with the option's statistics taken out
(mcts.cpp 300-325): total = the primitive child's (inclusive) visits,
n_o = the landing's visits, n_p = total - n_o; Q_o = the landing's Q from
the holder, Q_p = the primitive child's statistics minus the landing's;
score = norm(Q) + c_puct * prior * sqrt(total) / (1 + n), prior p(o) for
the option and 1 - p(o) for the primitive; the option iff its score is
STRICTLY greater (324). The option crosses the jump in one step.
  Departures and readings, each resolved toward OptionZero's code where
  the registration leaves it open:
  * DEPARTURE 2: a zero-visit option and a zero-visit primitive both use
    r + d * value -- the option its holder's jump row (r_jump, discount ^
    len, the jump row's value), the primitive its own edge's -- in place of
    OptionZero's init_q_value (317). The primitive side reads "zero-visit"
    as n_p == 0 (every visit of the primitive child came through the
    option), as OptionZero's primitive_count == 0.
    CONSEQUENCE (kept as registered; Kai's / the coordinator's decision,
    2026-11-08): an option CAN be crossed on the very first visit of its
    first move. At n_c == 0 both exploration terms are 0 (sqrt(0)), so that
    first choice is decided by the two zero-visit composites alone and p(o)
    plays no part in it. OptionZero's code cannot do this: at n_c == 0 both
    sides score init_q_value with no bonus, the tie goes to the primitive
    (324), so its first pass through a first move is always primitive. It
    matches v1's treatment of unvisited primitives (r + d * value).
  * v1's c_puct is the PUCT coefficient in both stages (OptionZero's
    puct_init + log((1 + N + base) / base), 71/319, is not adopted), and
    v1's normalisation (MinMaxStats with EfficientZero's floor, identity
    until a range exists) scores both sides, not OptionZero's [-1, 1]
    rescale with its clamp (57-63). Required for stop-only parity with v1.
    The decomposition (mean * total - mean_o * n_o) / n_p is done on the
    raw Q's then normalised: normalisation is affine, so this equals
    OptionZero's order (315) exactly in exact arithmetic.
  * sqrt(total) is the PRIMITIVE CHILD's count, as the code (308, 320,
    322), not the holder's.
  * No virtual loss (v1's sequential-within-tree argument holds).
  * max_depth caps SELECTION STEPS, as v1's loop: a jump is one step, so a
    descent can reach up to max_depth * L base steps. (OptionZero's code
    has no cap. In selection steps MAX_DEPTH 50 cannot bind at 50
    simulations; in base steps it would be a different, looser cap -- a
    descent of 17 or more length-3 jumps passes 50 base steps.)

EXPANSION (DEPARTURE 2: eager, as v1). Expanding node n evaluates its A
primitive children in one wave and, when n holds an option (and the edge
is on), the option's landing as one extra row of the same wave: dynamics
from n's latent with the option's moves as one [L] -1-padded jump input
(networks.DynamicsNetwork), prediction (with option heads) and one
value-prefix LSTM call from n's state. Holder semantics: a landing can have
two or more in-edges (its primitive edge from its tree parent, jumps from
one or more holders) and can already exist below a crossed ancestor
(repeats make it common). So
  * the jump's r, its discount (DISCOUNT ** len, DEPARTURE 4: one
    predicted reward per jump, undiscounted inside -- it is the jump row's
    reward or prefix increment, never a discounted sum of inner steps) and
    the jump row's value sit on the HOLDER (opt_r, opt_len, opt_v);
  * a node's own reward / value / prior fields belong to its PRIMITIVE
    in-edge (set when its tree parent expands);
  * the FIRST evaluation of a node keeps its latent, policy row, option
    suggestion, prefix state (h, c, prefix, prefix length), root-relative
    total reward, depth, and the value it backs up as a leaf (OptionZero
    evaluates a node once: backup 276-281 sets value / total reward from
    that one evaluation). A later evaluation of the same node (a jump onto
    an evaluated node, or a primitive row onto a jump-evaluated landing)
    writes only its edge's fields.
No option noise at the root (first-move mode: zero_actor.cpp 265,
enable_option_ is false at zero_disable_option_ratio 1); the primitive
priors take v1's Dirichlet noise as ever.

BACKUP. Parent pointers ARE OptionZero's node path: a jump from holder X to
landing L passes exactly X's primitive child and the placeholders between,
which are L's tree ancestors. Every node from the leaf to the root adds the
leaf's root-relative total. The range takes, post-increment as v1, every
path node's Q from its tree parent (v1's update) where that parent has an
evaluation (a placeholder parent has no total reward to be relative to),
plus, for every option edge the descent crossed, the landing's Q from the
holder. The root seeds the range from its primitive children's zero-visit
composites only, as v1 -- never its option's (audit r1: neither v1 nor
OptionZero seeds it, and a seeded option composite would make edge-on
targets differ from edge-off ones where no option is ever crossed).
OptionZero's own bound (a map of root-relative means, 365-374) is not
adopted (v1 parity).

READOUT (first-move mode, DEPARTURE 1). Visit fractions over the A
primitive root children -- an option's visits count on its first move's
primitive edge -- and the root's mean as the root value, both as v1.

PREFIX BN. One jump is one LSTM call from the holder's state, at the step
the holder's primitive rows take (1 after a reset, else the holder's
prefix length + 1), so a jump-evaluated landing holds prefix length =
that step; its edge reward follows v1's is_reset rule (the prefix after a
reset, else the increment over the holder's prefix).

OPTION_EDGE False (arm OH): the same heads, the same per-node and root
suggestions, no jump row and no option edge: v1's search to float
tolerance. Edge on against edge off with NO option crossed: identical when
every jump row evaluates as its primitive path would (tests pin it). They
can differ otherwise, with no crossing at all: a landing the jump row
evaluated first keeps that evaluation when its primitive parent later
expands (the first-evaluation rule above), so the primitive descent backs
up the jump's value and total. Measured in the r1 fix round: 61-63 of 80
roots on fake nets whose jump rows differ from the primitive path, 0 of 24
on random real nets at 50 simulations (on real nets, batch-shape wobble
from the jump rows also enters). OPTION_LENGTH 1 nets carry no heads: no option anywhere (the
production birth search keeps v1 there). Options replace the token
vocabulary: TOKEN_VOCAB with this engine raises.

MEMORY. Node tables hold 1 + (sims + 1) * (A + L - 1) rows (each expansion
adds at most A children and L - 1 path nodes), against v1's
1 + (sims + 1) * A: +22% at A 9, L 3.

RETURNS. search() -> (policies [T, A], root values [T], visit counts [T, A])
as v1; with return_info=True also a dict of numpy arrays (also kept on
self.last_info after every call):
  root_option [T, L] int (-1 padded), root_option_len [T] (0 = none),
  root_option_p [T]      the dominant option at the root, from the heads
                         (also when the edge is off), and its p(o)
  root_pmax [T]          the root policy head's argmax (zero_actor.cpp
                         304-307's p_max_id_): head 0's float32 softmax
                         through option_argmax, the very argmax the
                         option's first move reads, so the two never
                         disagree on a near-tie (ties to the first); L = 1
                         nets: the policy softmax's argmax
  expanded_nodes [T]     expansions (the root included); placeholders are
                         never expanded before they are filled
  expanded_with_option [T]  expansions whose node's heads suggest an option
                         (edge on: each got an edge; off: would have)
  option_len_hist [T, L+1]  those suggestions by length (index 0 = none)
  simulations [T]
  sims_crossing_option [T]  simulations whose descent crossed >= 1 option
  crossed_len_hist [T, L+1] option edges crossed, by length
  leaf_depth_sum [T], leaf_depth_max [T]  descents' leaf depth, base steps
  option_edge, option_length
keep_tree=True also keeps the node tables on self.last_tree (tests).
"""
import numpy as np
import torch

import muzero
from muzero_vtree import VectorizedPUCT


def option_argmax(option_probs):
    """[B, L] each head's argmax over its A+1 classes: the first maximum,
    as std::max_element (zero_actor.cpp 372). The one argmax both the
    dominant option and the root's PMAX read."""
    return option_probs.argmax(dim=2)


def dominant_option(option_probs, arg=None):
    """zero_actor.cpp calculateOption (368-381) + setOptionInfo's p(o)
    (395-396) on [B, L, A+1] per-head probabilities (class A = stop).

    Returns (moves [B, L] long, -1 past the option; length [B] long, 0 when
    there is none; p [B], the last head's probability of its own move, 0
    when none). Each head's argmax in turn (option_argmax; `arg` passes one
    already taken) ending at the first stop; a length under 2 is no
    option."""
    B, L, A1 = option_probs.shape
    stop = A1 - 1
    dev = option_probs.device
    if arg is None:
        arg = option_argmax(option_probs)                         # [B, L]
    before_stop = torch.cumsum((arg == stop).long(), dim=1) == 0
    length = before_stop.long().sum(dim=1)
    length = torch.where(length >= 2, length, torch.zeros_like(length))
    pos = torch.arange(L, device=dev).unsqueeze(0)
    moves = torch.where(pos < length.unsqueeze(1), arg,
                        torch.full_like(arg, -1))
    b = torch.arange(B, device=dev)
    last = (length - 1).clamp(min=0)
    p_last = option_probs[b, last, arg[b, last]]
    p = torch.where(length > 0, p_last, torch.zeros_like(p_last))
    return moves, length, p


def minmax_normalize(x, minv, maxv, floor):
    """v1's MinMaxStats normalisation of composites x ([T] or [T, k]) with
    per-tree bounds [T]: (x - min) / max(max - min, floor) once a range
    exists, x itself before."""
    if x.dim() == 2:
        minv, maxv = minv.unsqueeze(1), maxv.unsqueeze(1)
    span = (maxv - minv).clamp(min=floor)
    return torch.where(maxv > minv, (x - minv) / span, x)


def option_vs_primitive(tot_n, dacc_n, n_c, w_c, n_o, w_o, zv_option,
                        zv_primitive, p_o, c_puct, normalize):
    """mcts.cpp selectOptionChild (300-325) at holders n, all [T] float64:
    tot_n / dacc_n the holder's root-relative total reward and discount ^
    depth; n_c / w_c the option's first move's primitive child's visits and
    root-relative value sum (INCLUSIVE of the option's); n_o / w_o the
    landing's; zv_option = r_jump + discount^len * v_jump and zv_primitive
    = r + discount * v of the primitive edge, the zero-visit composites
    (DEPARTURE 2, for OptionZero's init_q_value, 317); p_o the option's
    prior (1 - p_o the primitive's, 305-306). Returns the two scores
    (s_option, s_primitive); the option wins iff s_option > s_primitive
    (324). Q's are raw then normalised (the decomposition commutes with an
    affine normalisation); the exploration terms use sqrt(n_c), the
    PRIMITIVE CHILD's count (308, 320, 322), with v1's c_puct."""
    n_p = n_c - n_o
    q_o = torch.where(n_o > 0, (w_o / n_o.clamp(min=1) - tot_n) / dacc_n,
                      zv_option)
    q_p = torch.where(n_p > 0,
                      ((w_c - w_o) / n_p.clamp(min=1) - tot_n) / dacc_n,
                      zv_primitive)
    sq = torch.sqrt(n_c)
    s_o = normalize(q_o) + c_puct * p_o * sq / (1 + n_o)
    s_p = normalize(q_p) + c_puct * (1 - p_o) * sq / (1 + n_p)
    return s_o, s_p


class VectorizedOptionPUCT(VectorizedPUCT):
    """v1's batch of T fresh-root PUCT searches plus OptionZero's option
    edge (module docstring). One engine instance serves one search call at
    a time, as v1."""

    def __init__(self, nets, action_space, device):
        super().__init__(nets, action_space, device)
        self.last_info = None
        self.last_tree = None
        self._with_opts = False

    def option_length(self):
        """L of the prediction net's heads (1 = no option heads)."""
        return int(getattr(self.nets["prediction"], "option_length", 1))

    # ---- network wrappers (float32 in, float64 stats out) ----

    def _predict(self, latent, with_options):
        """(pmax [B] long, policy [B,A] f64, value [B] f64, option moves
        [B,L] / length [B] / p(o) [B] f64, or None x 3 without heads).
        pmax is head 0's entry of the SAME option_argmax the option's first
        move comes from (head 0 = the policy plus a -100 stop logit, so it
        is the policy's argmax unless every logit sits under -100, where it
        falls back to the policy softmax's and no option exists); without
        heads, the policy softmax's argmax."""
        if with_options:
            logits, value, _, olog = self.nets["prediction"](
                latent, with_options=True)
            probs = torch.softmax(olog, dim=2)
            arg = option_argmax(probs)
            moves, length, p = dominant_option(probs, arg=arg)
            p = p.double()
            pmax = arg[:, 0]
        else:
            logits, value, _ = self.nets["prediction"](latent)
            moves = length = p = None
            pmax = None
        pol32 = torch.softmax(logits, dim=1)
        if pmax is None:
            pmax = pol32.argmax(dim=1)
        else:
            pmax = torch.where(pmax < logits.shape[1], pmax,
                               pol32.argmax(dim=1))
        return pmax, pol32.double(), value.squeeze(-1).double(), moves, \
            length, p

    def _root_forward(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                device=self.device) / 255.0
            latent = self.nets["representation"](x)
            return (latent,) + self._predict(latent, self._with_opts)

    def _wave_forward(self, parent_latent, h, c, step, jidx, jmoves):
        """One expansion wave: B parents x A primitive rows (v1's layout,
        parent-major) followed by J jump rows (holders jidx, in order).
        With no jump row the dynamics call is v1's exactly (a [B*A] action);
        with jump rows every row is a [L] -1-padded action, a primitive row
        [a, -1, ...]. Returns a dict of per-row outputs, primitive rows
        shaped [B, A, ...], jump rows [J, ...]."""
        B = parent_latent.shape[0]
        A = self.A
        J = int(jidx.numel())
        dev = self.device
        with torch.no_grad():
            lat = parent_latent.repeat_interleave(A, dim=0)
            acts = torch.arange(A, device=dev).repeat(B)
            if J:
                lat = torch.cat([lat, parent_latent[jidx]], dim=0)
                pad = torch.full((B * A, jmoves.shape[1]), -1,
                                 dtype=torch.long, device=dev)
                pad[:, 0] = acts
                acts = torch.cat([pad, jmoves], dim=0)
            nxt, rew, _ = self.nets["dynamics"](lat, acts)
            _, pol, val, moves, length, p = self._predict(nxt, self._with_opts)
            if self.use_prefix:
                hh = h.repeat_interleave(A, dim=1)
                cc = c.repeat_interleave(A, dim=1)
                st = None if step is None else step.repeat_interleave(A)
                if J:
                    hh = torch.cat([hh, h[:, jidx]], dim=1)
                    cc = torch.cat([cc, c[:, jidx]], dim=1)
                    if st is not None:
                        st = torch.cat([st, step[jidx]])
                prefixes, _, (nh, nc) = self.nets["value_prefix"](
                    nxt, (hh.contiguous(), cc.contiguous()), step=st)
                prefixes = prefixes.squeeze(-1).double()
            else:
                prefixes = torch.zeros(B * A + J, dtype=torch.float64,
                                       device=dev)
                nh = nc = None
            rew = (rew.squeeze(-1) if rew.dim() > 1 else rew).double()
        P = B * A
        out = {
            "nxt": nxt[:P].view(B, A, *nxt.shape[1:]),
            "rew": rew[:P].view(B, A),
            "pol": pol[:P].view(B, A, A),
            "val": val[:P].view(B, A),
            "pref": prefixes[:P].view(B, A),
            "j_nxt": nxt[P:], "j_rew": rew[P:], "j_pol": pol[P:],
            "j_val": val[P:], "j_pref": prefixes[P:],
        }
        if nh is not None:
            out["nh"] = nh[:, :P].reshape(nh.shape[0], B, A, -1)
            out["nc"] = nc[:, :P].reshape(nc.shape[0], B, A, -1)
            out["j_nh"], out["j_nc"] = nh[:, P:], nc[:, P:]
        if moves is not None:
            out["moves"] = moves[:P].view(B, A, -1)
            out["length"] = length[:P].view(B, A)
            out["p"] = p[:P].view(B, A)
            out["j_moves"], out["j_length"], out["j_p"] = (
                moves[P:], length[P:], p[P:])
        return out

    # ---- the search ----

    def search(self, obs_batch, num_simulations=None, c_puct=None,
               discount=None, max_depth=None, root_noise=True,
               noise_override=None, generator=None, tie_break="random",
               option_edge=None, return_info=False, keep_tree=False):
        """Run T searches; returns (policies [T,A], values [T],
        visit_counts [T,A]) as v1, plus the info dict when return_info
        (module docstring; always kept on self.last_info). option_edge:
        None = muzero.OPTION_EDGE. Every other argument is v1's."""
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)
        assert not muzero.USE_GUMBEL_SEARCH, (
            "VectorizedOptionPUCT is PUCT-only, as v1")
        assert (muzero.MAX_DEPTH if max_depth is None else max_depth) >= 1, (
            "max_depth=0 forbids leaving the root")
        if muzero.token_edge_discounts() is not None:
            raise ValueError(
                "the option search runs without TOKEN_VOCAB: Test O's options "
                "replace the vocabulary (per-slot edge discounts are not "
                "supported here)")
        c_puct = float(muzero.C_PUCT if c_puct is None else c_puct)
        discount = float(muzero.DISCOUNT if discount is None else discount)
        max_depth = int(muzero.MAX_DEPTH if max_depth is None else max_depth)
        horizon = int(muzero.PREFIX_LSTM_HORIZON)
        floor = float(muzero.MINMAX_RANGE_FLOOR)
        self.use_prefix = bool(muzero.USE_VALUE_PREFIX_IN_TREE)
        L = self.option_length()
        self._with_opts = L > 1
        edge_on = L > 1 and bool(muzero.OPTION_EDGE if option_edge is None
                                 else option_edge)
        if edge_on:
            dyn_l = int(getattr(self.nets["dynamics"], "option_length", 1))
            if dyn_l < L:
                raise ValueError(
                    f"option heads of length {L} need a dynamics net built "
                    f"for jumps of {L} (it has option_length {dyn_l})")
        dev = self.device
        A = self.A
        T = len(obs_batch)
        N = 1 + (A + (L - 1 if edge_on else 0)) * (sims + 1)
        f64 = dict(dtype=torch.float64, device=dev)
        lng = dict(dtype=torch.long, device=dev)
        bol = dict(dtype=torch.bool, device=dev)
        disc_t = torch.tensor(discount, **f64)

        def dacc(depths):
            """discount ** depth (root-relative discount), float64."""
            return torch.pow(disc_t, depths.double())

        parent = torch.full((T, N), -1, **lng)
        children = torch.full((T, N, A), -1, **lng)
        visit = torch.zeros(T, N, **f64)
        wsum = torch.zeros(T, N, **f64)       # sum of root-relative totals
        # the node's PRIMITIVE in-edge: reward, prior, value (zero-visit Q)
        reward = torch.zeros(T, N, **f64)
        prior = torch.zeros(T, N, **f64)
        evalue = torch.zeros(T, N, **f64)
        # the node's FIRST evaluation
        evaluated = torch.zeros(T, N, **bol)
        policy = torch.zeros(T, N, A, **f64)
        lval = torch.zeros(T, N, **f64)       # the value it backs up
        tot = torch.zeros(T, N, **f64)        # root-relative total reward
        depth = torch.zeros(T, N, **lng)      # base steps from the root
        pref = torch.zeros(T, N, **f64)
        pref_len = torch.zeros(T, N, **lng)
        has_state = torch.zeros(T, N, **bol)
        opt_len = torch.zeros(T, N, **lng)    # the node's suggestion
        opt_moves = torch.full((T, N, L), -1, **lng)
        opt_p = torch.zeros(T, N, **f64)
        # the node as a HOLDER (set at its expansion, edge on)
        opt_land = torch.full((T, N), -1, **lng)
        opt_r = torch.zeros(T, N, **f64)
        opt_v = torch.zeros(T, N, **f64)
        expanded = torch.zeros(T, N, **bol)
        minv = torch.full((T,), float("inf"), **f64)
        maxv = torch.full((T,), float("-inf"), **f64)
        count = torch.ones(T, **lng)          # node 0 = root
        # counters
        n_exp = torch.zeros(T, **lng)
        len_hist = torch.zeros(T, L + 1, **lng)
        cross_sims = torch.zeros(T, **lng)
        cross_hist = torch.zeros(T, L + 1, **lng)
        leaf_sum = torch.zeros(T, **f64)
        leaf_max = torch.zeros(T, **lng)

        (root_latent, root_pmax, root_policy, root_value, r_moves,
         r_len, r_p) = self._root_forward(obs_batch)
        latent = torch.zeros(T, N, *root_latent.shape[1:],
                             dtype=torch.float32, device=dev)
        latent[:, 0] = root_latent
        evalue[:, 0] = root_value
        lval[:, 0] = root_value
        prior[:, 0] = 1.0
        evaluated[:, 0] = True
        if self._with_opts:
            opt_len[:, 0] = r_len
            opt_moves[:, 0] = r_moves
            opt_p[:, 0] = r_p
        if self.use_prefix:
            st0 = self.nets["value_prefix"].initial_state(1, dev)[0]
            Ll, H = st0.shape[0], st0.shape[-1]
            pref_h = torch.zeros(Ll, T, N, H, dtype=torch.float32, device=dev)
            pref_c = torch.zeros(Ll, T, N, H, dtype=torch.float32, device=dev)
        else:
            pref_h = pref_c = None

        if root_noise:
            if noise_override is not None:
                noise = torch.as_tensor(np.asarray(noise_override), **f64)
            else:
                alpha = float(muzero.DIRICHLET_ALPHA)
                conc = torch.full((T, A), alpha, **f64)
                noise = torch.distributions.Dirichlet(conc).sample()
            root_policy = 0.75 * root_policy + 0.25 * noise
        policy[:, 0] = root_policy     # no option noise (first-move mode)

        rows = torch.arange(T, device=dev)

        def write_eval(ft, fn, nxt, pol, val, tot_v, depth_v, pref_v,
                       plen_v, nh, nc, moves, length, p):
            """The FIRST evaluation of nodes (ft, fn)."""
            latent[ft, fn] = nxt
            policy[ft, fn] = pol
            lval[ft, fn] = val
            tot[ft, fn] = tot_v
            depth[ft, fn] = depth_v
            pref[ft, fn] = pref_v
            pref_len[ft, fn] = plen_v
            has_state[ft, fn] = self.use_prefix
            if self.use_prefix:
                pref_h[:, ft, fn] = nh
                pref_c[:, ft, fn] = nc
            if moves is not None:
                opt_moves[ft, fn] = moves
                opt_len[ft, fn] = length
                opt_p[ft, fn] = p
            evaluated[ft, fn] = True

        def expand(node_ids, mask):
            """Expand node_ids[t] where mask: all A primitive children and,
            for holders (edge on), the option's landing row."""
            if not mask.any():
                return
            tsel = rows[mask]
            nsel = node_ids[mask]
            B = int(tsel.numel())
            pl = latent[tsel, nsel]
            if self.use_prefix:
                h = pref_h[:, tsel, nsel]
                c = pref_c[:, tsel, nsel]
                reset = (~has_state[tsel, nsel]) | (
                    pref_len[tsel, nsel] >= horizon)
                init = self.nets["value_prefix"].initial_state(B, dev)
                h = torch.where(reset.view(1, -1, 1), init[0], h)
                c = torch.where(reset.view(1, -1, 1), init[1], c)
                child_len = torch.where(
                    reset, torch.ones_like(pref_len[tsel, nsel]),
                    pref_len[tsel, nsel] + 1)
            else:
                h = c = None
                reset = torch.zeros(B, **bol)
                child_len = torch.zeros(B, **lng)
            if edge_on:
                jidx = (opt_len[tsel, nsel] >= 2).nonzero().squeeze(1)
            else:
                jidx = torch.zeros(0, **lng)
            jt, jn = tsel[jidx], nsel[jidx]
            jmoves = opt_moves[jt, jn]
            out = self._wave_forward(pl, h, c, child_len, jidx, jmoves)
            tot_p = tot[tsel, nsel]
            dacc_p = dacc(depth[tsel, nsel])
            if self.use_prefix:
                prefixes = out["pref"]
                edge = torch.where(reset.unsqueeze(1), prefixes,
                                   prefixes - pref[tsel, nsel].unsqueeze(1))
            else:
                edge = out["rew"]
                prefixes = torch.zeros_like(edge)

            # primitive children: fill an existing one (a placeholder or a
            # landing), else create it
            ex = children[tsel, nsel]                              # [B,A]
            new = ex < 0
            base = count[tsel]
            ids = torch.where(new, base.unsqueeze(1)
                              + torch.cumsum(new.long(), dim=1) - 1, ex)
            count[tsel] = base + new.long().sum(dim=1)
            children[tsel, nsel] = ids
            te = tsel.unsqueeze(1).expand_as(ids)
            parent[te, ids] = nsel.unsqueeze(1).expand_as(ids)
            reward[te, ids] = edge
            prior[te, ids] = policy[tsel, nsel]     # parent's policy row
            evalue[te, ids] = out["val"]
            fill = ~evaluated[te, ids]
            if fill.any():
                write_eval(
                    te[fill], ids[fill], out["nxt"][fill], out["pol"][fill],
                    out["val"][fill],
                    (tot_p.unsqueeze(1) + dacc_p.unsqueeze(1) * edge)[fill],
                    (depth[tsel, nsel] + 1).unsqueeze(1).expand_as(ids)[fill],
                    prefixes[fill],
                    child_len.unsqueeze(1).expand_as(ids)[fill],
                    out["nh"][:, fill] if self.use_prefix else None,
                    out["nc"][:, fill] if self.use_prefix else None,
                    out["moves"][fill] if self._with_opts else None,
                    out["length"][fill] if self._with_opts else None,
                    out["p"][fill] if self._with_opts else None)

            if int(jidx.numel()):
                jl = opt_len[jt, jn]
                if self.use_prefix:
                    jedge = torch.where(reset[jidx], out["j_pref"],
                                        out["j_pref"] - pref[jt, jn])
                    jpref = out["j_pref"]
                else:
                    jedge = out["j_rew"]
                    jpref = torch.zeros_like(jedge)
                opt_r[jt, jn] = jedge
                opt_v[jt, jn] = out["j_val"]
                # walk the option's moves through the primitive children,
                # creating the placeholders the path lacks
                cur = children[jt, jn, jmoves[:, 0]]
                for i in range(1, L):
                    act = i < jl
                    if not act.any():
                        break
                    mi = jmoves[:, i].clamp(min=0)
                    nx = children[jt, cur, mi]
                    make = act & (nx < 0)
                    nid = torch.where(make, count[jt], nx)
                    count[jt] = count[jt] + make.long()
                    if make.any():
                        children[jt[make], cur[make], mi[make]] = nid[make]
                        parent[jt[make], nid[make]] = cur[make]
                    cur = torch.where(act, nid, cur)
                opt_land[jt, jn] = cur
                first = ~evaluated[jt, cur]
                if first.any():
                    write_eval(
                        jt[first], cur[first], out["j_nxt"][first],
                        out["j_pol"][first], out["j_val"][first],
                        (tot[jt, jn] + dacc(depth[jt, jn]) * jedge)[first],
                        (depth[jt, jn] + jl)[first], jpref[first],
                        child_len[jidx][first],
                        out["j_nh"][:, first] if self.use_prefix else None,
                        out["j_nc"][:, first] if self.use_prefix else None,
                        out["j_moves"][first], out["j_length"][first],
                        out["j_p"][first])
            n_exp[tsel] += 1
            len_hist[tsel, opt_len[tsel, nsel]] += 1
            expanded[tsel, nsel] = True

        def normalize(x):
            """v1's min-max normalisation at the trees' current range."""
            return minmax_normalize(x, minv, maxv, floor)

        def child_scores(node_ids):
            """[T,A] v1 PUCT scores of node_ids' primitive children, Q
            root-relative: (mean - tot(n)) / discount^depth(n) once
            visited, r + d * value before."""
            ch = children[rows, node_ids].clamp(min=0)
            v = visit.gather(1, ch)
            w = wsum.gather(1, ch)
            r = reward.gather(1, ch)
            ev = evalue.gather(1, ch)
            pr = prior.gather(1, ch)
            tp = tot[rows, node_ids].unsqueeze(1)
            dp = dacc(depth[rows, node_ids]).unsqueeze(1)
            comp = torch.where(v > 0, (w / v.clamp(min=1) - tp) / dp,
                               r + discount * ev)
            total = v.sum(dim=1, keepdim=True)
            u = c_puct * pr * torch.sqrt(total) / (1 + v)
            return normalize(comp) + u

        def option_wins(n, c, land):
            """mcts.cpp 300-325 at holders n ([T]): True where the option
            beats its first move's primitive with the option's statistics
            taken out (option_vs_primitive)."""
            s_o, s_p = option_vs_primitive(
                tot[rows, n], dacc(depth[rows, n]),
                visit[rows, c], wsum[rows, c],
                visit[rows, land], wsum[rows, land],
                opt_r[rows, n] + dacc(opt_len[rows, n]) * opt_v[rows, n],
                reward[rows, c] + discount * evalue[rows, c],
                opt_p[rows, n], c_puct, normalize)
            return s_o > s_p

        def pick(scores, gen):
            if tie_break == "first":
                return scores.argmax(dim=1)
            best = scores.max(dim=1, keepdim=True).values
            ties = (scores == best).double()
            return torch.multinomial(ties, 1, generator=gen).squeeze(1)

        def range_update(q, m):
            nonlocal minv, maxv
            minv = torch.where(m, torch.minimum(minv, q), minv)
            maxv = torch.where(m, torch.maximum(maxv, q), maxv)

        # Root expansion + range seeding (before any simulation): the
        # primitive children's zero-visit composites only, as v1 -- never
        # the root option's (audit r1).
        expand(torch.zeros(T, **lng), torch.ones(T, **bol))
        rc = children[:, 0]
        seed = reward.gather(1, rc) + discount * evalue.gather(1, rc)
        minv = torch.minimum(minv, seed.min(dim=1).values)
        maxv = torch.maximum(maxv, seed.max(dim=1).values)

        gen = generator
        for _ in range(sims):
            # ---- selection: primitive PUCT, then option vs primitive ----
            cur = torch.zeros(T, **lng)
            descending = torch.ones(T, **bol)
            crossed = torch.zeros(T, **bol)
            crossings = []
            for _d in range(max_depth):
                descending = descending & expanded[rows, cur]
                if not descending.any():
                    break
                a = pick(child_scores(cur), gen)
                c = children[rows, cur, a]
                nxt = c
                if edge_on:
                    land = opt_land[rows, cur]
                    hold = descending & (land >= 0) & (
                        a == opt_moves[rows, cur, 0])
                    if hold.any():
                        take = hold & option_wins(cur, c.clamp(min=0),
                                                  land.clamp(min=0))
                        if take.any():
                            nxt = torch.where(take, land, c)
                            crossings.append((cur.clone(), land.clone(),
                                              take))
                            crossed = crossed | take
                            cross_hist[rows[take], opt_len[rows, cur][take]] += 1
                cur = torch.where(descending, nxt, cur)
            assert bool(evaluated[rows, cur].all()), (
                "a descent ended on a placeholder")

            # ---- expansion (skip depth-capped, already-expanded leaves) ----
            expand(cur, ~expanded[rows, cur])

            # ---- backup: the leaf's root-relative total on every node ----
            dl = depth[rows, cur]
            leaf_sum += dl.double()
            leaf_max = torch.maximum(leaf_max, dl)
            cross_sims += crossed.long()
            total_v = tot[rows, cur] + dacc(dl) * lval[rows, cur]
            node = cur.clone()
            alive = torch.ones(T, **bol)
            while alive.any():
                visit[rows[alive], node[alive]] += 1
                wsum[rows[alive], node[alive]] += total_v[alive]
                par = parent[rows, node]
                nonroot = alive & (par >= 0)
                if nonroot.any():
                    pc = par.clamp(min=0)
                    q = ((wsum[rows, node] / visit[rows, node].clamp(min=1)
                          - tot[rows, pc]) / dacc(depth[rows, pc]))
                    range_update(q, nonroot & evaluated[rows, pc])
                alive = nonroot
                node = torch.where(alive, par, node)
            for hn, ln, m in crossings:
                q = ((wsum[rows, ln] / visit[rows, ln].clamp(min=1)
                      - tot[rows, hn]) / dacc(depth[rows, hn]))
                range_update(q, m)

        assert int(count.max()) <= N, "node table overflow"
        counts = visit.gather(1, children[:, 0])
        pol = (counts / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        root_q = (wsum[:, 0] / visit[:, 0].clamp(min=1)).cpu().numpy()
        info = {
            "root_option": opt_moves[:, 0].cpu().numpy(),
            "root_option_len": opt_len[:, 0].cpu().numpy(),
            "root_option_p": opt_p[:, 0].cpu().numpy(),
            "root_pmax": root_pmax.cpu().numpy(),
            "expanded_nodes": n_exp.cpu().numpy(),
            "expanded_with_option": len_hist[:, 2:].sum(dim=1).cpu().numpy(),
            "option_len_hist": len_hist.cpu().numpy(),
            "simulations": np.full(T, sims, dtype=np.int64),
            "sims_crossing_option": cross_sims.cpu().numpy(),
            "crossed_len_hist": cross_hist.cpu().numpy(),
            "leaf_depth_sum": leaf_sum.cpu().numpy(),
            "leaf_depth_max": leaf_max.cpu().numpy(),
            "option_edge": bool(edge_on),
            "option_length": L,
        }
        self.last_info = info
        self.last_tree = None
        if keep_tree:
            self.last_tree = {
                "parent": parent, "children": children, "visit": visit,
                "wsum": wsum, "reward": reward, "prior": prior,
                "evalue": evalue, "evaluated": evaluated, "policy": policy,
                "lval": lval, "tot": tot, "depth": depth, "pref": pref,
                "pref_len": pref_len, "has_state": has_state,
                "opt_len": opt_len, "opt_moves": opt_moves, "opt_p": opt_p,
                "opt_land": opt_land, "opt_r": opt_r, "opt_v": opt_v,
                "expanded": expanded, "latent": latent, "count": count,
                "minv": minv, "maxv": maxv}
        if return_info:
            return pol, root_q, counts.cpu().numpy(), info
        return pol, root_q, counts.cpu().numpy()
