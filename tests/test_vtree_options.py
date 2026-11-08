"""Test O stage 2 (CURRICULUM.md "Test O", DESIGN "Dominant option" and
"Tree", DEPARTURES 2 and 4): muzero_vtree_options.VectorizedOptionPUCT.

OptionZero cannot serve as an oracle and a tree bug gives plausible
targets, so the tree is pinned three ways:
  * hand-built cases on FAKE nets whose outputs are set per path (two-stage
    choice, inclusive counts, a tie, placeholder fill, a shared landing
    with a jump and a primitive in-edge, a landing already below a crossed
    ancestor, max_depth, root noise), with the expected numbers worked out
    in the comments;
  * an independent scalar reference written the way OptionZero's code is
    (node objects, an explicit node path pushed through the option,
    mcts.cpp 192-220 / 271-288 / 300-325), compared NODE BY NODE with the
    engine's tables (and its range, counters, PMAX and prefix steps) on
    pseudo-random fake nets -- jump-made latents with their own logits,
    values and suggestions, L 3 and 4 (two placeholders on one path), the
    prefix on and off, shared and filled-primitive landings asserted to
    occur; batch-invariant, so the lockstep T = 4 search is also compared
    exactly with per-tree runs -- and on real de-zeroed nets at T = 1
    (matched batch shapes);
  * the range seed (the root's primitives only) and edge-on = edge-off bit
    for bit when no option is crossed and jump rows evaluate as their
    primitive paths would (read (3)'s on/off probe);
  * stop-only heads and OPTION_EDGE False against v1 (VectorizedPUCT) on
    real nets, to float tolerance (statistics are root-relative, so not bit
    for bit), plus the dominant-option rule, the two-stage arithmetic
    against a transcription of mcts.cpp 300-325, and the prefix-BN step of
    every jump row. CPU only."""
import math

import numpy as np
import pytest
import torch

import muzero
from muzero_vtree import VectorizedPUCT
from muzero_vtree_options import (VectorizedOptionPUCT, dominant_option,
                                  minmax_normalize, option_vs_primitive)

CPU = torch.device("cpu")
P = 16777213          # < 2**24: fake path codes stay exact in float32


# ==========================================================================
# fake nets: outputs set per path code (or pseudo-random per code)
# ==========================================================================
def mix(*keys):
    """Deterministic hash of ints -> [0, 1)."""
    h = 0x9E3779B97F4A7C15
    for k in keys:
        h = (h ^ (int(k) + 0x632BE59BD9B4E019)) * 0xBF58476D1CE4E5B9
        h &= (1 << 64) - 1
        h ^= h >> 31
    return (h >> 11) / float(1 << 53)


def code_of(path, root=7):
    c = root
    for m in path:
        c = (c * 4 + int(m) + 1) % P
    return c


def option_head_logits(A, L, moves, p):
    """[L-1, A+1] extra-head logits whose dominant option is `moves` with
    p(o) = p (others at -5; the last move's logit solves the softmax)."""
    rows = torch.full((L - 1, A + 1), -5.0)
    k = len(moves) if len(moves) >= 2 else 0
    for i in range(1, L):
        if i < k:
            rows[i - 1, moves[i]] = (8.0 if i < k - 1 else
                                     math.log(p / (1 - p)) + math.log(A) - 5.0)
        else:
            rows[i - 1, A] = 8.0
    return rows


class FakeNets:
    """latent = [path code, made-by-a-jump flag] (float32). preds: {code or
    (code, flag): {"logits", "value", "value_jump", "option": (moves, p)}};
    rewards: {(parent code, moves tuple): r}. Unlisted codes get
    pseudo-random outputs (seeded by `seed`): with flag_dep (the default)
    a jump-made latent gets its OWN logits, value and suggestion, so a
    second evaluation overwriting a first is visible in every field (audit
    r1); p_len[k] = P(a suggestion of length k + 2). Every output depends on
    its own row only, so any batch layout gives the same numbers."""

    def __init__(self, A=3, L=3, seed=0, preds=None, rewards=None,
                 p_len=(0.35, 0.35), rscale=1.0, default=None, flag_dep=True):
        self.A, self.L, self.seed = A, L, seed
        self.preds = preds or {}
        self.rewards = rewards or {}
        self.p_len = p_len
        self.rscale = rscale
        self.default = default        # unlisted codes' spec (None: random)
        self.flag_dep = flag_dep
        self.steps = []
        outer = self

        class Rep:
            def __call__(self, x):
                codes = 7 + torch.round(x[:, 0, 0, 0] * 255).long()
                return torch.stack([codes.float(), torch.zeros(len(x))], 1)

        class Dyn:
            option_length = L

            def __call__(self, lat, act):
                return outer.dynamics(lat, act)

        class Pred:
            option_length = L

            def __call__(self, lat, with_options=False):
                return outer.prediction(lat, with_options)

        class Prefix:
            def initial_state(self, B, dev):
                z = torch.zeros(1, B, 2)
                return z, z.clone()

            def __call__(self, lat, state, step=None):
                return outer.prefix(lat, state, step)

        self.nets = {"representation": Rep(), "dynamics": Dyn(),
                     "prediction": Pred(), "value_prefix": Prefix()}

    def dynamics(self, lat, act):
        B = lat.shape[0]
        acts = act.view(B, 1) if act.dim() == 1 else act
        out = torch.zeros(B, 2)
        rew = torch.zeros(B, 1)
        for k in range(B):
            c0 = int(lat[k, 0])
            moves = tuple(int(a) for a in acts[k] if a >= 0)
            c = c0
            for m in moves:
                c = (c * 4 + m + 1) % P
            out[k, 0] = c
            out[k, 1] = 1.0 if len(moves) >= 2 else 0.0
            r = self.rewards.get((c0, moves))
            if r is None:
                r = round((mix(self.seed, 3, c0, *moves) - 0.5) * 2, 3) \
                    * self.rscale
            rew[k, 0] = r
        return out, rew, None

    def spec(self, code, flag):
        s = self.preds.get((code, flag), self.preds.get(code))
        if s is not None:
            return s
        if self.default is not None:
            return self.default
        A = self.A
        f = int(flag) if self.flag_dep else 0
        logits = [(mix(self.seed, 1, f, code, j) - 0.5) * 3 for j in range(A)]
        best = int(np.argmax(logits))
        r, acc, n = mix(self.seed, 50, f, code), 0.0, 0
        for k, pk in enumerate(self.p_len):
            acc += pk
            if r < acc:
                n = k + 2
                break
        n = min(n, self.L)
        moves = [best] + [int(mix(self.seed, 60 + i, f, code) * A)
                          for i in range(1, n)]
        return {"logits": logits,
                "value": (mix(self.seed, 99 - f, code) - 0.5) * 4,
                "option": (tuple(moves),
                           0.3 + 0.6 * mix(self.seed, 51, f, code))
                if n >= 2 else None}

    def prediction(self, lat, with_options):
        B, A, L = lat.shape[0], self.A, self.L
        logits = torch.zeros(B, A)
        value = torch.zeros(B, 1)
        olog = torch.zeros(B, L, A + 1)
        for k in range(B):
            code, flag = int(lat[k, 0]), int(lat[k, 1])
            s = self.spec(code, flag)
            logits[k] = torch.tensor(s.get("logits", [0.0] * A))
            v = s.get("value_jump") if flag and "value_jump" in s else \
                s.get("value", 0.0)
            value[k, 0] = v
            olog[k, 0, :A] = logits[k]
            olog[k, 0, A] = -100.0
            opt = s.get("option")
            if L > 1:
                olog[k, 1:] = option_head_logits(
                    A, L, opt[0] if opt else (), opt[1] if opt else 0.5)
        if with_options:
            return logits, value, None, olog
        return logits, value, None

    def prefix(self, lat, state, step):
        h, _ = state
        self.steps.append(None if step is None else
                          [int(s) for s in torch.as_tensor(step).reshape(-1)])
        g = torch.tensor([mix(self.seed, 31, int(lat[k, 0]), int(lat[k, 1]))
                          - 0.5 for k in range(lat.shape[0])])
        new0 = h[0, :, 0] + g
        new1 = h[0, :, 1] + 1
        nh = torch.stack([new0, new1], dim=1).unsqueeze(0)
        return new0.unsqueeze(1), None, (nh, nh.clone())


def fake_obs(T):
    obs = np.zeros((T, 12, 84, 84), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(T)
    return obs


@pytest.fixture
def knobs(monkeypatch):
    for key, val in (("USE_GUMBEL_SEARCH", False), ("TOKEN_VOCAB", None),
                     ("USE_VALUE_PREFIX_IN_TREE", False),
                     ("MINMAX_RANGE_FLOOR", 0.01), ("MAX_DEPTH", 12),
                     ("PREFIX_LSTM_HORIZON", 3), ("OPTION_EDGE", True)):
        monkeypatch.setattr(muzero, key, val)
    return monkeypatch


def node_at(tree, t, path):
    nid = 0
    for m in path:
        nid = int(tree["children"][t, nid, m])
        if nid < 0:
            return -1
    return nid


def run(fake, sims, T=1, **kw):
    eng = VectorizedOptionPUCT(fake.nets, fake.A, CPU)
    kw.setdefault("root_noise", False)
    kw.setdefault("tie_break", "first")
    pol, val, counts, info = eng.search(fake_obs(T), num_simulations=sims,
                                        return_info=True, keep_tree=True,
                                        **kw)
    return eng.last_tree, pol, val, counts, info


# ==========================================================================
# the dominant option (zero_actor.cpp 368-381, 395-396)
# ==========================================================================
def probs_from(argmaxes, A=4, top=0.6):
    """[1, L, A+1] heads whose argmaxes are `argmaxes` (A = stop)."""
    L = len(argmaxes)
    p = torch.full((1, L, A + 1), (1 - top) / A)
    for i, m in enumerate(argmaxes):
        p[0, i, m] = top
    return p


def test_each_heads_argmax_in_turn_to_the_first_stop():
    moves, length, p = dominant_option(probs_from([2, 2, 1]))
    assert moves.tolist() == [[2, 2, 1]] and length.tolist() == [3]
    moves, length, p = dominant_option(probs_from([3, 0, 4]))   # 4 = stop
    assert moves.tolist() == [[3, 0, -1]] and length.tolist() == [2]


def test_length_under_two_is_no_option():
    moves, length, p = dominant_option(probs_from([1, 4, 2]))
    assert length.tolist() == [0] and moves.tolist() == [[-1, -1, -1]]
    assert p.tolist() == [0.0]
    # L = 1 (the policy head alone) never holds an option
    moves, length, p = dominant_option(probs_from([1]))
    assert length.tolist() == [0]
    # head 0 at stop (a policy below -100 everywhere) is no option either
    moves, length, _ = dominant_option(probs_from([4, 1, 1]))
    assert length.tolist() == [0]


def test_a_move_after_the_first_stop_is_ignored():
    moves, length, _ = dominant_option(probs_from([0, 4, 3, 3], A=4))
    assert length.tolist() == [0]
    moves, length, _ = dominant_option(probs_from([0, 1, 4, 3], A=4))
    assert length.tolist() == [2] and moves.tolist() == [[0, 1, -1, -1]]


def test_p_is_the_last_heads_probability_not_a_product():
    pr = probs_from([2, 1, 3])
    pr[0, 1, 1], pr[0, 2, 3] = 0.7, 0.45
    _, length, p = dominant_option(pr)
    assert length.tolist() == [3]
    assert p.item() == pytest.approx(0.45)
    pr2 = probs_from([2, 1, 4])
    pr2[0, 1, 1] = 0.55
    _, length, p = dominant_option(pr2)
    assert length.tolist() == [2] and p.item() == pytest.approx(0.55)


def test_ties_go_to_the_first_index_as_max_element():
    pr = torch.tensor([[[0.1, 0.4, 0.4, 0.1, 0.0],
                        [0.3, 0.0, 0.0, 0.3, 0.3],      # move 0 before 3 / stop
                        [0.0, 0.0, 0.0, 0.5, 0.5]]])    # move 3 before stop
    moves, length, p = dominant_option(pr)
    assert moves.tolist() == [[1, 0, 3]] and p.item() == pytest.approx(0.5)


def test_batch_rows_are_independent():
    pr = torch.cat([probs_from([1, 1, 1]), probs_from([1, 4, 1]),
                    probs_from([0, 2, 4])])
    moves, length, _ = dominant_option(pr)
    assert length.tolist() == [3, 0, 2]
    assert moves.tolist() == [[1, 1, 1], [-1, -1, -1], [0, 2, -1]]


# ==========================================================================
# the two-stage arithmetic (mcts.cpp 300-325)
# ==========================================================================
def transcribed(tot, dacc, n_c, w_c, n_o, w_o, zv_o, zv_p, p_o, c, lo, hi,
                floor=0.01):
    """mcts.cpp selectOptionChild, line for line, with our departures
    marked: normalisation is v1's (applied to each mean), c is v1's c_puct
    (for puct_bias), and a zero count takes r + d * value (DEPARTURE 2,
    for init_q_value)."""
    def norm(x):
        return (x - lo) / max(hi - lo, floor) if hi > lo else x
    total_simulation = n_c                                        # 308
    total_mean = norm((w_c / n_c - tot) / dacc) if total_simulation > 0 \
        else 0.0                                                  # 309
    option_count = n_o                                            # 311
    option_mean = norm((w_o / n_o - tot) / dacc) if option_count > 0 \
        else 0.0                                                  # 312
    primitive_count = total_simulation - option_count             # 314
    primitive_mean = ((total_mean * total_simulation
                       - option_mean * option_count) / primitive_count
                      if primitive_count > 0 else 0.0)            # 315
    u_o = c * p_o * math.sqrt(total_simulation) / (1 + option_count)
    q_o = norm(zv_o) if option_count == 0 else option_mean        # 321
    u_p = c * (1 - p_o) * math.sqrt(total_simulation) / (1 + primitive_count)
    q_p = norm(zv_p) if primitive_count == 0 else primitive_mean  # 323
    return u_o + q_o, u_p + q_p


def engine_scores(tot, dacc, n_c, w_c, n_o, w_o, zv_o, zv_p, p_o, c, lo, hi,
                  floor=0.01):
    f = lambda x: torch.tensor([float(x)], dtype=torch.float64)  # noqa: E731
    lo_t, hi_t = f(lo), f(hi)
    s_o, s_p = option_vs_primitive(
        f(tot), f(dacc), f(n_c), f(w_c), f(n_o), f(w_o), f(zv_o), f(zv_p),
        f(p_o), c, lambda x: minmax_normalize(x, lo_t, hi_t, floor))
    return s_o.item(), s_p.item()


def test_two_stage_arithmetic_matches_the_transcription():
    rng = np.random.default_rng(3)
    for _ in range(400):
        n_c = int(rng.integers(0, 9))
        n_o = int(rng.integers(0, n_c + 1))
        tot, dacc = rng.normal(), 0.9 ** int(rng.integers(0, 5))
        w_o = n_o * (tot + dacc * rng.normal())
        w_c = w_o + (n_c - n_o) * (tot + dacc * rng.normal())
        lo, hi = sorted(rng.normal(size=2))
        if rng.random() < 0.2:
            lo, hi = np.inf, -np.inf          # no range yet: identity
        args = (tot, dacc, n_c, w_c, n_o, w_o, rng.normal(), rng.normal(),
                rng.uniform(0.25, 1.0), 1.25, lo, hi)
        a, b = transcribed(*args), engine_scores(*args)
        assert a[0] == pytest.approx(b[0], rel=1e-9, abs=1e-9)
        assert a[1] == pytest.approx(b[1], rel=1e-9, abs=1e-9)


def test_the_primitive_side_takes_the_options_statistics_out():
    """The primitive child's 4 visits average 0.5 INCLUDING the option's 2
    at 0.8; without them the primitive is at 0.2. Range [0, 1], p(o) 0.25:
    option 0.8 + 1.25*0.25*2/3 = 1.008 > primitive 0.2 + 1.25*0.75*2/3 =
    0.825 -- scoring the primitive on its inclusive 0.5 (1.125) would
    flip it."""
    args = (0.0, 1.0, 4, 2.0, 2, 1.6, 0.0, 0.0, 0.25, 1.25, 0.0, 1.0)
    s_o, s_p = engine_scores(*args)
    assert s_o == pytest.approx(0.8 + 1.25 * 0.25 * 2 / 3)
    assert s_p == pytest.approx(0.2 + 1.25 * 0.75 * 2 / 3)
    assert s_o > s_p
    inclusive = 0.5 + 1.25 * 0.75 * 2 / 3
    assert inclusive > s_o                    # the wrong reading would flip


def test_zero_counts_use_r_plus_d_value_and_sqrt_reads_the_primitive():
    # nothing visited: sqrt(0) kills both bonuses, the zero-visit
    # composites decide (DEPARTURE 2)
    s_o, s_p = engine_scores(0.0, 1.0, 0, 0.0, 0, 0.0, 0.7, 0.6, 0.9, 1.25,
                             0.0, 1.0)
    assert (s_o, s_p) == pytest.approx((0.7, 0.6))
    # every primitive visit came through the option: n_p = 0 -> the edge's
    # own r + d * value, bonus over sqrt(n_c) = 2
    s_o, s_p = engine_scores(0.0, 1.0, 4, 2.0, 4, 2.0, 9.0, 0.3, 0.75, 1.0,
                             -10.0, 10.0)
    assert s_o == pytest.approx((0.5 + 10) / 20 + 0.75 * 2 / 5)
    assert s_p == pytest.approx((0.3 + 10) / 20 + 0.25 * 2 / 1)


def test_an_exact_tie_goes_to_the_primitive():
    s_o, s_p = engine_scores(0.0, 1.0, 0, 0.0, 0, 0.0, 0.5, 0.5, 0.6, 1.25,
                             0.0, 1.0)
    assert s_o == s_p          # and the engine takes `s_o > s_p` (324)


# ==========================================================================
# hand-built trees on fake nets
# ==========================================================================
def test_two_stage_choice_by_hand(knobs):
    """d 0.9, c 1.25, uniform priors. Root option (0, 0), p(o) 0.9; v[0] 1,
    the landing [0,0] 2 (by the jump), every reward 0, other values 0.
    Seed range (the primitives only): c0 0.9, c1 c2 0 -> [0, 0.9]; the
    option's 0 + 0.81 * 2 = 1.62 is NOT seeded.
    Sim 1: all unvisited, u = 0, c0 tops (1.0) -> option vs primitive at
    zero visits: norm(1.62) = 1.8 > norm(0.9) = 1 -> JUMP to the landing
    (a first-visit crossing: n_c = 0, so p(o) plays no part),
    expand it, back up 1.62 on root, [0] and [0,0]; the range takes c0's
    1.62, the landing's 1.8 from [0] and the jump's 1.62 -> [0, 1.8].
    Sim 2: c0 0.9 + 0.208 > c1 0.417 -> c0; n_c = n_o = 1, n_p = 0:
    option 0.9 + 1.25*0.9/2 = 1.4625 > primitive norm(0.9) = 0.5 +
    1.25*0.1 = 0.625 -> JUMP; the landing is expanded, its children tie at
    0 -> move 0 -> expand [0,0,0], back up 0. [0] holds 2 visits and was
    never expanded: its count is the option's (inclusive)."""
    fake = FakeNets(A=3, L=3, rscale=0.0, default={"value": 0.0}, preds={
        code_of([]): {"logits": [0, 0, 0], "value": 0.0,
                      "option": ((0, 0), 0.9)},
        code_of([0]): {"value": 1.0},
        code_of([0, 0]): {"value": 2.0, "value_jump": 2.0},
    })
    tree, pol, val, counts, info = run(fake, 2, discount=0.9, c_puct=1.25)
    V = lambda p: int(tree["visit"][0, node_at(tree, 0, p)])  # noqa: E731
    assert [V([]), V([0]), V([0, 0]), V([0, 0, 0])] == [2, 2, 2, 1]
    assert counts.tolist() == [[2, 0, 0]]
    assert not bool(tree["expanded"][0, node_at(tree, 0, [0])])
    assert bool(tree["expanded"][0, node_at(tree, 0, [0, 0])])
    assert val[0] == pytest.approx(0.81, abs=1e-12)
    assert info["sims_crossing_option"].tolist() == [2]
    assert info["crossed_len_hist"].tolist() == [[0, 0, 2, 0]]
    assert info["expanded_nodes"].tolist() == [3]
    assert info["root_option"].tolist() == [[0, 0, -1]]
    assert info["root_option_p"][0] == pytest.approx(0.9, abs=1e-6)
    assert info["leaf_depth_max"].tolist() == [3]


def test_a_tie_between_option_and_primitive_takes_the_primitive(knobs):
    """d 0.5 (exact): primitive c0 = 0 + 0.5 * 1.0, option = 0.25 + 0.25 *
    1.0 -- both 0.5 at zero visits, bonuses 0: the primitive (strict >)."""
    fake = FakeNets(A=3, L=3, preds={
        code_of([]): {"option": ((0, 0), 0.9)},
        code_of([0]): {"value": 1.0},
        code_of([0, 0]): {"value_jump": 1.0, "value": 1.0},
        code_of([1]): {"value": -1.0}, code_of([2]): {"value": -1.0},
    }, rewards={(code_of([]), (0, 0)): 0.25, (code_of([]), (0,)): 0.0},
        rscale=0.0, default={"value": 0.0})
    tree, _, _, _, info = run(fake, 1, discount=0.5, c_puct=1.25)
    assert info["sims_crossing_option"].tolist() == [0]
    assert bool(tree["expanded"][0, node_at(tree, 0, [0])])


def test_shared_landing_a_jump_then_its_primitive_in_edge(knobs):
    """Root option (0, 0): its landing [0,0] is evaluated in the ROOT's
    wave (by the jump: flag 1, value 0.5, r_jump 0.25). Sim 1 takes the
    primitive (zero-visit 0.9 > 0.25 + 0.81 * 0.5 = 0.655) and expands
    [0]: the landing becomes [0]'s primitive child too. It keeps its first
    evaluation (latent, policy row, value 0.5, root-relative total 0.25,
    depth 2, the jump row's suggestion) and takes only the primitive
    edge's fields (r 0.5, value -2.0, prior [0]'s policy row); the root
    keeps the jump's r and value as the holder."""
    root, c0, land = code_of([]), code_of([0]), code_of([0, 0])
    fake = FakeNets(A=3, L=3, preds={
        root: {"option": ((0, 0), 0.5)},
        c0: {"value": 1.0, "logits": [0.0, 1.0, 0.0]},
        (land, 1): {"value": 0.5, "logits": [2.0, 0.0, 0.0],
                    "option": ((0, 2), 0.6)},
        (land, 0): {"value": -2.0, "logits": [0.0, 0.0, 2.0]},
        code_of([1]): {"value": 0.0}, code_of([2]): {"value": 0.0},
    }, rewards={(root, (0, 0)): 0.25, (c0, (0,)): 0.5}, rscale=0.0,
        default={"value": 0.0})
    tree, _, _, _, info = run(fake, 1, discount=0.9, c_puct=1.25)
    L_id = node_at(tree, 0, [0, 0])
    c0_id = node_at(tree, 0, [0])
    assert int(tree["opt_land"][0, 0]) == L_id
    assert int(tree["parent"][0, L_id]) == c0_id
    assert bool(tree["expanded"][0, c0_id]) and not bool(tree["expanded"][0, L_id])
    assert info["sims_crossing_option"].tolist() == [0]
    # first evaluation kept (the jump's)
    assert tree["latent"][0, L_id].tolist() == [float(land), 1.0]
    assert tree["lval"][0, L_id].item() == 0.5
    assert tree["tot"][0, L_id].item() == pytest.approx(0.25, abs=1e-12)
    assert int(tree["depth"][0, L_id]) == 2
    pol_jump = torch.softmax(torch.tensor([2.0, 0.0, 0.0]), 0).double()
    assert torch.allclose(tree["policy"][0, L_id], pol_jump, rtol=0,
                          atol=1e-7)
    assert tree["opt_moves"][0, L_id].tolist() == [0, 2, -1]
    # its primitive in-edge's own fields
    assert tree["reward"][0, L_id].item() == 0.5
    assert tree["evalue"][0, L_id].item() == -2.0
    pol_c0 = torch.softmax(torch.tensor([0.0, 1.0, 0.0]), 0).double()
    assert tree["prior"][0, L_id].item() == pytest.approx(pol_c0[0].item(),
                                                          abs=1e-7)
    # the holder's jump fields
    assert tree["opt_r"][0, 0].item() == 0.25
    assert tree["opt_v"][0, 0].item() == 0.5
    assert int(tree["opt_len"][0, 0]) == 2


def scenario3():
    """Root option (0,0,0) p 0.3 -> landing [0,0,0]; [0]'s option (0,0)
    p 0.5 -> the SAME node. d 0.9, c 1.25, root priors [.909, .045, .045].
    Rewards: root jump 0.3, [0]'s jump 0.2, primitive ([0,0], 0) 0.1, every
    other 0. Values: [0] 1, [1] [2] -5, the landing 3 by a jump and -1 by a
    primitive row, the rest 0.
    Seed range (the primitives only): [-4.5, 0.9].
    Sim 1: root zero-visit, c0 tops; option 0.3 + 0.729*3 = 2.487 >
      primitive 0.9 -> JUMP over the placeholder [0,0]; expand the landing;
      back up 2.487 (root, [0], [0,0], landing). Range [-4.5, 2.763]
      (the placeholder's Q from [0] 2.487/0.9; the landing's from the
      placeholder is skipped: it has no evaluation).
    Sim 2: c0; n_c = n_o = 1, n_p = 0: option 0.962 + 1.25*0.3/2 = 1.15 <
      primitive 0.744 + 1.25*0.7 = 1.62 -> PRIMITIVE: expand [0]. The
      placeholder [0,0] is FILLED (keeps 1 visit, 2.487); [0]'s jump lands
      on the existing, expanded landing, which keeps its first evaluation.
      Back up 0.9.
    Sim 3: c0; n_c 2, n_o 1, n_p 1: option 0.962 + 0.265 = 1.227 <
      primitive norm(0.9) 0.744 + 0.619 = 1.362 -> primitive; [0] is
      expanded: [0,0] (Q 2.763 -> 1.0, +0.208) beats [0,1] (0.620 + 0.417)
      -> [0]'s option vs [0,0]: n_c = n_o = 1 (the landing's visit is the
      root's jump's): option 1.0 + 1.25*0.5/2 = 1.3125 > [0,0]'s zero-visit
      composite r([0], 0) + 0.9 * 0 = 0 -> 0.620 + 0.625 = 1.245 -> JUMP
      from [0]; the landing is expanded, its children tie -> move 0 ->
      expand [0,0,0,0]; back up 0.3. (Rewards pass a float32 tensor, so
      0.3 / 0.2 / 0.1 are their float32 roundings: tolerances 1e-6.)"""
    root = code_of([])
    c0, c00, land = code_of([0]), code_of([0, 0]), code_of([0, 0, 0])
    preds = {
        root: {"logits": [3.0, 0.0, 0.0], "option": ((0, 0, 0), 0.3)},
        c0: {"value": 1.0, "option": ((0, 0), 0.5)},
        code_of([1]): {"value": -5.0}, code_of([2]): {"value": -5.0},
        c00: {"value": 0.0},
        (land, 1): {"value": 3.0}, (land, 0): {"value": -1.0},
    }
    rewards = {(root, (0, 0, 0)): 0.3, (c0, (0, 0)): 0.2, (c00, (0,)): 0.1}
    return FakeNets(A=3, L=3, preds=preds, rewards=rewards, rscale=0.0,
                    default={"value": 0.0}), (root, c0, c00, land)


def test_landing_below_a_crossed_ancestor_and_placeholder_fill(knobs):
    fake, (_, _, c00, land) = scenario3()
    tree, _, val, counts, info = run(fake, 1, discount=0.9, c_puct=1.25)
    P00, PL = node_at(tree, 0, [0, 0]), node_at(tree, 0, [0, 0, 0])
    # sim 1: the placeholder sits on the path, unevaluated, with the visit
    assert not bool(tree["evaluated"][0, P00])
    assert int(tree["visit"][0, P00]) == 1
    assert tree["wsum"][0, P00].item() == pytest.approx(2.487, abs=1e-6)
    assert (tree["children"][0, P00] >= 0).tolist() == [True, False, False]
    assert int(tree["opt_land"][0, 0]) == PL
    assert bool(tree["expanded"][0, PL])
    assert info["sims_crossing_option"].tolist() == [1]

    tree, _, val, counts, info = run(fake, 2, discount=0.9, c_puct=1.25)
    C0 = node_at(tree, 0, [0])
    P00, PL = node_at(tree, 0, [0, 0]), node_at(tree, 0, [0, 0, 0])
    assert bool(tree["expanded"][0, C0])
    # the placeholder is filled and keeps its statistics
    assert bool(tree["evaluated"][0, P00])
    assert int(tree["visit"][0, P00]) == 1
    assert tree["wsum"][0, P00].item() == pytest.approx(2.487, abs=1e-6)
    assert tree["latent"][0, P00].tolist() == [float(c00), 0.0]
    assert tree["tot"][0, P00].item() == 0.0       # r([0], 0) = 0
    assert not bool(tree["expanded"][0, P00])
    # [0]'s option lands on the root's landing; first evaluation kept
    assert int(tree["opt_land"][0, C0]) == PL
    assert tree["latent"][0, PL].tolist() == [float(land), 1.0]
    assert tree["tot"][0, PL].item() == pytest.approx(0.3, abs=1e-6)
    assert int(tree["depth"][0, PL]) == 3
    assert tree["lval"][0, PL].item() == 3.0
    assert tree["opt_r"][0, C0].item() == pytest.approx(0.2, abs=1e-7)
    assert tree["opt_v"][0, C0].item() == 3.0
    assert tree["opt_r"][0, 0].item() == pytest.approx(0.3, abs=1e-7)
    assert info["sims_crossing_option"].tolist() == [1]
    assert counts.tolist() == [[2, 0, 0]]

    tree, _, val, counts, info = run(fake, 3, discount=0.9, c_puct=1.25)
    V = lambda p: int(tree["visit"][0, node_at(tree, 0, p)])  # noqa: E731
    assert [V([]), V([0]), V([0, 0]), V([0, 0, 0]), V([0, 0, 0, 0])] == \
        [3, 3, 2, 2, 1]
    assert V([0, 1]) == 0 and V([0, 2]) == 0
    assert info["sims_crossing_option"].tolist() == [2]
    assert info["crossed_len_hist"].tolist() == [[0, 0, 1, 1]]
    assert val[0] == pytest.approx((2.487 + 0.9 + 0.3) / 3, abs=1e-6)
    # the landing has three in-edges now: the root's jump, [0]'s jump and
    # its primitive edge from [0,0] (unexpanded, so not yet filled)
    assert int(tree["parent"][0, node_at(tree, 0, [0, 0, 0])]) == \
        node_at(tree, 0, [0, 0])


def test_a_jump_is_one_selection_step_under_max_depth(knobs):
    """Scenario 1's root option at max_depth 1: sim 1 jumps (one step) to
    a depth-2 leaf and expands it; sim 2 stops after the jump on that
    expanded landing (the cap), expands nothing and backs up its value."""
    fake = FakeNets(A=3, L=3, preds={
        code_of([]): {"option": ((0, 0), 0.9)},
        code_of([0]): {"value": 1.0},
        code_of([0, 0]): {"value": 2.0, "value_jump": 2.0}},
        rscale=0.0, default={"value": 0.0})
    tree, _, val, counts, info = run(fake, 2, discount=0.9, c_puct=1.25,
                                     max_depth=1)
    assert info["leaf_depth_max"].tolist() == [2]
    assert info["expanded_nodes"].tolist() == [2]
    assert info["sims_crossing_option"].tolist() == [2]
    assert int(tree["visit"][0, node_at(tree, 0, [0, 0])]) == 2
    assert val[0] == pytest.approx(1.62, abs=1e-12)


def test_no_option_noise_at_the_root(knobs):
    """First-move mode: the root's primitive priors take v1's Dirichlet mix,
    its p(o) stays the head's (zero_actor.cpp 265)."""
    fake = FakeNets(A=3, L=3, preds={code_of([]): {"option": ((0, 1), 0.8)}},
                    default={"value": 0.0})
    noise = np.array([[0.7, 0.2, 0.1]])
    tree, _, _, _, info = run(fake, 3, root_noise=True, noise_override=noise)
    assert tree["opt_p"][0, 0].item() == pytest.approx(0.8, abs=1e-6)
    want = 0.75 * torch.full((3,), 1 / 3).double() + 0.25 * torch.tensor(
        noise[0])
    for a in range(3):
        cid = node_at(tree, 0, [a])
        assert tree["prior"][0, cid].item() == pytest.approx(want[a].item(),
                                                             abs=1e-12)


def test_edge_off_keeps_the_suggestions_and_builds_no_jump(knobs):
    fake, _ = scenario3()
    tree, _, _, _, info = run(fake, 6, discount=0.9, c_puct=1.25,
                              option_edge=False)
    assert info["option_edge"] is False
    assert info["root_option"].tolist() == [[0, 0, 0]]
    assert info["root_option_len"].tolist() == [3]
    assert info["root_option_p"][0] == pytest.approx(0.3, abs=1e-6)
    assert (tree["opt_land"] < 0).all()
    assert info["sims_crossing_option"].tolist() == [0]
    assert info["expanded_with_option"][0] >= 1     # nodes still suggest one
    n = int(tree["count"][0])
    assert bool(tree["evaluated"][0, :n].all())     # no placeholder anywhere
    knobs.setattr(muzero, "OPTION_EDGE", False)     # the knob's default path
    _, _, _, _, info2 = run(fake, 6, discount=0.9, c_puct=1.25)
    assert info2["option_edge"] is False


def test_the_root_option_is_not_in_the_range_seed(knobs):
    """The range is seeded from the root's PRIMITIVE children only (v1;
    audit r1). Root option (0, 0) whose jump composite is 0.81 * -50 =
    -40.5: zero-visit primitive c0 0.9 beats it, so sim 1 expands [0] and
    backs up 0.9 -- the range stays [0, 0.9] (c1 c2 0, c0 0.9)."""
    fake = FakeNets(A=3, L=3, rscale=0.0, default={"value": 0.0}, preds={
        code_of([]): {"option": ((0, 0), 0.9)},
        code_of([0]): {"value": 1.0},
        (code_of([0, 0]), 1): {"value": -50.0}})
    tree, _, _, _, info = run(fake, 1, discount=0.9, c_puct=1.25)
    assert info["sims_crossing_option"].tolist() == [0]
    assert tree["opt_v"][0, 0].item() == -50.0          # the root holds it
    assert tree["minv"][0].item() == 0.0
    assert tree["maxv"][0].item() == pytest.approx(0.9, abs=1e-12)


def never_cross(monkeypatch):
    import muzero_vtree_options as mvo
    real = mvo.option_vs_primitive

    def scores(*args, **kw):
        s_o, s_p = real(*args, **kw)
        return torch.full_like(s_o, -math.inf), s_p

    monkeypatch.setattr(mvo, "option_vs_primitive", scores)


def test_edge_on_is_edge_off_when_no_option_is_crossed(knobs):
    """Read (3)'s on/off probe counts every difference as the edge's: with
    jump rows that evaluate exactly as the primitive path would (a
    flag-independent spec, every reward 0) and no option ever crossed, the
    edge-on search must equal the edge-off one bit for bit -- landings,
    placeholders and jump rows built and never used change nothing. (A
    root option composite in the range seed broke this.)"""
    never_cross(knobs)
    outside = landed = 0
    for seed in range(8):
        fake = FakeNets(A=3, L=3, seed=seed, flag_dep=False, rscale=0.0,
                        p_len=(0.45, 0.45))
        on = run(fake, 30, T=2, discount=0.9, c_puct=1.25, option_edge=True)
        off = run(fake, 30, T=2, discount=0.9, c_puct=1.25, option_edge=False)
        tree, _, v_on, c_on, i_on = on
        tree_off, _, v_off, c_off, _ = off
        assert i_on["sims_crossing_option"].tolist() == [0, 0]
        assert c_on.tolist() == c_off.tolist(), seed
        assert v_on.tolist() == v_off.tolist(), seed
        assert torch.equal(tree["minv"], tree_off["minv"]), seed
        assert torch.equal(tree["maxv"], tree_off["maxv"]), seed
        landed += int((tree["opt_land"] >= 0).sum())
        for t in range(2):
            if int(tree["opt_land"][t, 0]) >= 0:
                rc = tree["children"][t, 0]
                comp = tree["reward"][t, rc] + 0.9 * tree["evalue"][t, rc]
                oc = tree["opt_r"][t, 0] + 0.9 ** int(tree["opt_len"][t, 0]) \
                    * tree["opt_v"][t, 0]
                outside += bool(oc < comp.min() or oc > comp.max())
    assert landed > 0, "potency: no option edge was built"
    assert outside > 0, "potency: no root option fell outside the seed"


@pytest.mark.parametrize("p", [0.26, 0.95])
def test_an_option_is_crossed_on_its_first_visit_whatever_its_prior(knobs, p):
    """DEPARTURE 2 as registered (kept, coordinator 2026-11-08): at n_c = 0
    both bonuses vanish, the zero-visit composites decide, so the option
    is crossed on its first move's very first visit whatever p(o) is.
    OptionZero's init_q_value tie would send that visit to the primitive."""
    fake = FakeNets(A=3, L=3, rscale=0.0, default={"value": 0.0}, preds={
        code_of([]): {"option": ((0, 0), p)},
        code_of([0]): {"value": 1.0},
        code_of([0, 0]): {"value": 2.0}})
    tree, _, _, _, info = run(fake, 1, discount=0.9, c_puct=1.25)
    assert info["sims_crossing_option"].tolist() == [1]
    assert int(tree["visit"][0, node_at(tree, 0, [0, 0])]) == 1
    assert not bool(tree["expanded"][0, node_at(tree, 0, [0])])


def test_root_pmax_is_the_options_first_move_on_a_near_tie(knobs):
    """Logits 0 and 1e-8 round to one float32 probability: the logits'
    argmax says 1, head 0's softmax argmax (the option's first move) says
    0. root_pmax must read the same argmax as the option (audit r1)."""
    logits = [0.0, 1e-8, -1.0]
    assert torch.tensor(logits).argmax().item() == 1
    probs = torch.softmax(torch.tensor(logits + [-100.0]), 0)
    assert probs[0] == probs[1]
    fake = FakeNets(A=3, L=3, rscale=0.0, default={"value": 0.0}, preds={
        code_of([]): {"logits": logits, "option": ((0, 1), 0.8)}})
    _, _, _, _, info = run(fake, 2, discount=0.9, c_puct=1.25)
    assert info["root_option"].tolist() == [[0, 1, -1]]
    assert info["root_pmax"].tolist() == [0]


# ==========================================================================
# the scalar reference, OptionZero-shaped
# ==========================================================================
class RNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.count, self.wsum = 0, 0.0
        self.r = self.ev = self.prior = 0.0            # primitive in-edge
        self.evaluated = self.expanded = self.has_state = False
        self.latent = self.policy = self.h = self.c = None
        self.value = self.total = self.pref = 0.0
        self.depth = self.pref_len = 0
        self.option, self.p_o = (), 0.0
        self.landing, self.opt_r, self.opt_v = None, 0.0, 0.0


def ref_option(head_probs, A):
    """calculateOption + setOptionInfo's p, on python lists."""
    moves = []
    for head in head_probs:
        m = head.index(max(head))
        if m == A:
            break
        moves.append(m)
    if len(moves) <= 1:
        return (), 0.0
    return tuple(moves), float(head_probs[len(moves) - 1][moves[-1]])


class Ref:
    """The option search written the way OptionZero's code is: node
    objects, an explicit node path (the option's nodes pushed one by one,
    mcts.cpp 203-208), the root-relative backup over that path (271-288),
    selectOptionChild transcribed (300-325) -- with the registration's
    departures (eager expansion with the landing row, r + d * value at
    zero visits, holder semantics, v1's PUCT and range: seeded from the
    root's primitive children only, plus every crossed jump's Q)."""

    def __init__(self, nets, A, L, discount, c_puct, max_depth, edge,
                 use_prefix, horizon, floor=0.01):
        self.nets, self.A, self.L = nets, A, L
        self.d, self.c, self.max_depth = discount, c_puct, max_depth
        self.edge, self.use_prefix, self.horizon = edge, use_prefix, horizon
        self.floor = floor
        self.lo, self.hi = math.inf, -math.inf
        self.stats = {"expanded": 0, "len_hist": [0] * (L + 1),
                      "cross_sims": 0, "cross_hist": [0] * (L + 1),
                      "leaf_sum": 0.0, "leaf_max": 0,
                      "path_placeholders": 0}

    def norm(self, x):
        if self.hi > self.lo:
            return (x - self.lo) / max(self.hi - self.lo, self.floor)
        return x

    def upd(self, q):
        self.lo, self.hi = min(self.lo, q), max(self.hi, q)

    def dacc(self, depth):
        return self.d ** depth

    def write(self, node, nxt, pol, val, total, depth, pref, plen, h, c,
              probs):
        node.latent, node.policy, node.value = nxt, pol.tolist(), val
        node.total, node.depth = total, depth
        node.pref, node.pref_len, node.h, node.c = pref, plen, h, c
        node.has_state = self.use_prefix
        node.option, node.p_o = ref_option(probs.tolist(), self.A)
        node.evaluated = True

    def expand(self, n):
        A, L = self.A, self.L
        jump = self.edge and len(n.option) >= 2
        rows = A + (1 if jump else 0)
        lat = n.latent.unsqueeze(0).repeat(rows, *([1] * n.latent.dim()))
        if jump:
            acts = torch.full((rows, L), -1, dtype=torch.long)
            acts[:A, 0] = torch.arange(A)
            acts[A, :len(n.option)] = torch.tensor(n.option)
        else:
            acts = torch.arange(A)
        with torch.no_grad():
            nxt, rew, _ = self.nets["dynamics"](lat, acts)
            logits, val, _, olog = self.nets["prediction"](nxt,
                                                           with_options=True)
            pol = torch.softmax(logits, dim=1).double()
            val = val.squeeze(-1).double().tolist()
            probs = torch.softmax(olog, dim=2)
            if self.use_prefix:
                vp = self.nets["value_prefix"]
                reset = (not n.has_state) or n.pref_len >= self.horizon
                h, c = vp.initial_state(1, CPU) if reset else (n.h, n.c)
                step = 1 if reset else n.pref_len + 1
                pre, _, (nh, nc) = vp(nxt, (h.repeat(1, rows, 1),
                                            c.repeat(1, rows, 1)),
                                      step=torch.full((rows,), step,
                                                      dtype=torch.long))
                pre = pre.squeeze(-1).double().tolist()
                edges = [p if reset else p - n.pref for p in pre]
            else:
                step = 0
                rew = rew.squeeze(-1) if rew.dim() > 1 else rew
                edges = rew.double().tolist()
                pre = [0.0] * rows
                nh = nc = None

        def hc(k):
            if nh is None:
                return None, None
            return nh[:, k:k + 1], nc[:, k:k + 1]

        for a in range(A):
            ch = n.children.get(a)
            if ch is None:
                ch = n.children[a] = RNode(n)
            ch.r, ch.ev, ch.prior = edges[a], val[a], n.policy[a]
            if not ch.evaluated:
                self.write(ch, nxt[a], pol[a], val[a],
                           n.total + self.dacc(n.depth) * edges[a],
                           n.depth + 1, pre[a], step, *hc(a), probs[a])
        if jump:
            n.opt_r, n.opt_v = edges[A], val[A]
            cur = n.children[n.option[0]]
            between = 0           # unevaluated nodes strictly inside the path
            for i, m in enumerate(n.option[1:]):
                nx = cur.children.get(m)
                if nx is None:
                    nx = cur.children[m] = RNode(cur)
                cur = nx
                if i < len(n.option) - 2 and not cur.evaluated:
                    between += 1
            self.stats["path_placeholders"] = max(
                self.stats["path_placeholders"], between)
            n.landing = cur
            if not cur.evaluated:
                self.write(cur, nxt[A], pol[A], val[A],
                           n.total + self.dacc(n.depth) * edges[A],
                           n.depth + len(n.option), pre[A], step, *hc(A),
                           probs[A])
        n.expanded = True
        self.stats["expanded"] += 1
        self.stats["len_hist"][len(n.option)] += 1

    def option_wins(self, n, child):
        land = n.landing
        total, n_o = child.count, land.count
        n_p = total - n_o
        tp, dp = n.total, self.dacc(n.depth)
        q_o = ((land.wsum / n_o - tp) / dp if n_o > 0 else
               n.opt_r + self.dacc(len(n.option)) * n.opt_v)
        q_p = (((child.wsum - land.wsum) / n_p - tp) / dp if n_p > 0 else
               child.r + self.d * child.ev)
        s_o = self.norm(q_o) + self.c * n.p_o * math.sqrt(total) / (1 + n_o)
        s_p = self.norm(q_p) + self.c * (1 - n.p_o) * math.sqrt(total) / (
            1 + n_p)
        return s_o > s_p

    def run(self, obs_one, sims, noise=None):
        x = torch.as_tensor(obs_one[None], dtype=torch.float32) / 255.0
        with torch.no_grad():
            lat = self.nets["representation"](x)
            logits, value, _, olog = self.nets["prediction"](
                lat, with_options=True)
        root = RNode()
        pol = torch.softmax(logits, dim=1).double()[0]
        self.write(root, lat[0], pol, value.squeeze(-1).double()[0].item(),
                   0.0, 0, 0.0, 0, None, None, torch.softmax(olog, 2)[0])
        root.has_state = False
        if noise is not None:
            root.policy = [0.75 * p + 0.25 * float(z)
                           for p, z in zip(root.policy, noise)]
        root.prior = 1.0
        self.root = root
        head0 = torch.softmax(olog, 2)[0, 0].tolist()
        self.pmax = head0.index(max(head0))          # p_max_id_ (304-307)
        self.expand(root)
        for k in root.children.values():             # the primitives only
            self.upd(k.r + self.d * k.ev)
        for _ in range(sims):
            self.simulate(root)
        return root

    def simulate(self, root):
        node, path, crossed = root, [root], []
        for _ in range(self.max_depth):
            if not node.expanded:
                break
            kids = [node.children[a] for a in range(self.A)]
            total = float(sum(k.count for k in kids))
            dp = self.dacc(node.depth)
            best = None
            for a, k in enumerate(kids):
                q = ((k.wsum / k.count - node.total) / dp if k.count > 0
                     else k.r + self.d * k.ev)
                s = self.norm(q) + self.c * k.prior * math.sqrt(total) / (
                    1 + k.count)
                if best is None or s > best[0]:
                    best = (s, a)
            a = best[1]
            child = kids[a]
            if (self.edge and node.landing is not None
                    and node.option[0] == a and self.option_wins(node, child)):
                cur = child
                path.append(cur)
                for m in node.option[1:]:
                    cur = cur.children[m]
                    path.append(cur)
                assert cur is node.landing
                crossed.append((node, cur))
                self.stats["cross_hist"][len(node.option)] += 1
                node = cur
            else:
                path.append(child)
                node = child
        assert node.evaluated
        if not node.expanded:
            self.expand(node)
        self.stats["cross_sims"] += bool(crossed)
        self.stats["leaf_sum"] += node.depth
        self.stats["leaf_max"] = max(self.stats["leaf_max"], node.depth)
        total_v = node.total + self.dacc(node.depth) * node.value
        for p in path:
            p.count += 1
            p.wsum += total_v
        for p in path[1:]:
            if p.parent.evaluated:
                self.upd((p.wsum / p.count - p.parent.total)
                         / self.dacc(p.parent.depth))
        for h, land in crossed:
            self.upd((land.wsum / land.count - h.total) / self.dacc(h.depth))


def compare(tree, t, root, A, tol=1e-12):
    """Node by node: the engine's tables against the reference tree."""
    pairs, stack = [], [(root, 0)]
    while stack:
        r, nid = stack.pop()
        pairs.append((r, nid))
        g = lambda k: tree[k][t, nid]  # noqa: E731
        assert int(g("visit")) == r.count, ("visit", nid)
        assert abs(g("wsum").item() - r.wsum) <= tol * max(1, abs(r.wsum))
        assert bool(g("evaluated")) == r.evaluated, ("evaluated", nid)
        assert bool(g("expanded")) == r.expanded, ("expanded", nid)
        if r.evaluated:
            assert abs(g("tot").item() - r.total) <= tol
            assert int(g("depth")) == r.depth
            assert abs(g("lval").item() - r.value) <= tol
            assert int(g("opt_len")) == len(r.option)
            if r.option:
                assert tree["opt_moves"][t, nid, :len(r.option)].tolist() == \
                    list(r.option)
                assert abs(g("opt_p").item() - r.p_o) <= tol
            assert int(g("pref_len")) == r.pref_len
            assert abs(g("pref").item() - r.pref) <= tol
        if r.parent is not None and r.parent.expanded:
            assert abs(g("reward").item() - r.r) <= tol
            assert abs(g("evalue").item() - r.ev) <= tol
            assert abs(g("prior").item() - r.prior) <= tol
        kids = tree["children"][t, nid]
        assert sorted(a for a in range(A) if kids[a] >= 0) == sorted(r.children)
        for a, rc in r.children.items():
            cid = int(kids[a])
            assert int(tree["parent"][t, cid]) == nid
            stack.append((rc, cid))
    ids = {id(r): nid for r, nid in pairs}
    for r, nid in pairs:
        if r.landing is not None:
            assert int(tree["opt_land"][t, nid]) == ids[id(r.landing)]
            assert abs(tree["opt_r"][t, nid].item() - r.opt_r) <= tol
            assert abs(tree["opt_v"][t, nid].item() - r.opt_v) <= tol
        else:
            assert int(tree["opt_land"][t, nid]) == -1
    assert int(tree["count"][t]) == len(pairs)
    return pairs


def shape_of(pairs):
    """(placeholders ever filled, landings with 2+ holders, landings that
    are also a filled primitive child) -- the potency counts."""
    holders = {}
    for r, _ in pairs:
        if r.landing is not None:
            holders.setdefault(id(r.landing), []).append(r)
    shared = sum(len(v) >= 2 for v in holders.values())
    prim = sum(1 for r, _ in pairs if id(r) in holders and r.parent.expanded)
    return shared, prim


def check_against_ref(eng_nets, ref_nets, A, L, obs, sims, discount, c_puct,
                      max_depth, edge, use_prefix, horizon, noise=None,
                      tol=1e-12, steps=None):
    eng = VectorizedOptionPUCT(eng_nets, A, CPU)
    out = []
    for t in range(len(obs)):
        if steps is not None:
            steps.clear()
        pol, val, counts, info = eng.search(
            obs[t:t + 1], num_simulations=sims, c_puct=c_puct,
            discount=discount, max_depth=max_depth, root_noise=noise is not None,
            noise_override=None if noise is None else noise[t:t + 1],
            tie_break="first", option_edge=edge, return_info=True,
            keep_tree=True)
        eng_steps = list(steps) if steps is not None else None
        if steps is not None:
            steps.clear()
        ref = Ref(ref_nets, A, L, discount, c_puct, max_depth, edge,
                  use_prefix, horizon)
        root = ref.run(obs[t], sims, None if noise is None else noise[t])
        ref_steps = list(steps) if steps is not None else None
        pairs = compare(eng.last_tree, 0, root, A, tol)
        # the min-max range (v1's updates plus every crossed jump's Q)
        assert abs(eng.last_tree["minv"][0].item() - ref.lo) <= tol
        assert abs(eng.last_tree["maxv"][0].item() - ref.hi) <= tol
        assert counts[0].tolist() == [root.children[a].count for a in range(A)]
        assert abs(val[0] - root.wsum / root.count) <= tol
        s = ref.stats
        assert info["expanded_nodes"].tolist() == [s["expanded"]]
        assert info["option_len_hist"][0].tolist() == s["len_hist"]
        assert info["sims_crossing_option"].tolist() == [s["cross_sims"]]
        assert info["crossed_len_hist"][0].tolist() == s["cross_hist"]
        assert info["leaf_depth_sum"][0] == s["leaf_sum"]
        assert info["leaf_depth_max"].tolist() == [s["leaf_max"]]
        assert info["root_option_len"].tolist() == [len(root.option)]
        assert info["root_pmax"].tolist() == [ref.pmax]
        if root.option:
            assert int(info["root_option"][0, 0]) == ref.pmax
        if steps is not None:
            assert eng_steps == ref_steps
        info["_ref"] = s
        out.append((info, pairs))
    return out


def potency(res):
    """(crossings, shared landings, filled-primitive landings, most
    placeholders inside one option path) over check_against_ref's trees."""
    c = sh = pr = ph = 0
    for info, pairs in res:
        c += int(info["sims_crossing_option"][0])
        s, p = shape_of(pairs)
        sh, pr = sh + s, pr + p
        ph = max(ph, info["_ref"]["path_placeholders"])
    return c, sh, pr, ph


@pytest.mark.parametrize("seed,use_prefix,horizon,max_depth", [
    (0, False, 3, 12), (1, True, 3, 12), (2, True, 2, 12), (3, True, 100, 4),
    (4, False, 3, 2), (5, True, 3, 1), (6, True, 2, 12), (7, False, 3, 12)])
def test_engine_matches_the_reference_on_fake_nets(knobs, seed, use_prefix,
                                                   horizon, max_depth):
    knobs.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", use_prefix)
    knobs.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
    fake = FakeNets(A=3, L=3, seed=seed)
    obs = fake_obs(3)
    noise = np.random.default_rng(seed).dirichlet([0.3] * 3, size=3) \
        if seed % 2 else None
    res = check_against_ref(fake.nets, fake.nets, 3, 3, obs, 30, 0.9, 1.25,
                            max_depth, True, use_prefix, horizon, noise=noise,
                            steps=fake.steps if use_prefix else None)
    crossed = sum(int(i["sims_crossing_option"][0]) for i, _ in res)
    assert crossed > 0, "potency: no option edge was crossed"


P_LEN = {3: (0.4, 0.45), 4: (0.3, 0.3, 0.3)}


@pytest.mark.parametrize("L", [3, 4])
@pytest.mark.parametrize("use_prefix,horizon", [(False, 3), (True, 2),
                                                (True, 4)])
def test_the_reference_matches_with_jump_dependent_evaluations(
        knobs, L, use_prefix, horizon):
    """Audit r1's stronger check, ported: a jump-made latent gets its own
    logits, value and suggestion (the random fake spec is flag-dependent),
    L 3 and 4 (two placeholders on one path), the prefix on and off; every
    tree matched node by node, and each config must actually cross options
    and fill landings as primitive children."""
    knobs.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", use_prefix)
    knobs.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
    total = [0, 0, 0, 0]
    for seed in range(30, 34):
        fake = FakeNets(A=3, L=L, seed=seed, p_len=P_LEN[L])
        res = check_against_ref(fake.nets, fake.nets, 3, L, fake_obs(2), 40,
                                0.9, 1.25, 12, True, use_prefix, horizon,
                                steps=fake.steps if use_prefix else None)
        c, sh, pr, ph = potency(res)
        total = [total[0] + c, total[1] + sh, total[2] + pr, max(total[3], ph)]
    assert total[0] > 0 and total[2] > 0, total
    if L == 4:
        assert total[3] >= 2, "potency: no path held two placeholders"


def test_the_reference_runs_exercise_every_landing_case(knobs):
    """Potency over seeds, L 3 and 4, prefix off and on: shared landings
    (two or more holders) and landings that are also a filled primitive
    child must both occur -- and were matched node by node."""
    shared = prim = 0
    for L in (3, 4):
        for use_prefix in (False, True):
            knobs.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", use_prefix)
            for seed in range(10, 14):
                fake = FakeNets(A=3, L=L, seed=seed, p_len=P_LEN[L])
                res = check_against_ref(
                    fake.nets, fake.nets, 3, L, fake_obs(2), 40, 0.9, 1.25,
                    12, True, use_prefix, 3,
                    steps=fake.steps if use_prefix else None)
                _, s, p, _ = potency(res)
                shared += s
                prim += p
    assert shared > 0 and prim > 0, (shared, prim)


@pytest.mark.parametrize("L", [3, 4])
def test_lockstep_batch_equals_per_tree_runs(knobs, L):
    """Fake nets are batch-invariant, so the T = 4 lockstep (jump rows of
    several holders in one wave, per-tree node allocation) must equal four
    T = 1 runs exactly, jump-dependent evaluations and all."""
    knobs.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", True)
    knobs.setattr(muzero, "PREFIX_LSTM_HORIZON", 2)
    fake = FakeNets(A=3, L=L, seed=21 + L, p_len=P_LEN[L])
    eng = VectorizedOptionPUCT(fake.nets, 3, CPU)
    obs = fake_obs(4)
    kw = dict(num_simulations=30, discount=0.9, c_puct=1.25,
              root_noise=False, tie_break="first", return_info=True,
              keep_tree=True)
    pol, val, counts, info = eng.search(obs, **kw)
    batch_tree = eng.last_tree
    assert info["sims_crossing_option"].sum() > 0
    for t in range(4):
        p1, v1, c1, i1 = eng.search(obs[t:t + 1], **kw)
        one = eng.last_tree
        assert c1[0].tolist() == counts[t].tolist()
        assert v1[0] == val[t]
        n = int(one["count"][0])
        assert int(batch_tree["count"][t]) == n
        for k in ("visit", "wsum", "tot", "lval", "parent", "children",
                  "opt_land", "opt_moves", "opt_r", "opt_v", "policy",
                  "reward", "evalue", "prior", "evaluated", "expanded",
                  "pref", "pref_len", "latent"):
            assert torch.equal(batch_tree[k][t, :n], one[k][0, :n]), k
        assert batch_tree["minv"][t] == one["minv"][0]
        assert batch_tree["maxv"][t] == one["maxv"][0]
        for k in info:
            if isinstance(info[k], np.ndarray):
                assert np.array_equal(info[k][t], i1[k][0]), k


# ==========================================================================
# real nets
# ==========================================================================
def real_world(monkeypatch, A=4, L=3, heads="random", per_step=True,
               use_prefix=True, horizon=3, seed=4):
    for key, val in (("USE_GUMBEL_SEARCH", False), ("TOKEN_VOCAB", None),
                     ("USE_VALUE_PREFIX_IN_TREE", use_prefix),
                     ("MINMAX_RANGE_FLOOR", 0.01), ("MAX_DEPTH", 12),
                     ("PREFIX_LSTM_HORIZON", horizon),
                     ("UNROLL_STEPS", horizon),
                     ("PREFIX_BN_PER_STEP", per_step),
                     ("LATENT_CHANNELS", 32), ("OPTION_LENGTH", L),
                     ("OPTION_EDGE", True), ("ACTION_ENCODING", "embed")):
        monkeypatch.setattr(muzero, key, val)
    torch.manual_seed(seed)
    nets = {"representation": muzero.build_representation(),
            "dynamics": muzero.build_dynamics(A),
            "prediction": muzero.build_prediction(A),
            "value_prefix": muzero.build_value_prefix()}
    g = torch.Generator().manual_seed(seed + 17)
    with torch.no_grad():
        for layer in (nets["prediction"].policy_fc, nets["prediction"].value_fc,
                      nets["dynamics"].reward_fc, nets["value_prefix"].fc):
            for p in layer.parameters():
                p.normal_(0, 0.5, generator=g)
        if L > 1:
            for fc in nets["prediction"].option_fcs:
                if heads == "stop":
                    fc.weight.zero_()        # bias 0 with 10 on stop: all stop
                else:
                    fc.weight.normal_(0, 0.5, generator=g)
                    fc.bias.normal_(0, 0.5, generator=g)
            for emb in nets["dynamics"].option_embeddings:
                emb.weight.normal_(0, 0.5, generator=g)
        vp = nets["value_prefix"]
        if per_step:
            for name in vp.norm_names():
                getattr(vp, f"{name}_step_mean").normal_(0, 0.3, generator=g)
                getattr(vp, f"{name}_step_var").uniform_(0.5, 2.0, generator=g)
    for n in nets.values():
        n.eval()
    obs = np.random.default_rng(seed).integers(0, 255, (3, 12, 84, 84),
                                               dtype=np.uint8)
    return nets, obs


def record_steps(monkeypatch, nets, log):
    vp = nets["value_prefix"]
    real = type(vp).forward.__get__(vp, type(vp))

    def forward(latent, state, step=None):
        log.append(None if step is None else
                   [int(s) for s in torch.as_tensor(step).reshape(-1)])
        return real(latent, state, step=step)

    monkeypatch.setattr(vp, "forward", forward)


def test_engine_matches_the_reference_on_real_nets(monkeypatch):
    nets, obs = real_world(monkeypatch)
    log = []
    record_steps(monkeypatch, nets, log)
    noise = np.random.default_rng(2).dirichlet([0.3] * 4, size=3)
    res = check_against_ref(nets, nets, 4, 3, obs, 20, 0.99, 1.25, 12, True,
                            True, 3, noise=noise, tol=1e-9, steps=log)
    crossed, _, prim, _ = potency(res)
    # (no head setting tried on real nets built a SHARED landing -- 32
    # tried in the r1 fix round; the fake-net checks cover that case)
    assert crossed > 0 and prim > 0, (crossed, prim)


def v1_counts(nets, A, obs, **kw):
    eng = VectorizedPUCT(nets, A, CPU)
    return eng.search(obs, tie_break="first", **kw)


@pytest.mark.parametrize("use_prefix", [True, False])
def test_stop_only_heads_match_v1(monkeypatch, use_prefix):
    """Every extra head at stop: no option anywhere, so the subclass is
    v1's search; root-relative statistics give the same visit counts and
    values to float tolerance (not bit for bit)."""
    nets, obs = real_world(monkeypatch, heads="stop", use_prefix=use_prefix)
    noise = np.random.default_rng(5).dirichlet([0.3] * 4, size=3)
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    for t in range(3):                      # T = 1: matched batch shapes
        for nz in (None, noise[t:t + 1]):
            kw = dict(num_simulations=25, c_puct=1.25, discount=0.99,
                      max_depth=12, root_noise=nz is not None,
                      noise_override=nz)
            p1, v1, c1 = v1_counts(nets, 4, obs[t:t + 1], **kw)
            p2, v2, c2, info = eng.search(obs[t:t + 1], tie_break="first",
                                          return_info=True, **kw)
            assert c2.tolist() == c1.tolist()
            assert abs(v2[0] - v1[0]) <= 1e-9
            assert info["expanded_with_option"].tolist() == [0]
            assert info["root_option_len"].tolist() == [0]
    # T = 3 lockstep: both engines see the same batch shapes too
    kw = dict(num_simulations=25, c_puct=1.25, discount=0.99, max_depth=3,
              root_noise=False)
    _, v1, c1 = v1_counts(nets, 4, obs, **kw)
    _, v2, c2 = eng.search(obs, tie_break="first", **kw)
    assert c2.tolist() == c1.tolist()
    assert np.allclose(v2, v1, rtol=0, atol=1e-9)


def test_option_edge_off_matches_v1_and_keeps_the_root_suggestions(
        monkeypatch):
    nets, obs = real_world(monkeypatch, heads="random")
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    lens = []
    for t in range(3):
        kw = dict(num_simulations=25, c_puct=1.25, discount=0.99,
                  max_depth=12, root_noise=False)
        _, v1, c1 = v1_counts(nets, 4, obs[t:t + 1], **kw)
        _, v_off, c_off, info = eng.search(
            obs[t:t + 1], tie_break="first", option_edge=False,
            return_info=True, **kw)
        assert c_off.tolist() == c1.tolist()
        assert abs(v_off[0] - v1[0]) <= 1e-9
        assert info["sims_crossing_option"].tolist() == [0]
        _, _, _, info_on = eng.search(obs[t:t + 1], tie_break="first",
                                      option_edge=True, return_info=True,
                                      **kw)
        for k in ("root_option", "root_option_len", "root_option_p",
                  "root_pmax"):
            assert np.array_equal(info[k], info_on[k]), k
        lens.append(int(info["root_option_len"][0]))
    assert any(n >= 2 for n in lens), "potency: no root suggested an option"


def test_jump_rows_take_their_holders_step(monkeypatch):
    """PREFIX_BN_PER_STEP: one jump = one LSTM call at the step its
    holder's primitive rows take. Lockstep T = 3 with horizon 2 (resets in
    the tree): every jump row of a wave carries the step of the parent
    whose latent it starts from."""
    nets, obs = real_world(monkeypatch, horizon=2)
    waves = []
    dyn, vp = nets["dynamics"], nets["value_prefix"]
    real_dyn = type(dyn).forward.__get__(dyn, type(dyn))
    real_vp = type(vp).forward.__get__(vp, type(vp))

    def dforward(latent, action):
        waves.append({"lat": latent.clone(), "act": action.clone()})
        return real_dyn(latent, action)

    def vforward(latent, state, step=None):
        waves[-1]["step"] = torch.as_tensor(step).reshape(-1).clone()
        return real_vp(latent, state, step=step)

    monkeypatch.setattr(dyn, "forward", dforward)
    monkeypatch.setattr(vp, "forward", vforward)
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    _, _, _, info = eng.search(obs, num_simulations=30, c_puct=1.25,
                               discount=0.99, max_depth=12, root_noise=False,
                               tie_break="first", return_info=True)
    jumps = 0
    for w in waves:
        act, step, lat = w["act"], w["step"], w["lat"]
        if act.dim() == 1:
            continue
        J = int((act[:, 1] >= 0).sum())
        B = (act.shape[0] - J) // 4
        assert (act[B * 4:, 1] >= 0).all() and (act[:B * 4, 1] < 0).all()
        for j in range(J):
            row = B * 4 + j
            owners = [b for b in range(B) if torch.equal(lat[row], lat[b * 4])]
            assert owners, "a jump row must start from one of its wave's parents"
            assert all(int(step[row]) == int(step[b * 4]) for b in owners)
            jumps += 1
        for b in range(B):
            assert (step[b * 4:(b + 1) * 4] == step[b * 4]).all()
    assert jumps > 0 and info["sims_crossing_option"].sum() > 0
    assert any(int(s) == 2 for w in waves if "step" in w for s in w["step"])


def test_a_jump_landing_holds_one_lstm_step(knobs):
    """Fake prefix, horizon 3: the root's jump row takes step 1 (as its
    primitive rows), so the landing holds prefix length 1 and its own
    children take step 2 -- one jump is one LSTM call."""
    knobs.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", True)
    knobs.setattr(muzero, "PREFIX_LSTM_HORIZON", 3)
    fake = FakeNets(A=3, L=3, preds={
        code_of([]): {"option": ((0, 0, 0), 0.9)},
        code_of([0]): {"value": 0.0},
        code_of([0, 0, 0]): {"value_jump": 5.0},
        code_of([1]): {"value": -5.0}, code_of([2]): {"value": -5.0}},
        default={"value": 0.0})
    tree, _, _, _, info = run(fake, 1, discount=0.9, c_puct=1.25)
    assert info["sims_crossing_option"].tolist() == [1]
    land = node_at(tree, 0, [0, 0, 0])
    assert int(tree["pref_len"][0, land]) == 1
    assert int(tree["pref_len"][0, node_at(tree, 0, [0])]) == 1
    assert fake.steps[0] == [1, 1, 1, 1]        # root wave: 3 + the jump
    assert fake.steps[1][:3] == [2, 2, 2]       # the landing's children
    kid = node_at(tree, 0, [0, 0, 0, 0])
    assert int(tree["pref_len"][0, kid]) == 2
    # the landing's edge reward is the jump's prefix (a reset at the root)
    assert tree["opt_r"][0, 0].item() == pytest.approx(
        tree["pref"][0, land].item(), abs=1e-12)


# ==========================================================================
# the interface
# ==========================================================================
def test_drop_in_returns_and_act(monkeypatch):
    nets, obs = real_world(monkeypatch)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    out = eng.search(obs, num_simulations=10, root_noise=True)
    assert len(out) == 3
    pol, val, counts = out
    assert pol.shape == (3, 4) and val.shape == (3,) and counts.shape == (3, 4)
    assert (counts.sum(axis=1) == 10).all() and np.isfinite(val).all()
    info = eng.last_info
    for k, shape in (("root_option", (3, 3)), ("root_option_len", (3,)),
                     ("root_option_p", (3,)), ("root_pmax", (3,)),
                     ("expanded_nodes", (3,)), ("expanded_with_option", (3,)),
                     ("option_len_hist", (3, 4)), ("simulations", (3,)),
                     ("sims_crossing_option", (3,)),
                     ("crossed_len_hist", (3, 4)), ("leaf_depth_sum", (3,)),
                     ("leaf_depth_max", (3,))):
        assert info[k].shape == shape, k
    assert eng.last_tree is None                    # not kept unless asked
    assert (info["expanded_nodes"] <= 11).all()
    assert (info["sims_crossing_option"] <= 10).all()
    assert (info["crossed_len_hist"].sum(1) >= info["sims_crossing_option"]).all()
    assert (info["option_len_hist"].sum(1) == info["expanded_nodes"]).all()
    assert (info["expanded_with_option"]
            == info["option_len_hist"][:, 2:].sum(1)).all()
    assert (info["simulations"] == 10).all()
    # root_pmax is the root policy head's argmax through these nets
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32) / 255.0
        logits = nets["prediction"](nets["representation"](x))[0]
    assert info["root_pmax"].tolist() == logits.argmax(1).tolist()
    held = info["root_option_len"] >= 2
    assert held.any()
    assert (info["root_option"][held, 0] == info["root_pmax"][held]).all()
    actions, p2, v2 = eng.act(obs, greedy=True, num_simulations=6)
    assert actions.tolist() == p2.argmax(axis=1).tolist()
    actions, _, _ = eng.act(obs, greedy=False, training_steps=0,
                            num_simulations=6)
    assert all(0 <= a < 4 for a in actions)


def test_l1_nets_carry_no_option(monkeypatch):
    nets, obs = real_world(monkeypatch, L=1)
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    kw = dict(num_simulations=15, c_puct=1.25, discount=0.99, max_depth=12,
              root_noise=False)
    _, v2, c2, info = eng.search(obs[:1], tie_break="first",
                                 return_info=True, **kw)
    _, v1, c1 = v1_counts(nets, 4, obs[:1], **kw)
    assert c2.tolist() == c1.tolist() and abs(v2[0] - v1[0]) <= 1e-9
    assert info["option_edge"] is False and info["option_length"] == 1
    assert info["root_option"].tolist() == [[-1]]
    assert info["expanded_with_option"].tolist() == [0]


def test_refusals(monkeypatch):
    nets, obs = real_world(monkeypatch)
    eng = VectorizedOptionPUCT(nets, 4, CPU)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", True)
    with pytest.raises(AssertionError, match="PUCT-only"):
        eng.search(obs[:1], num_simulations=2)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", [(a, 4) for a in range(4)])
    with pytest.raises(ValueError, match="TOKEN_VOCAB"):
        eng.search(obs[:1], num_simulations=2)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 1)
    short = dict(nets, dynamics=muzero.build_dynamics(4).eval())
    with pytest.raises(ValueError, match="option_length 1"):
        VectorizedOptionPUCT(short, 4, CPU).search(obs[:1], num_simulations=2)
    # the edge off needs no jump dynamics
    VectorizedOptionPUCT(short, 4, CPU).search(obs[:1], num_simulations=2,
                                               option_edge=False)
