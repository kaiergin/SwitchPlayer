"""Phase 4: vectorized PUCT search -- many trees as tensors, one device.

The 2026-08-30 re-audit's constraint is the design law here: WITHIN-tree
search stays strictly sequential (one simulation completes -- selection,
expansion, backup -- before the next begins in that tree); the
vectorization is ACROSS trees, mctx-style. Every tree runs the exact
per-tree semantics of MCTSHelper.run_full_mcts at sim_batches=1:

  * selection score  = normalize(reward + discount * onward) + u,
    onward = q if visited else expanded_value (0 at a terminal),
    u = c_puct * prior * sqrt(total_child_visits) / (1 + visits);
    per-tree MinMaxStats with the EfficientZero range floor, identity
    until a range exists; random tie-break among exact score ties.
  * virtual loss DOES NOT EXIST here -- provably: at sim_batches=1 the
    sequential engine sets vl on the descent path and clears it in the
    same simulation's backup, and selection at depth d scores only
    children of the current node, none of which acquired vl earlier in
    the same descent. No in-flight siblings ever, so no penalty term.
  * expansion creates ALL action children of the selected leaf in one
    dynamics/prediction/value-prefix forward (batch_expand_nodes
    semantics), including EfficientZero's is_reset edge rewards: a
    fresh-LSTM boundary's edge reward IS the prefix; within a segment
    it is the increment over the parent's accumulation; segments reset
    every PREFIX_LSTM_HORIZON plies and at the root (no state).
  * backup walks leaf -> root adding the leaf's network value (0 at a
    terminal), folding edge rewards at each hop, updating the tree's
    min-max range with reward + discount * q_new for every path node
    EXCEPT the root (whose in-edge lies outside the searched subtree).
  * seeding: the range is primed from the root children's composites
    before the first simulation; Dirichlet root noise (self-play only)
    is mixed into the root policy and written through to child priors.
  * readout: RAW visit fractions (never tempered), root q as value;
    acting samples at the staged temperature (argmax when greedy).

Score arithmetic runs in float64. The parity contract (audited
2026-09-01): EXACT -- bitwise visit counts and 1e-12 values -- when
inference is per-row reproducible, which in practice means comparing at
matched net batch shapes (the tests run T=1 for the exact gate; the
representation conv stack is NOT batch-size-invariant, ~1e-7 latent
wobble at B>1 amplifying to ~1e-3 in decoded values). At T>1 visit
counts are empirically stable across batch shapes (0 flips in ~280
audited trees) but not guaranteed: selection margins can reach 5e-6.
Two residual float asymmetries, both benign and documented rather than
chased: the sequential engine rounds c_puct*prior and the Dirichlet mix
through float32 (its prior is an np.float32 scalar; this engine keeps
float64), ~1e-8 relative. Production Dirichlet noise draws use torch's
GLOBAL RNG (torch.distributions takes no generator); the `generator`
argument seeds tie-breaks only. tests/test_vtree.py pins the contract
against the real MCTSHelper with de-zeroed heads (fresh factory nets
zero-init every final head, which makes all values a shared constant
and would leave the score arithmetic untested).

Scope (v1): fresh-root acting/refresh searches -- the act_search shape.
Chain trees (communal refresh) and Gumbel are out of scope; the
sequential engine remains authoritative for both.
"""
import numpy as np
import torch

import muzero


class VectorizedPUCT:
    """A batch of T independent PUCT searches on one device.

    Construction takes the four function nets (already on `device`, eval
    mode); every knob (discount, c_puct, depth, sims, horizon) is read
    from the muzero module at call time unless passed, matching the
    harness's arm-override contract.
    """

    def __init__(self, nets, action_space, device):
        self.nets = nets
        self.A = int(action_space)
        self.device = torch.device(device)
        # Resolved per search() from muzero (arm-override contract); one
        # engine instance therefore serves ONE search call at a time.
        self.use_prefix = None

    # ---- network wrappers (float32 in, float64 stats out) ----

    def _initial(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                device=self.device) / 255.0
            latent = self.nets["representation"](x)
            logits, value, _ = self.nets["prediction"](latent)
            policy = torch.softmax(logits, dim=1).double()
        return latent, policy, value.squeeze(-1).double()

    def _expand_forward(self, parent_latent, h, c, reset):
        """One expansion wave: parent latents [B,C,H,W] -> per-action
        children. Returns (child_latents [B,A,...], edge_rewards [B,A],
        child_policy [B,A,A], child_value [B,A], prefixes [B,A],
        new_h, new_c) with is_reset edge semantics applied by the CALLER
        (needs parent prefix)."""
        B = parent_latent.shape[0]
        A = self.A
        with torch.no_grad():
            lat = parent_latent.repeat_interleave(A, dim=0)
            acts = torch.arange(A, device=self.device).repeat(B)
            nxt, rew, _ = self.nets["dynamics"](lat, acts)
            logits, values, _ = self.nets["prediction"](nxt)
            pol = torch.softmax(logits, dim=1)
            if self.use_prefix:
                hh = h.repeat_interleave(A, dim=1).contiguous()
                cc = c.repeat_interleave(A, dim=1).contiguous()
                prefixes, _, (nh, nc) = self.nets["value_prefix"](
                    nxt, (hh, cc))
                prefixes = prefixes.squeeze(-1).double().view(B, A)
                nh = nh.view(nh.shape[0], B, A, -1)
                nc = nc.view(nc.shape[0], B, A, -1)
            else:
                prefixes = torch.zeros(B, A, dtype=torch.float64,
                                       device=self.device)
                nh = nc = None
            rew = rew.squeeze(-1) if rew.dim() > 1 else rew
        return (nxt.view(B, A, *nxt.shape[1:]),
                rew.double().view(B, A),
                pol.double().view(B, A, A),
                values.squeeze(-1).double().view(B, A),
                prefixes, nh, nc)

    # ---- the search ----

    def search(self, obs_batch, num_simulations=None, c_puct=None,
               discount=None, max_depth=None, root_noise=True,
               noise_override=None, generator=None,
               tie_break="random"):
        """Run T full searches; returns (policies [T,A] np, values [T] np,
        visit_counts [T,A] np).

        noise_override: optional [T,A] Dirichlet draws (tests inject the
        sequential engine's exact draw for parity). generator: torch
        Generator for TIE-BREAKS ONLY -- production noise uses torch's
        global RNG (torch.distributions takes no generator; see the
        module docstring). tie_break: "random" (production, matching
        the sequential engine's random.choice among exact ties) or
        "first" (deterministic; the parity tests patch the sequential
        engine to match)."""
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)  # sequential clamps too
        assert not muzero.USE_GUMBEL_SEARCH, (
            "VectorizedPUCT is PUCT-only (v1 scope); the sequential engine "
            "dispatches Gumbel -- wiring this engine under a Gumbel arm "
            "would silently swap algorithms")
        assert (muzero.MAX_DEPTH if max_depth is None else max_depth) >= 1, (
            "max_depth=0 forbids leaving the root: zero visits, NaN "
            "policies (termination-round F2; fail loud instead)")
        c_puct = float(muzero.C_PUCT if c_puct is None else c_puct)
        discount = float(muzero.DISCOUNT if discount is None else discount)
        max_depth = int(muzero.MAX_DEPTH if max_depth is None else max_depth)
        horizon = int(muzero.PREFIX_LSTM_HORIZON)
        floor = float(muzero.MINMAX_RANGE_FLOOR)
        self.use_prefix = bool(muzero.USE_VALUE_PREFIX_IN_TREE)
        dev = self.device
        A = self.A
        T = len(obs_batch)
        N = 1 + A * (sims + 1)          # root + its children + per-sim waves
        f64 = dict(dtype=torch.float64, device=dev)

        parent = torch.full((T, N), -1, dtype=torch.long, device=dev)
        children = torch.full((T, N, A), -1, dtype=torch.long, device=dev)
        visit = torch.zeros(T, N, **f64)
        wsum = torch.zeros(T, N, **f64)
        reward = torch.zeros(T, N, **f64)
        prior = torch.zeros(T, N, **f64)
        policy = torch.zeros(T, N, A, **f64)
        evalue = torch.zeros(T, N, **f64)
        expanded = torch.zeros(T, N, dtype=torch.bool, device=dev)
        pref = torch.zeros(T, N, **f64)
        pref_len = torch.zeros(T, N, dtype=torch.long, device=dev)
        has_state = torch.zeros(T, N, dtype=torch.bool, device=dev)
        minv = torch.full((T,), float("inf"), **f64)
        maxv = torch.full((T,), float("-inf"), **f64)
        count = torch.ones(T, dtype=torch.long, device=dev)  # node 0 = root

        root_latent, root_policy, root_value = self._initial(obs_batch)
        latent = torch.zeros(T, N, *root_latent.shape[1:],
                             dtype=torch.float32, device=dev)
        latent[:, 0] = root_latent
        evalue[:, 0] = root_value
        prior[:, 0] = 1.0
        if self.use_prefix:
            L, H = self.nets["value_prefix"].initial_state(1, dev)[0].shape[0], \
                self.nets["value_prefix"].initial_state(1, dev)[0].shape[-1]
            pref_h = torch.zeros(L, T, N, H, dtype=torch.float32, device=dev)
            pref_c = torch.zeros(L, T, N, H, dtype=torch.float32, device=dev)
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
        policy[:, 0] = root_policy

        rows = torch.arange(T, device=dev)

        def expand(node_ids, mask):
            """Expand `node_ids[t]` for trees where mask; all A children."""
            if not mask.any():
                return
            tsel = rows[mask]
            nsel = node_ids[mask]
            pl = latent[tsel, nsel]
            if self.use_prefix:
                h = pref_h[:, tsel, nsel]
                c = pref_c[:, tsel, nsel]
                reset = (~has_state[tsel, nsel]) | (
                    pref_len[tsel, nsel] >= horizon)
                init = self.nets["value_prefix"].initial_state(
                    int(tsel.numel()), dev)
                h = torch.where(reset.view(1, -1, 1), init[0], h)
                c = torch.where(reset.view(1, -1, 1), init[1], c)
            else:
                h = c = None
                reset = torch.zeros(tsel.numel(), dtype=torch.bool,
                                    device=dev)
            nxt, rew, pol, val, prefixes, nh, nc = self._expand_forward(
                pl, h, c, reset)
            if self.use_prefix:
                parent_pref = pref[tsel, nsel]
                edge = torch.where(reset.unsqueeze(1), prefixes,
                                   prefixes - parent_pref.unsqueeze(1))
                child_len = torch.where(
                    reset, torch.ones_like(pref_len[tsel, nsel]),
                    pref_len[tsel, nsel] + 1)
            else:
                edge = rew
                prefixes = torch.zeros_like(edge)
                child_len = torch.zeros(tsel.numel(), dtype=torch.long,
                                        device=dev)
            base = count[tsel]                       # [B]
            ids = base.unsqueeze(1) + torch.arange(A, device=dev)  # [B,A]
            children[tsel, nsel] = ids
            tsel_e = tsel.unsqueeze(1).expand_as(ids)
            parent[tsel_e, ids] = nsel.unsqueeze(1).expand_as(ids)
            reward[tsel_e, ids] = edge
            prior[tsel_e, ids] = policy[tsel, nsel]  # parent's policy row
            policy[tsel_e, ids] = pol
            evalue[tsel_e, ids] = val
            latent[tsel_e, ids] = nxt
            pref[tsel_e, ids] = prefixes
            pref_len[tsel_e, ids] = child_len.unsqueeze(1).expand_as(ids)
            has_state[tsel_e, ids] = self.use_prefix
            if self.use_prefix:
                pref_h[:, tsel_e, ids] = nh
                pref_c[:, tsel_e, ids] = nc
            expanded[tsel, nsel] = True
            count[tsel] = base + A

        def child_scores(node_ids):
            """[T,A] float64 scores of node_ids' children (assumes all
            expanded), matching selection_task_no_expand's arithmetic."""
            # clamp: leaf rows carry -1 child ids; their scores are
            # discarded by the descend mask, but gather itself must see
            # valid indices (CUDA device-asserts on negatives).
            ch = children[rows, node_ids].clamp(min=0)  # [T,A]
            v = visit.gather(1, ch)
            w = wsum.gather(1, ch)
            r = reward.gather(1, ch)
            ev = evalue.gather(1, ch)
            pr = prior.gather(1, ch)
            q = torch.where(v > 0, w / v.clamp(min=1), ev)
            comp = r + discount * q
            has_range = (maxv > minv)
            span = (maxv - minv).clamp(min=floor)
            norm = torch.where(has_range.unsqueeze(1),
                               (comp - minv.unsqueeze(1)) / span.unsqueeze(1),
                               comp)
            total = v.sum(dim=1, keepdim=True)
            u = c_puct * pr * torch.sqrt(total) / (1 + v)
            return norm + u

        def pick(scores, gen):
            if tie_break == "first":
                return scores.argmax(dim=1)
            best = scores.max(dim=1, keepdim=True).values
            ties = (scores == best).double()
            return torch.multinomial(ties, 1, generator=gen).squeeze(1)

        # Root expansion + range seeding (before any simulation).
        expand(torch.zeros(T, dtype=torch.long, device=dev),
               torch.ones(T, dtype=torch.bool, device=dev))
        rc = children[:, 0]                          # [T,A]
        seed_comp = reward.gather(1, rc) + discount * evalue.gather(1, rc)
        minv = torch.minimum(minv, seed_comp.min(dim=1).values)
        maxv = torch.maximum(maxv, seed_comp.max(dim=1).values)

        gen = generator
        for _ in range(sims):
            # ---- selection (all trees descend in lockstep; backup walks
            # parent pointers, so no explicit path bookkeeping) ----
            cur = torch.zeros(T, dtype=torch.long, device=dev)
            descending = torch.ones(T, dtype=torch.bool, device=dev)
            for d in range(max_depth):
                at_leaf = ~expanded[rows, cur]
                descending = descending & ~at_leaf
                if not descending.any():
                    break
                scores = child_scores(cur)
                nxt_local = pick(scores, gen)
                nxt_node = children[rows, cur, nxt_local]
                cur = torch.where(descending, nxt_node, cur)

            # ---- expansion (skip depth-capped already-expanded leaves) ----
            expand(cur, ~expanded[rows, cur])

            # ---- backup (leaf -> root), matching backup_simulation ----
            bv = evalue[rows, cur].clone()
            node = cur.clone()
            alive = torch.ones(T, dtype=torch.bool, device=dev)
            while alive.any():
                visit[rows[alive], node[alive]] += 1
                wsum[rows[alive], node[alive]] += bv[alive]
                bv = torch.where(alive, reward[rows, node] + discount * bv,
                                 bv)
                # Range update for every path node EXCEPT the root, with
                # the POST-increment q (sequential updates after +=).
                nonroot = alive & (node != 0)
                if nonroot.any():
                    qn = wsum[rows, node] / visit[rows, node].clamp(min=1)
                    upd = reward[rows, node] + discount * qn
                    minv = torch.where(nonroot, torch.minimum(minv, upd),
                                       minv)
                    maxv = torch.where(nonroot, torch.maximum(maxv, upd),
                                       maxv)
                nxt = parent[rows, node]
                alive = alive & (nxt >= 0)
                node = torch.where(alive, nxt, node)

        counts = visit.gather(1, children[:, 0])     # [T,A]
        pol = (counts / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        root_q = (wsum[:, 0] / visit[:, 0].clamp(min=1)).cpu().numpy()
        return pol, root_q, counts.cpu().numpy()

    def act(self, obs_batch, greedy=False, training_steps=0, **kw):
        """Vectorized act_search: returns (actions [T], policies, values).
        Greedy: argmax, no root noise. Self-play: Dirichlet noise +
        staged-temperature sampling, matching act_search's PUCT branch."""
        pol, val, _ = self.search(obs_batch, root_noise=not greedy, **kw)
        if greedy:
            actions = pol.argmax(axis=1)
        else:
            temp = muzero.visit_softmax_temperature(training_steps)
            actions = np.array([
                np.random.choice(self.A, p=muzero.apply_temperature(p, temp))
                for p in pol])
        return actions, pol, val


# ===========================================================================
# v2: COMMUNAL chain trees -- the archfix refresh topology, vectorized.
# ===========================================================================
#
# See VectorizedCommunalPUCT's docstring for the contract. Scalar "type"
# codes for the type-faithful arithmetic (NumPy 2 / NEP 50 promotion):
#   WEAK -- a Python float (or int): weak, adopts the other operand's dtype.
#   F32  -- an np.float32 scalar: strong; WEAK op F32 -> float32 arithmetic.
#   F64  -- an np.float64 scalar: strong; anything op F64 -> float64.
# Every stored scalar keeps its float64 VALUE (float32 values embed
# exactly) plus its type code; a binary op computes both precisions and
# keeps the one NumPy would have produced.
WEAK, F32, F64 = 0, 1, 2


def _scalar_type(x):
    """Type code of a Python/NumPy scalar as the sequential engine holds it."""
    if isinstance(x, np.floating):
        return F32 if x.dtype == np.float32 else F64
    return WEAK


def _tb(op, a, ta, b, tb):
    """Typed binary op on (value f64, type u8) pairs; broadcasting."""
    t = torch.maximum(ta, tb)
    r64 = op(a, b)
    r32 = op(a.float(), b.float()).double()
    return torch.where(t == F32, r32, r64), t


def _tc(op, a, ta, b, tb):
    """Typed comparison: float32 comparison when NumPy would cast the
    weak operand down (np.float32 vs Python float), float64 otherwise."""
    t = torch.maximum(ta, tb)
    return torch.where(t == F32, op(a.float(), b.float()), op(a, b))


_ADD = torch.add
_SUB = torch.sub
_MUL = torch.mul
_DIV = torch.div
_GT = torch.gt
_LT = torch.lt


class ChainForest:
    """Tensor node table holding T seeded chain trees (one per trajectory).

    Flat, shared: node ids are global; trees never share nodes, so a
    window's roots from any number of trees run in one batch. Chain nodes
    of tree t occupy ids base[t] .. base[t] + length[t] - 1 in trajectory
    order; imagined nodes are appended by expansion. Every scalar field is
    (float64 value, type code) -- see the module's WEAK/F32/F64 note.

    Per-node fields
      parent, children[A]      tree links (-1 = none)
      tree, cidx, owner        owning tree; chain index (-1 imagined);
                               chain index whose imagined subtree holds
                               the node (chain nodes own themselves; -1 =
                               dead slot, see _expand)
      beh                      behaviour action for chain nodes that have
                               a gameplay successor, else -1 -- the
                               sequential dict order puts that child
                               FIRST (chain link precedes expansion), and
                               expansion keeps it instead of re-creating
      visit, wsum(+t)          MCTS statistics; vl the in-round virtual loss
      reward(+t), prior(+t)    edge reward INTO the node, PUCT prior
      policy[A](+t per row)    the node's own policy (F32 from the net,
                               F64 once Dirichlet-noised as a root)
      evalue(+t)               network value at creation (F32 for chain
                               nodes -- eval_obs's numpy row -- WEAK for
                               imagined nodes' .item())
      expanded, done           lazy expansion flag; terminal chain tail
      pref, pref_len, has_state  value-prefix accumulation (chain nodes
                               never carry state: every chain node is an
                               LSTM reset, as in the sequential engine)
      latent[C,H,W]            float32; pref_h/pref_c[L,N,H] LSTM state
    """

    def __init__(self, engine, capacity):
        self.engine = engine
        self.A = engine.A
        self.dev = engine.device
        self.count = 0
        self.capacity = 0
        self.base = []          # per tree: first chain node id
        self.length = []        # per tree: chain length
        self.terminated = []    # per tree
        self.released = []      # per tree: last released chain index
        self.latent_shape = None
        self.state_shape = None   # (L_lstm, H) or None
        self.f = {}
        self._alloc(capacity)

    # ---- storage -------------------------------------------------------

    def _spec(self):
        A, dev = self.A, self.dev
        f64 = dict(dtype=torch.float64, device=dev)
        u8 = dict(dtype=torch.uint8, device=dev)
        lg = dict(dtype=torch.long, device=dev)
        bl = dict(dtype=torch.bool, device=dev)
        spec = {
            "parent": ((), lg, -1), "children": ((A,), lg, -1),
            "tree": ((), lg, -1), "cidx": ((), lg, -1), "owner": ((), lg, -1),
            "beh": ((), lg, -1),
            "visit": ((), f64, 0.0), "wsum": ((), f64, 0.0),
            "wsum_t": ((), u8, WEAK),
            "reward": ((), f64, 0.0), "reward_t": ((), u8, WEAK),
            "prior": ((), f64, 0.0), "prior_t": ((), u8, WEAK),
            "policy": ((A,), f64, 0.0), "policy_t": ((), u8, F32),
            "evalue": ((), f64, 0.0), "evalue_t": ((), u8, WEAK),
            "expanded": ((), bl, False), "done": ((), bl, False),
            "pref": ((), f64, 0.0), "pref_len": ((), lg, 0),
            "has_state": ((), bl, False), "vl": ((), f64, 0.0),
        }
        if self.latent_shape is not None:
            spec["latent"] = (tuple(self.latent_shape),
                              dict(dtype=torch.float32, device=dev), 0.0)
        return spec

    def _alloc(self, capacity, keep=None):
        """(Re)allocate every field at `capacity`, carrying over the rows
        `keep` (a LongTensor of old ids, in new-id order) or, when None,
        the first `count` rows."""
        new = {}
        for name, (suffix, kw, fill) in self._spec().items():
            t = torch.full((capacity, *suffix), fill, **kw)
            if name in self.f:
                old = self.f[name]
                src = old[:self.count] if keep is None else old[keep]
                t[:src.shape[0]] = src
            new[name] = t
        if self.state_shape is not None:
            # LSTM state lives in the net's (layers, batch, H) layout.
            L, H = self.state_shape
            for name in ("pref_h", "pref_c"):
                t = torch.zeros(L, capacity, H, dtype=torch.float32,
                                device=self.dev)
                if name in self.f:
                    old = self.f[name]
                    src = old[:, :self.count] if keep is None else old[:, keep]
                    t[:, :src.shape[1]] = src
                new[name] = t
        self.f = new
        self.capacity = capacity
        if keep is not None:
            self.count = int(keep.numel())

    def ensure_capacity(self, needed):
        if needed > self.capacity:
            self._alloc(int(needed))

    def set_geometry(self, latent_shape, state_shape):
        """Called once by build() before the first chain is appended."""
        assert self.count == 0
        self.latent_shape = tuple(latent_shape)
        self.state_shape = state_shape
        self._alloc(self.capacity)

    # ---- chain construction -------------------------------------------

    def append_chain(self, latents, policies, values, actions, rewards,
                     terminated, seed_values, seed_discount):
        """Link one trajectory as create_tree_from_game does, seeding the
        chain with the Monte-Carlo backup (computed with the SAME NumPy
        scalar objects the sequential engine folds, so every value AND its
        type code is reproduced by construction)."""
        L = len(actions)
        assert L >= 1
        self.ensure_capacity(self.count + L)
        f, dev = self.f, self.dev
        base = self.count
        ids = torch.arange(base, base + L, device=dev)
        t = len(self.base)
        self.base.append(base)
        self.length.append(L)
        self.terminated.append(bool(terminated))
        self.released.append(-1)

        f["tree"][ids] = t
        f["cidx"][ids] = torch.arange(L, device=dev)
        f["owner"][ids] = torch.arange(L, device=dev)
        f["parent"][ids[1:]] = ids[:-1]
        f["parent"][ids[0]] = -1
        acts = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=dev)
        if L > 1:
            f["children"][ids[:-1], acts[:-1]] = ids[1:]
            f["beh"][ids[:-1]] = acts[:-1]
        f["beh"][ids[-1]] = -1
        pol = np.asarray(policies)
        assert pol.dtype == np.float32, "chain policies must be eval_obs's float32 rows"
        f["policy"][ids] = torch.as_tensor(pol, dtype=torch.float64, device=dev)
        f["policy_t"][ids] = F32
        vals = np.asarray(values)
        assert vals.dtype == np.float32, "chain values must be eval_obs's float32 row"
        f["evalue"][ids] = torch.as_tensor(vals, dtype=torch.float64, device=dev)
        f["evalue_t"][ids] = F32
        # Priors: node k's prior is policy[k-1][action[k-1]] (np.float32);
        # the chain head keeps the constructor's 1.0 (Python float).
        f["prior"][ids[0]] = 1.0
        f["prior_t"][ids[0]] = WEAK
        if L > 1:
            pr = pol[np.arange(L - 1), np.asarray(actions[:-1], dtype=np.int64)]
            f["prior"][ids[1:]] = torch.as_tensor(pr, dtype=torch.float64, device=dev)
            f["prior_t"][ids[1:]] = F32
        rtypes = [_scalar_type(r) for r in rewards]
        f["reward"][ids] = torch.as_tensor(
            np.asarray([float(r) for r in rewards], dtype=np.float64), device=dev)
        f["reward_t"][ids] = torch.as_tensor(np.asarray(rtypes, dtype=np.uint8), device=dev)
        f["done"][ids[-1]] = bool(terminated)
        f["latent"][ids] = latents.to(dev, torch.float32)
        self.count += L

        if seed_values:
            # EXACTLY create_tree_from_game + backup_simulation, with the
            # sequential engine's scalar objects: leaf 0.0 at a terminal,
            # `expanded_value or 0.0` (np.float32, unless it is 0 -> falsy
            # -> Python 0.0) at a truncated tail; then
            # value_sum += bv; bv = reward + discount * bv up the chain.
            leaf = 0.0 if terminated else (vals[-1] or 0.0)
            bv = leaf
            vs, vt = [], []
            for k in range(L - 1, -1, -1):
                s = 0.0 + bv           # value_sum starts as Python 0.0
                vs.append(float(s))
                vt.append(_scalar_type(s))
                bv = rewards[k] + seed_discount * bv
            vs.reverse()
            vt.reverse()
            f["visit"][ids] = 1.0
            f["wsum"][ids] = torch.as_tensor(np.asarray(vs, dtype=np.float64), device=dev)
            f["wsum_t"][ids] = torch.as_tensor(np.asarray(vt, dtype=np.uint8), device=dev)
        return t

    def node_id(self, tree, idx):
        return self.base[tree] + int(idx)

    # ---- release / compaction -----------------------------------------

    def release(self, bounds):
        """release_span for every tree in `bounds` {tree: last_root_idx}:
        drop every imagined node hanging off chain nodes <= the bound
        (nothing later can descend backward into them) and every dead
        slot; keep chain nodes (their statistics are communal across
        windows) and the imagined subtrees past the bound (the next
        window's reuse). Compacts the table in place."""
        f = self.f
        n = self.count
        bound = torch.full((len(self.base),), -1, dtype=torch.long,
                           device=self.dev)
        for t in range(len(self.base)):
            bound[t] = self.released[t]
        for t, b in bounds.items():
            self.released[t] = max(int(b), self.released[t])
            bound[t] = self.released[t]
        owner = f["owner"][:n]
        tree = f["tree"][:n]
        keep = (f["cidx"][:n] >= 0) | ((owner >= 0) & (owner > bound[tree]))
        keep_ids = keep.nonzero().squeeze(1)
        remap = torch.full((n + 1,), -1, dtype=torch.long, device=self.dev)
        remap[keep_ids] = torch.arange(keep_ids.numel(), device=self.dev)
        # -1 links index the sentinel row (n) -> stay -1.
        par = f["parent"][:n].clone()
        par[par < 0] = n
        f["parent"][:n] = remap[par]
        ch = f["children"][:n].clone()
        ch[ch < 0] = n
        f["children"][:n] = remap[ch]
        old_base = list(self.base)
        self._alloc(int(keep_ids.numel()), keep=keep_ids)
        for t in range(len(self.base)):
            self.base[t] = int(remap[old_base[t]])


class VectorizedCommunalPUCT(VectorizedPUCT):
    """Communal chain-tree refresh (archfix's topology) on tensors.

    WHAT IS REPRODUCED (against MCTSHelper.create_tree_from_game +
    run_communal_mcts, the arm effzero-repro-refresh5x-communal runs):

    * The SEEDED chain tree: one node per stored position linked by the
      behaviour actions, priors policy[k-1][a[k-1]], real edge rewards,
      the terminal flag on a terminated tail, and the Monte-Carlo backup
      that gives every chain node one bookkeeping visit and its realized
      return (0 at a terminal; the network value at a truncated tail).
    * The communal window: the window's unexpanded roots expand in ONE
      batch (index order); each root in index order gets Dirichlet noise
      (muzero.add_dirichlet_noise itself, on the same float32 row -> same
      RNG consumption and float32 mixing) written through to its
      children's priors, and its own MinMaxStats seeded from its children
      in dict order (gameplay child first, then imagined by action).
    * Each round: ONE simulation per root. The sequential engine's
      asyncio.gather runs the selections to completion in INDEX ORDER
      (no task ever suspends), so root j's descent sees the virtual loss
      that roots < j deposited THIS round on shared nodes along the chain
      -- root j-k standing on root j's depth-d node at its own depth d+k.
      This engine reproduces that with a WAVE-STAGGERED lockstep: root
      slot j starts at wave 2j; at wave w root j is at depth w-2j, so
      every earlier root's deposit that root j can read was made in an
      earlier wave, and no two roots read/write the same node in one wave
      (proof in DESIGN.md). Selections never see same-round backups.
    * Expansion: unique unexpanded, non-terminal leaves in root order,
      one dynamics/prediction/value-prefix batch (the sequential engine
      chunks at 40 leaves; a window of <= 40 roots is one chunk, so the
      batch layout is identical). Chain parents keep their gameplay
      child; imagined children take the parent's policy row as priors.
    * Backup in root order: visit/value increments on shared nodes are
      applied in ROOT-RANK order (the k-th path through a node adds
      k-th), each root's range update reading the post-increment q as
      the sequential loop does, every path node except the root, leaf
      first. Virtual loss clears with the round.
    * Readout: visit fractions minus exactly one bookkeeping visit on the
      behaviour child (chain_seeded), root q as the value.

    TYPE-FAITHFUL ARITHMETIC. Under NumPy 2 the sequential engine mixes
    np.float32 scalars (chain nodes' eval_obs values and priors) with
    Python floats, and NEP 50 makes those operations float32: a backup
    through a chain leaf turns value_sum float32 for good, min/max ranges
    inherit whichever object won, comparisons cast the weak side down.
    Every stored scalar here carries the value in float64 plus NumPy's
    type code, and every op is evaluated in the precision NumPy used
    (WEAK/F32/F64 above), so root values match to the bit rather than to
    ~1e-7. The prior term c_puct*prior rounds through float32 for
    unnoised priors exactly as the sequential engine's np.float32 prior
    does (v1 documents this as a residual; here it is reproduced).

    NOT REPRODUCED (documented deltas): the Python `random.choice`
    tie-break stream (tie_break="first" is the parity mode, as in v1;
    "random" draws from a torch generator); a window wider than 40
    roots would split the sequential engine's expansion batch into
    chunks (batch-shape numerics only). Everything else is bitwise at
    T=1 on the CPU parity gate (tests/test_vtree_communal.py).

    Multi-tree: the flat table takes any number of trajectories; roots
    from different trees never share nodes and run in the same waves
    (slot = position within the tree's window). The integration path
    runs one trajectory per call (matching _search_traj_communal);
    batching trajectories changes network batch shapes (the documented
    ~1e-3 value wobble) and is the throughput lever left for later.
    """

    # ---- trajectory evaluation (eval_obs, batch 32) --------------------

    def evaluate(self, obs_seq, batch_size=32):
        """MCTSHelper.eval_obs's numerics: chunks of `batch_size` through
        representation + prediction. Returns (latents [L,C,H,W] float32
        on device, policies [L,A] np.float32, values [L] np.float32)."""
        lats, pols, vals = [], [], []
        with torch.no_grad():
            for i in range(0, len(obs_seq), batch_size):
                chunk = obs_seq[i:i + batch_size]
                sample = np.asarray(chunk[0])
                batch = np.empty((len(chunk), *sample.shape), dtype=sample.dtype)
                for j, o in enumerate(chunk):
                    batch[j] = o
                x = torch.tensor(batch, dtype=torch.float32,
                                 device=self.device) / 255.0
                latent = self.nets["representation"](x)
                logits, value, _ = self.nets["prediction"](latent)
                lats.append(latent)
                pols.append(torch.softmax(logits, dim=1).cpu().numpy())
                vals.append(value.squeeze(-1).cpu().numpy())
        return (torch.cat(lats, dim=0), np.concatenate(pols),
                np.concatenate(vals))

    # ---- forest construction -----------------------------------------

    def build(self, trajectories, seed_values=True, seed_discount=None,
              extra_capacity=0):
        """Build a ChainForest from evaluated trajectories: each a dict
        with latents (tensor [L,C,H,W]), policies/values (np.float32 from
        evaluate()), actions, rewards (the stored records' scalars),
        terminated. seed_discount defaults to muzero.DISCOUNT -- the
        module global create_tree_from_game's backup reads."""
        self.use_prefix = bool(muzero.USE_VALUE_PREFIX_IN_TREE)
        seed_discount = (muzero.DISCOUNT if seed_discount is None
                         else seed_discount)
        total = sum(len(t["actions"]) for t in trajectories)
        forest = ChainForest(self, 0)
        first = trajectories[0]["latents"]
        state_shape = None
        if self.use_prefix:
            init = self.nets["value_prefix"].initial_state(1, self.device)[0]
            state_shape = (int(init.shape[0]), int(init.shape[-1]))
        forest.set_geometry(first.shape[1:], state_shape)
        forest.ensure_capacity(total + int(extra_capacity))
        for tr in trajectories:
            forest.append_chain(tr["latents"], tr["policies"], tr["values"],
                                tr["actions"], tr["rewards"],
                                tr["terminated"], seed_values, seed_discount)
        return forest

    # ---- expansion (batch_expand_nodes) ------------------------------

    def _expand(self, forest, pids):
        """Expand the nodes `pids` (unexpanded, non-terminal, in the
        order the sequential engine would batch them): all A children in
        one forward; a chain parent keeps its gameplay child and its slot
        becomes a dead node (owner -1, dropped at release)."""
        B = int(pids.numel())
        if B == 0:
            return
        f, A, dev = forest.f, self.A, self.device
        assert forest.count + B * A <= forest.capacity, "forest capacity"
        horizon = int(muzero.PREFIX_LSTM_HORIZON)
        pl = f["latent"][pids]
        if self.use_prefix:
            reset = (~f["has_state"][pids]) | (f["pref_len"][pids] >= horizon)
            h = f["pref_h"][:, pids]
            c = f["pref_c"][:, pids]
            init = self.nets["value_prefix"].initial_state(B, dev)
            h = torch.where(reset.view(1, -1, 1), init[0], h)
            c = torch.where(reset.view(1, -1, 1), init[1], c)
            net = self.nets["value_prefix"]
            was_training = net.training
            net.eval()   # expand_prefixes toggles eval for the prefix net
            try:
                nxt, rew, pol, val, prefixes, nh, nc = self._expand_forward(
                    pl, h, c, reset)
            finally:
                net.train(was_training)
            parent_pref = f["pref"][pids]
            edge = torch.where(reset.unsqueeze(1), prefixes,
                               prefixes - parent_pref.unsqueeze(1))
            child_len = torch.where(reset, torch.ones_like(f["pref_len"][pids]),
                                    f["pref_len"][pids] + 1)
        else:
            reset = torch.zeros(B, dtype=torch.bool, device=dev)
            nxt, rew, pol, val, prefixes, nh, nc = self._expand_forward(
                pl, None, None, reset)
            edge = rew
            prefixes = torch.zeros_like(edge)
            child_len = torch.zeros(B, dtype=torch.long, device=dev)

        base = forest.count
        ids = base + torch.arange(B * A, device=dev).view(B, A)
        arangeA = torch.arange(A, device=dev)
        beh = f["beh"][pids]
        is_beh = arangeA.unsqueeze(0) == beh.unsqueeze(1)          # [B,A]
        f["children"][pids] = torch.where(is_beh, f["children"][pids], ids)
        flat = ids.view(-1)
        pexp = pids.unsqueeze(1).expand(B, A).reshape(-1)
        dead = is_beh.view(-1)
        f["parent"][flat] = torch.where(dead, torch.full_like(pexp, -1), pexp)
        f["tree"][flat] = f["tree"][pexp]
        f["cidx"][flat] = -1
        f["owner"][flat] = torch.where(dead, torch.full_like(pexp, -1),
                                       f["owner"][pexp])
        f["beh"][flat] = -1
        f["visit"][flat] = 0.0
        f["wsum"][flat] = 0.0
        f["wsum_t"][flat] = WEAK
        f["reward"][flat] = edge.reshape(-1)
        f["reward_t"][flat] = WEAK
        # prior = parent.policy[a]: np.float32 from a net row, np.float64
        # from a noised (root) row.
        f["prior"][flat] = f["policy"][pids].reshape(-1)
        f["prior_t"][flat] = f["policy_t"][pexp]
        f["policy"][flat] = pol.reshape(B * A, A)
        f["policy_t"][flat] = F32
        f["evalue"][flat] = val.reshape(-1)
        f["evalue_t"][flat] = WEAK
        f["expanded"][flat] = False
        f["done"][flat] = False
        f["pref"][flat] = prefixes.reshape(-1)
        f["pref_len"][flat] = child_len.unsqueeze(1).expand(B, A).reshape(-1)
        f["has_state"][flat] = bool(self.use_prefix)
        f["vl"][flat] = 0.0
        f["latent"][flat] = nxt.reshape(B * A, *nxt.shape[2:])
        if self.use_prefix:
            f["pref_h"][:, flat] = nh.reshape(nh.shape[0], B * A, -1)
            f["pref_c"][:, flat] = nc.reshape(nc.shape[0], B * A, -1)
        f["expanded"][pids] = True
        forest.count = base + B * A

    # ---- selection ---------------------------------------------------

    def _composite(self, f, ch, discount):
        """reward + discount * onward for child ids `ch` (typed)."""
        v = f["visit"][ch]
        q, qt = _tb(_DIV, f["wsum"][ch], f["wsum_t"][ch], v.clamp(min=1),
                    torch.zeros_like(f["wsum_t"][ch]))
        dn = f["done"][ch]
        zero = torch.zeros_like(q)
        onward = torch.where(v > 0, q, torch.where(dn, zero, f["evalue"][ch]))
        onward_t = torch.where(v > 0, qt, torch.where(
            dn, torch.zeros_like(qt), f["evalue_t"][ch]))
        d = torch.full_like(q, discount)
        dt = torch.zeros_like(qt)
        prod, prodt = _tb(_MUL, d, dt, onward, onward_t)
        return _tb(_ADD, f["reward"][ch], f["reward_t"][ch], prod, prodt)

    def _child_scores(self, f, n, st, g, c_puct, discount, floor):
        """[G,A] float64 PUCT scores of n's children with the typed
        arithmetic of selection_task_no_expand, including the virtual-
        loss clamp on children that earlier roots chose this round."""
        ch = f["children"][n]                               # [G,A]
        v = f["visit"][ch]
        vl = f["vl"][ch]
        comp, compt = self._composite(f, ch, discount)
        minv = st["minv"][g].unsqueeze(1)
        mint = st["mint"][g].unsqueeze(1)
        maxv = st["maxv"][g].unsqueeze(1)
        maxt = st["maxt"][g].unsqueeze(1)
        has_range = _tc(_GT, maxv, maxt, minv, mint)          # [G,1]
        num, numt = _tb(_SUB, comp, compt, minv, mint)
        span, spant = _tb(_SUB, maxv, maxt, minv, mint)
        fl = torch.full_like(span, floor)
        flt = torch.zeros_like(spant)
        use_floor = _tc(_GT, fl, flt, span, spant)            # max(span, floor)
        den = torch.where(use_floor, fl, span)
        dent = torch.where(use_floor, flt, spant)
        norm, normt = _tb(_DIV, num, numt, den, dent)
        score = torch.where(has_range, norm, comp)
        scoret = torch.where(has_range, normt, compt)
        nvis = v.clamp(min=1)
        zt = torch.zeros_like(scoret)
        pen, pent = _tb(_MUL, score, scoret, nvis, zt)
        pen, pent = _tb(_DIV, pen, pent, nvis + vl, zt)
        use_pen = (vl > 0) & _tc(_LT, pen, pent, score, scoret)  # min(score, pen)
        score = torch.where(use_pen, pen, score)
        pr = f["prior"][ch]
        prt = f["prior_t"][ch]
        cp32 = (torch.tensor(c_puct, dtype=torch.float32, device=pr.device)
                * pr.float()).double()
        cp = torch.where(prt == F32, cp32, c_puct * pr)
        total = (v + vl).sum(dim=1, keepdim=True)
        u = cp * torch.sqrt(total) / (1 + v + vl)
        return score + u

    def _pick(self, scores, beh, tie_break, gen):
        best = scores.max(dim=1, keepdim=True).values
        tie = scores == best
        A = scores.shape[1]
        if tie_break == "first":
            # dict order: the gameplay child first, then actions ascending
            ar = torch.arange(A, device=scores.device).unsqueeze(0)
            order = ar + 1
            order = torch.where(ar == beh.unsqueeze(1), torch.zeros_like(order),
                                order)
            key = torch.where(tie, order, torch.full_like(order, A + 2))
            return key.argmin(dim=1)
        return torch.multinomial(tie.double(), 1, generator=gen).squeeze(1)

    def _select_round(self, forest, root_ids, slots, st, c_puct, discount,
                      max_depth, tie_break, gen):
        """One simulation per root, wave-staggered (slot j starts at wave
        2j) so every cross-root virtual-loss read happens after the
        deposit the sequential order would have made. Returns (path
        [R, max_depth+1] node ids, -1 padded; plen [R])."""
        f, dev = forest.f, self.device
        R = int(root_ids.numel())
        cur = root_ids.clone()
        depth = torch.zeros(R, dtype=torch.long, device=dev)
        active = torch.ones(R, dtype=torch.bool, device=dev)
        path = torch.full((R, max_depth + 1), -1, dtype=torch.long, device=dev)
        path[:, 0] = root_ids
        rows = torch.arange(R, device=dev)
        floor = float(muzero.MINMAX_RANGE_FLOOR)
        wave = 0
        while bool(active.any()):
            step = active & (2 * slots <= wave)
            wave += 1
            if not bool(step.any()):
                continue
            at_leaf = (~f["expanded"][cur]) | f["done"][cur] | (depth >= max_depth)
            stop = step & at_leaf
            active = active & ~stop
            go = step & ~at_leaf
            if not bool(go.any()):
                continue
            g = rows[go]
            n = cur[g]
            scores = self._child_scores(f, n, st, g, c_puct, discount, floor)
            a = self._pick(scores, f["beh"][n], tie_break, gen)
            child = f["children"][n, a]
            f["vl"][child] += 1.0   # distinct nodes within a wave (DESIGN.md)
            cur[g] = child
            depth[g] += 1
            path[g, depth[g]] = child
        return path, depth + 1

    # ---- backup (expand_selected's drain, in root order) --------------

    def _backup_round(self, forest, path, plen, st, discount):
        f, dev = forest.f, self.device
        R, D = path.shape
        rows = torch.arange(R, device=dev)
        valid = torch.arange(D, device=dev).unsqueeze(0) < plen.unsqueeze(1)
        leaf = path[rows, plen - 1]
        lv = torch.where(f["done"][leaf], torch.zeros_like(f["evalue"][leaf]),
                         f["evalue"][leaf])
        lvt = torch.where(f["done"][leaf], torch.zeros_like(f["evalue_t"][leaf]),
                          f["evalue_t"][leaf])
        # Per-path contributions: c[len-1] = leaf value; c[i] = r[p_{i+1}]
        # + discount * c[i+1]  (backup_simulation's fold, typed).
        contrib = torch.zeros(R, D, dtype=torch.float64, device=dev)
        contrib_t = torch.zeros(R, D, dtype=torch.uint8, device=dev)
        bv, bvt = lv, lvt
        d = torch.full_like(bv, discount)
        dt = torch.zeros_like(bvt)
        for h in range(D):
            i = plen - 1 - h
            ok = i >= 0
            if not bool(ok.any()):
                break
            ii = i.clamp(min=0)
            node = path[rows, ii]
            contrib[rows[ok], ii[ok]] = bv[ok]
            contrib_t[rows[ok], ii[ok]] = bvt[ok]
            prod, prodt = _tb(_MUL, d, dt, bv, bvt)
            nb, nbt = _tb(_ADD, f["reward"][node], f["reward_t"][node], prod, prodt)
            bv = torch.where(ok, nb, bv)
            bvt = torch.where(ok, nbt, bvt)
        # Rank of each (root, path node) among the roots (ascending) whose
        # paths contain that node: the sequential drain applies root k's
        # increment k-th, and each root's range update reads the value
        # right after its own increment.
        jj, ii = valid.nonzero(as_tuple=True)
        nodes = path[jj, ii]
        key = nodes * R + jj
        order = torch.argsort(key)
        sn = nodes[order]
        is_start = torch.ones_like(sn, dtype=torch.bool)
        is_start[1:] = sn[1:] != sn[:-1]
        gid = torch.cumsum(is_start.long(), dim=0) - 1
        pos = torch.arange(sn.numel(), device=dev)
        first = torch.zeros(int(gid[-1]) + 1, dtype=torch.long, device=dev)
        first[gid[is_start]] = pos[is_start]
        rank_sorted = pos - first[gid]
        rank = torch.empty_like(rank_sorted)
        rank[order] = rank_sorted
        rank_mat = torch.full((R, D), -1, dtype=torch.long, device=dev)
        rank_mat[jj, ii] = rank
        post_w = torch.zeros(R, D, dtype=torch.float64, device=dev)
        post_wt = torch.zeros(R, D, dtype=torch.uint8, device=dev)
        post_v = torch.zeros(R, D, dtype=torch.float64, device=dev)
        for rho in range(int(rank.max()) + 1):
            sel = rank_mat == rho
            nd = path[sel]
            w, wt = _tb(_ADD, f["wsum"][nd], f["wsum_t"][nd], contrib[sel],
                        contrib_t[sel])
            f["wsum"][nd] = w
            f["wsum_t"][nd] = wt
            f["visit"][nd] += 1.0
            post_w[sel] = w
            post_wt[sel] = wt
            post_v[sel] = f["visit"][nd]
        # Range updates: every path node except the root, leaf first,
        # composite r + discount * q_post, typed max()/min() replacement.
        for h in range(D):
            i = plen - 1 - h
            ok = i >= 1
            if not bool(ok.any()):
                break
            ii = i.clamp(min=0)
            node = path[rows, ii]
            q, qt = _tb(_DIV, post_w[rows, ii], post_wt[rows, ii],
                        post_v[rows, ii].clamp(min=1), dt)
            prod, prodt = _tb(_MUL, d, dt, q, qt)
            upd, updt = _tb(_ADD, f["reward"][node], f["reward_t"][node],
                            prod, prodt)
            gt = ok & _tc(_GT, upd, updt, st["maxv"], st["maxt"])
            st["maxv"] = torch.where(gt, upd, st["maxv"])
            st["maxt"] = torch.where(gt, updt, st["maxt"])
            lt = ok & _tc(_LT, upd, updt, st["minv"], st["mint"])
            st["minv"] = torch.where(lt, upd, st["minv"])
            st["mint"] = torch.where(lt, updt, st["mint"])
        f["vl"].zero_()

    # ---- one communal window -----------------------------------------

    def search_window(self, forest, roots, num_simulations=None, c_puct=None,
                      discount=None, max_depth=None, tie_break="first",
                      generator=None, chain_seeded=True, epsilon=0.25):
        """run_communal_mcts over `roots` -- (tree, idx) pairs (plain ints
        mean tree 0). Returns {(tree, idx): (policy [A] np.float64,
        root_value float, has_policy bool, raw_counts [A] np.int64)};
        raw_counts are the root children's visit counts BEFORE the
        bookkeeping subtraction (the sequential node.visit_count)."""
        assert not muzero.USE_GUMBEL_SEARCH, (
            "VectorizedCommunalPUCT is PUCT-only; the sequential engine "
            "falls back to per-root searches under USE_GUMBEL_SEARCH -- "
            "wiring this engine there would silently swap algorithms")
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)
        c_puct = float(muzero.C_PUCT if c_puct is None else c_puct)
        discount = float(muzero.DISCOUNT if discount is None else discount)
        max_depth = int(muzero.MAX_DEPTH if max_depth is None else max_depth)
        assert max_depth >= 1, "max_depth=0 forbids leaving the root"
        self.use_prefix = bool(muzero.USE_VALUE_PREFIX_IN_TREE)
        A, dev = self.A, self.device
        roots = sorted((r if isinstance(r, tuple) else (0, int(r)))
                       for r in roots)
        R = len(roots)
        if R == 0:
            return {}
        forest.ensure_capacity(forest.count + R * (sims + 1) * A)
        f = forest.f
        root_ids = torch.tensor([forest.node_id(t, i) for t, i in roots],
                                dtype=torch.long, device=dev)
        slots = []
        per_tree = {}
        for t, _ in roots:
            slots.append(per_tree.get(t, 0))
            per_tree[t] = per_tree.get(t, 0) + 1
        slots = torch.tensor(slots, dtype=torch.long, device=dev)

        # 1. Expand the unexpanded roots in one batch (index order;
        #    batch_expand_nodes drops the terminal ones).
        exp = f["expanded"][root_ids].tolist()
        dn = f["done"][root_ids].tolist()
        to_expand = [int(rid) for rid, e, d in zip(root_ids.tolist(), exp, dn)
                     if not e and not d]
        self._expand(forest, torch.tensor(to_expand, dtype=torch.long, device=dev))

        # 2. Per root in index order: noise (RNG order preserved), then
        #    seed its MinMaxStats from its children in dict order.
        alpha = float(muzero.DIRICHLET_ALPHA)
        st = {"minv": torch.full((R,), float("inf"), dtype=torch.float64, device=dev),
              "maxv": torch.full((R,), float("-inf"), dtype=torch.float64, device=dev),
              "mint": torch.zeros(R, dtype=torch.uint8, device=dev),
              "maxt": torch.zeros(R, dtype=torch.uint8, device=dev)}
        for rid in root_ids.tolist():
            row = f["policy"][rid]
            if int(f["policy_t"][rid]) == F32:
                row_np = row.float().cpu().numpy()
            else:
                row_np = row.cpu().numpy()
            noisy = np.asarray(muzero.add_dirichlet_noise(
                row_np, epsilon=epsilon, alpha=alpha))
            ptype = F32 if noisy.dtype == np.float32 else F64
            f["policy"][rid] = torch.as_tensor(noisy, dtype=torch.float64, device=dev)
            f["policy_t"][rid] = ptype
            ch = f["children"][rid]
            has = ch >= 0
            f["prior"][ch[has]] = f["policy"][rid][has]
            f["prior_t"][ch[has]] = ptype
        beh = f["beh"][root_ids]
        arangeA = torch.arange(A, device=dev)
        rows = torch.arange(R, device=dev)
        # dict order of a root's children: the gameplay child (if any)
        # first, then imagined children by action.
        others = arangeA.unsqueeze(0).expand(R, A)
        keyed = torch.where(others == beh.unsqueeze(1), torch.zeros_like(others),
                            others + 1)
        dict_order = keyed.argsort(dim=1)                          # [R,A]
        for k in range(A):
            a = dict_order[:, k]
            ch = f["children"][root_ids, a]
            ok = ch >= 0
            if not bool(ok.any()):
                continue
            comp, compt = self._composite(f, ch.clamp(min=0), discount)
            gt = ok & _tc(_GT, comp, compt, st["maxv"], st["maxt"])
            st["maxv"] = torch.where(gt, comp, st["maxv"])
            st["maxt"] = torch.where(gt, compt, st["maxt"])
            lt = ok & _tc(_LT, comp, compt, st["minv"], st["mint"])
            st["minv"] = torch.where(lt, comp, st["minv"])
            st["mint"] = torch.where(lt, compt, st["mint"])

        # 3. Rounds: select (staggered lockstep) -> expand unique
        #    leaves in root order -> back up in root order.
        for _ in range(sims):
            path, plen = self._select_round(forest, root_ids, slots, st,
                                            c_puct, discount, max_depth,
                                            tie_break, generator)
            leaf_ids = path[rows, plen - 1]
            leaves = leaf_ids.tolist()
            exp = f["expanded"][leaf_ids].tolist()
            dn = f["done"][leaf_ids].tolist()
            seen, uniq = set(), []
            for leaf, e, d in zip(leaves, exp, dn):
                if leaf in seen:
                    continue
                seen.add(leaf)
                if not e and not d:
                    uniq.append(leaf)
            self._expand(forest, torch.tensor(uniq, dtype=torch.long, device=dev))
            self._backup_round(forest, path, plen, st, discount)

        # 4. Readout.
        ch = f["children"][root_ids]                                # [R,A]
        has = ch >= 0
        raw = torch.where(has, f["visit"][ch.clamp(min=0)],
                          torch.zeros_like(f["visit"][ch.clamp(min=0)]))
        counts = raw
        if chain_seeded:
            book = (arangeA.unsqueeze(0) == beh.unsqueeze(1)) & has
            counts = raw - book.double()
        rv, _ = _tb(_DIV, f["wsum"][root_ids], f["wsum_t"][root_ids],
                    f["visit"][root_ids].clamp(min=1),
                    torch.zeros_like(f["wsum_t"][root_ids]))
        rv = torch.where(f["visit"][root_ids] > 0, rv, torch.zeros_like(rv))
        raw_np = raw.cpu().numpy()
        counts_np = counts.cpu().numpy()
        rv_np = rv.cpu().numpy()
        out = {}
        for j, key in enumerate(roots):
            c = counts_np[j]
            total = c.sum()
            if total > 0:
                pol = c / total
                has_policy = True
            else:
                pol = np.ones(A) / A
                has_policy = False
            out[key] = (pol.astype(np.float64), float(rv_np[j]), has_policy,
                        raw_np[j].astype(np.int64))
        return out

    # ---- whole-trajectory driver (the _search_traj_communal shape) ----

    def search_trajectories(self, trajectories, indices_per_traj, window=None,
                            num_simulations=None, c_puct=None, discount=None,
                            max_depth=None, tie_break="random", generator=None,
                            seed_values=True):
        """Refresh-search the eligible `indices_per_traj[t]` (ascending) of
        each trajectory dict (obs, actions, rewards, terminated) window by
        window through ONE shared chain tree per trajectory, exactly as
        _search_traj_communal drives the sequential engine: windows of
        COMMUNAL_WINDOW_ROOTS in ascending order, release_span after each.
        Trees advance their windows in lockstep. Returns, per trajectory,
        {idx: (policy, root_value)} for roots with a policy."""
        window = max(1, int(muzero.COMMUNAL_WINDOW_ROOTS if window is None
                            else window))
        evaluated = []
        for tr in trajectories:
            lat, pol, val = self.evaluate(tr["obs"])
            evaluated.append({"latents": lat, "policies": pol, "values": val,
                              "actions": tr["actions"], "rewards": tr["rewards"],
                              "terminated": tr["terminated"]})
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)
        forest = self.build(evaluated, seed_values=seed_values,
                            extra_capacity=window * len(trajectories)
                            * (sims + 1) * self.A)
        windows = [[list(idx[w0:w0 + window])
                    for w0 in range(0, len(idx), window)]
                   for idx in (sorted(i) for i in indices_per_traj)]
        results = [dict() for _ in trajectories]
        n_rounds = max((len(w) for w in windows), default=0)
        for w in range(n_rounds):
            roots = [(t, i) for t, ws in enumerate(windows)
                     if w < len(ws) for i in ws[w]]
            res = self.search_window(forest, roots, num_simulations=sims,
                                     c_puct=c_puct, discount=discount,
                                     max_depth=max_depth, tie_break=tie_break,
                                     generator=generator, chain_seeded=seed_values)
            for (t, i), (pol, val, has_policy, _) in res.items():
                if has_policy:
                    results[t][i] = (np.asarray(pol, dtype=np.float64), float(val))
            forest.release({t: ws[w][-1] for t, ws in enumerate(windows)
                            if w < len(ws)})
        return results
