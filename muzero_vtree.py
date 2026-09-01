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
        Generator for noise/tie-breaks. tie_break: "random" (production,
        matching the sequential engine's random.choice among exact ties)
        or "first" (deterministic; the parity tests patch the sequential
        engine to match)."""
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)  # sequential clamps too
        assert not muzero.USE_GUMBEL_SEARCH, (
            "VectorizedPUCT is PUCT-only (v1 scope); the sequential engine "
            "dispatches Gumbel -- wiring this engine under a Gumbel arm "
            "would silently swap algorithms")
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
