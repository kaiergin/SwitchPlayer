"""Root Q-spread vs PUCT exploration probe (2026-10-24, read-only on every tree).

Why do the FS run's birth-search visit distributions flatten between
iterations 45 and 60? This probe re-runs the birth search exactly as
Run._birth_search does (online nets in eval mode, ACT_SEARCH_SIMS, C_PUCT,
DISCOUNT, root noise) through an INSTRUMENTED COPY of
muzero_vtree.VectorizedPUCT.search -- parity-checked bit-for-bit against the
real engine at the same RNG state -- and dumps the root's per-child stats.

usage:
  python probe_qspread.py collect           # states from the iter-45 prior
  python probe_qspread.py search 45 60 75   # per-snapshot root records
CPU only, 2 threads.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import json
import pathlib
import sys
import time

TREE = pathlib.Path(r"Y:\SwitchPlayer-distill-run")
sys.path.insert(0, str(TREE)); sys.path.insert(0, str(TREE / "experiments"))
import numpy as np
import torch
torch.set_num_threads(2)
import muzero
import muzero_vtree
import ablation

assert not torch.cuda.is_available()
HERE = pathlib.Path(__file__).resolve().parent
RUN_DIR = TREE / "benchmarks/results/mspacman-student-fresh-s0-2026-10-21"
ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
       "tokens-rgrow-student-fresh")
SNAP_DIR = RUN_DIR / f"snapshots-mspacmannoframeskip-skip4-{ARM}-seed0"
VOCAB_JSON = TREE / "benchmarks/results/distill-prep-s0-2026-10-21/vocab.json"
ITERS, TRAIN_STEPS = 90, 2000          # the run's length (90 rows x 2000 steps)
ENV = "MsPacmanNoFrameskip-v4"

ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = \
    ENV, 4, 3000, 9


def snap(it):
    return SNAP_DIR / f"weights-iter{it:03d}.pth"


def refuse_mined(vocab):
    """E3: this probe reads (action, frames) duration tokens only -- a
    vocabulary holding MINED tokens (sequences of steps) is refused by name
    (it imports the frozen distill-run tree's muzero, which predates them)."""
    mined = [t for t in vocab if isinstance(t[0], (list, tuple))]
    if mined:
        raise SystemExit(f"the vocabulary holds mined tokens (E3) {mined}: "
                         f"this probe reads (action, frames) duration tokens "
                         f"only")
    return vocab


def snapshot_vocab(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    return refuse_mined(tuple(tuple(t) for t in d["spec"]["TOKEN_VOCAB"]))


def make_run():
    """The FS arm's Run with the student vocabulary (in memory only)."""
    vocab = snapshot_vocab(snap(45))
    vj = json.load(open(VOCAB_JSON))
    vj_vocab = vj.get("vocab") if isinstance(vj, dict) else vj
    if vj_vocab is not None:
        assert tuple(tuple(t) for t in vj_vocab) == vocab, (vj_vocab, vocab)
    ablation.ARMS[ARM] = {**ablation.ARMS[ARM], "TOKEN_VOCAB": vocab}
    return vocab


def load(run, path, acting=False):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    v = refuse_mined(tuple(tuple(t) for t in d["spec"]["TOKEN_VOCAB"]))
    assert v == muzero.TOKEN_VOCAB, (v, muzero.TOKEN_VOCAB)
    assert d["arm"] == ARM, d["arm"]
    for n in ablation.NET_NAMES:
        getattr(run.trainer, n).load_state_dict(d[n])
    renorm = d.get("spec", {}).get("LATENT_RENORMALIZE", True)
    run.trainer.representation.renormalize_output = renorm
    run.trainer.dynamics.renormalize_output = renorm
    run.trainer.training_steps = d.get("training_steps", 0)
    run.searcher.network_manager.training_steps = run.trainer.training_steps
    if acting:
        for n, sd in d["acting_states"].items():
            getattr(run.trainer, n).load_state_dict(sd)
    run.set_train_mode(False)
    return d


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def collect(n_games=8, keep=480, seed=4545):
    """Play whole games with the iteration-45 DEPLOYED prior (acting_states,
    as the actor sampled), sampled at the actor's temperature at that point;
    keep `keep` acted states evenly spaced over every life."""
    np.random.seed(seed); torch.manual_seed(seed)
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    d = load(run, snap(45), acting=True)
    # The actor of iteration 45 played on end-of-44 weights: 88000 steps.
    run.searcher.network_manager.training_steps = 88000
    T = muzero.prior_acting_temperature(88000)
    b = muzero.token_long_bias(88000)
    print(f"collect: snapshot steps {d['training_steps']}, acting T {T}, "
          f"long bias {b}, A {ablation.ACTIONS}", flush=True)
    obs, meta = [], []
    for g in range(n_games):
        run.env.was_real_done = True
        run.env.reset(seed=seed + g)
        life, game_ret = 0, 0.0
        while True:
            traj, ret, steps, term = run.play_episode(greedy=False, cap=3000)
            game_ret += ret
            for i, (o, a, r) in enumerate(traj[:-1]):
                obs.append(o)
                nxt = traj[i + 1][2]
                meta.append((g, life, i, int(a), float(sum(nxt))))
            print(f" game {g} life {life}: {len(traj) - 1} decisions, "
                  f"{steps} base steps, return {ret}", flush=True)
            life += 1
            if not term or getattr(run.env, "was_real_done", True):
                break
        print(f" game {g} return {game_ret}", flush=True)
    obs = np.stack(obs)
    meta = np.array(meta, dtype=np.float64)
    idx = np.unique(np.linspace(0, len(obs) - 1, keep).round().astype(int))
    np.savez_compressed(HERE / "states.npz", obs=obs[idx], meta=meta[idx],
                        n_total=len(obs))
    print(f"kept {len(idx)} of {len(obs)} states -> states.npz", flush=True)


# ---------------------------------------------------------------------------
# the instrumented search: a line-for-line copy of VectorizedPUCT.search
# (muzero_vtree.py, distill-run tree) plus read-only bookkeeping (depth,
# argmin/argmax node of the min-max range, root extras). Variants (my copy
# only): range_mode "engine" (exact) | "root" (min-max from the root
# children's composites only, for the outlier counterfactual).
# ---------------------------------------------------------------------------
class InstrumentedPUCT(muzero_vtree.VectorizedPUCT):
    step_nets = None      # {k: value_prefix copy with step-k BN stats}

    def _expand_forward(self, parent_latent, h, c, reset):
        if not (self.use_prefix and self.step_nets):
            return super()._expand_forward(parent_latent, h, c, reset)
        # COUNTERFACTUAL ("stepbn"): identical to the engine's forward, but
        # each parent's children take their prefix from the value-prefix copy
        # whose BN statistics were recalibrated at that unroll step.
        B = parent_latent.shape[0]
        A = self.A
        with torch.no_grad():
            lat = parent_latent.repeat_interleave(A, dim=0)
            acts = torch.arange(A, device=self.device).repeat(B)
            nxt, rew, _ = self.nets["dynamics"](lat, acts)
            logits, values, _ = self.nets["prediction"](nxt)
            pol = torch.softmax(logits, dim=1)
            hh = h.repeat_interleave(A, dim=1).contiguous()
            cc = c.repeat_interleave(A, dim=1).contiguous()
            steps = self._steps.repeat_interleave(A)
            prefixes = torch.zeros(B * A, dtype=torch.float64)
            nh = torch.zeros_like(hh); nc = torch.zeros_like(cc)
            for k in steps.unique().tolist():
                m = steps == k
                net = self.step_nets[min(int(k), max(self.step_nets))]
                p, _, (h2, c2) = net(nxt[m], (hh[:, m], cc[:, m]))
                prefixes[m] = p.squeeze(-1).double()
                nh[:, m] = h2; nc[:, m] = c2
            prefixes = prefixes.view(B, A)
            nh = nh.view(nh.shape[0], B, A, -1)
            nc = nc.view(nc.shape[0], B, A, -1)
            rew = rew.squeeze(-1) if rew.dim() > 1 else rew
        return (nxt.view(B, A, *nxt.shape[1:]), rew.double().view(B, A),
                pol.double().view(B, A, A), values.squeeze(-1).double().view(B, A),
                prefixes, nh, nc)

    def search(self, obs_batch, num_simulations=None, c_puct=None,
               discount=None, max_depth=None, root_noise=True,
               noise_override=None, generator=None,
               tie_break="random", range_mode="engine"):
        sims = max(int(muzero.ACT_SEARCH_SIMS if num_simulations is None
                       else num_simulations), 1)
        assert not muzero.USE_GUMBEL_SEARCH
        c_puct = float(muzero.C_PUCT if c_puct is None else c_puct)
        discount = float(muzero.DISCOUNT if discount is None else discount)
        max_depth = int(muzero.MAX_DEPTH if max_depth is None else max_depth)
        horizon = int(muzero.PREFIX_LSTM_HORIZON)
        floor = float(muzero.MINMAX_RANGE_FLOOR)
        self.use_prefix = bool(muzero.USE_VALUE_PREFIX_IN_TREE)
        dev = self.device
        A = self.A
        T = len(obs_batch)
        N = 1 + A * (sims + 1)
        f64 = dict(dtype=torch.float64, device=dev)
        edge_disc = muzero.token_edge_discounts()
        ndisc = slot_disc = None
        if edge_disc is not None:
            assert len(edge_disc) == A
            assert discount == float(muzero.DISCOUNT)
            slot_disc = torch.tensor(edge_disc, **f64)
            ndisc = torch.full((T, N), discount, **f64)

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
        count = torch.ones(T, dtype=torch.long, device=dev)
        # -- instrumentation --
        depth = torch.zeros(T, N, dtype=torch.long, device=dev)
        min_node = torch.full((T,), -1, dtype=torch.long, device=dev)
        max_node = torch.full((T,), -1, dtype=torch.long, device=dev)

        root_latent, root_policy, root_value = self._initial(obs_batch)
        raw_root_policy = root_policy.clone()
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
            if self.use_prefix:
                # (instrumentation only: the child's step index within its
                # prefix segment, read by the "stepbn" counterfactual)
                self._steps = torch.where(
                    reset, torch.ones_like(pref_len[tsel, nsel]),
                    pref_len[tsel, nsel] + 1)
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
            base = count[tsel]
            ids = base.unsqueeze(1) + torch.arange(A, device=dev)
            children[tsel, nsel] = ids
            tsel_e = tsel.unsqueeze(1).expand_as(ids)
            parent[tsel_e, ids] = nsel.unsqueeze(1).expand_as(ids)
            reward[tsel_e, ids] = edge
            if ndisc is not None:
                ndisc[tsel_e, ids] = slot_disc.unsqueeze(0).expand_as(edge)
            prior[tsel_e, ids] = policy[tsel, nsel]
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
            depth[tsel_e, ids] = (depth[tsel, nsel] + 1).unsqueeze(1).expand_as(ids)

        def child_scores(node_ids):
            ch = children[rows, node_ids].clamp(min=0)
            v = visit.gather(1, ch)
            w = wsum.gather(1, ch)
            r = reward.gather(1, ch)
            ev = evalue.gather(1, ch)
            pr = prior.gather(1, ch)
            q = torch.where(v > 0, w / v.clamp(min=1), ev)
            d = ndisc.gather(1, ch) if ndisc is not None else discount
            comp = r + d * q
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

        expand(torch.zeros(T, dtype=torch.long, device=dev),
               torch.ones(T, dtype=torch.bool, device=dev))
        rc = children[:, 0]
        d0 = ndisc.gather(1, rc) if ndisc is not None else discount
        seed_comp = reward.gather(1, rc) + d0 * evalue.gather(1, rc)
        minv = torch.minimum(minv, seed_comp.min(dim=1).values)
        maxv = torch.maximum(maxv, seed_comp.max(dim=1).values)
        min_node = rc.gather(1, seed_comp.argmin(dim=1, keepdim=True)).squeeze(1)
        max_node = rc.gather(1, seed_comp.argmax(dim=1, keepdim=True)).squeeze(1)

        gen = generator
        for _ in range(sims):
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

            expand(cur, ~expanded[rows, cur])

            bv = evalue[rows, cur].clone()
            node = cur.clone()
            alive = torch.ones(T, dtype=torch.bool, device=dev)
            while alive.any():
                visit[rows[alive], node[alive]] += 1
                wsum[rows[alive], node[alive]] += bv[alive]
                dn = ndisc[rows, node] if ndisc is not None else discount
                bv = torch.where(alive, reward[rows, node] + dn * bv,
                                 bv)
                nonroot = alive & (node != 0)
                if range_mode == "root":
                    # counterfactual: only ROOT CHILDREN update the range
                    nonroot = nonroot & (parent[rows, node] == 0)
                if nonroot.any():
                    qn = wsum[rows, node] / visit[rows, node].clamp(min=1)
                    upd = reward[rows, node] + dn * qn
                    lo = nonroot & (upd < minv)
                    hi = nonroot & (upd > maxv)
                    min_node = torch.where(lo, node, min_node)
                    max_node = torch.where(hi, node, max_node)
                    minv = torch.where(nonroot, torch.minimum(minv, upd),
                                       minv)
                    maxv = torch.where(nonroot, torch.maximum(maxv, upd),
                                       maxv)
                nxt = parent[rows, node]
                alive = alive & (nxt >= 0)
                node = torch.where(alive, nxt, node)

        counts = visit.gather(1, children[:, 0])
        pol = (counts / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        root_q = (wsum[:, 0] / visit[:, 0].clamp(min=1)).cpu().numpy()

        # ---------------- root instrumentation (read-only) ----------------
        ch = children[:, 0]
        v = visit.gather(1, ch); w = wsum.gather(1, ch)
        r = reward.gather(1, ch); ev = evalue.gather(1, ch)
        pr = prior.gather(1, ch)
        dch = ndisc.gather(1, ch) if ndisc is not None else torch.full_like(r, discount)
        q = torch.where(v > 0, w / v.clamp(min=1), ev)
        comp = r + dch * q
        comp_init = r + dch * ev
        span = (maxv - minv).clamp(min=floor)
        norm = (comp - minv.unsqueeze(1)) / span.unsqueeze(1)
        total = v.sum(dim=1, keepdim=True)
        u = c_puct * pr * torch.sqrt(total) / (1 + v)
        # all visited non-root nodes: depth profile of edge reward / value
        vis = (visit > 0)
        vis[:, 0] = False
        dep = depth.clone()
        node_comp = reward + (ndisc if ndisc is not None else discount) * \
            torch.where(visit > 0, wsum / visit.clamp(min=1), evalue)
        prof = {}
        for k in range(1, 9):
            m = vis & (dep == k)
            cnt = m.sum(1).double()
            prof[k] = dict(
                n=cnt.cpu().numpy(),
                ev=(torch.where(m, evalue, 0).sum(1) / cnt.clamp(min=1)).cpu().numpy(),
                rew=(torch.where(m, reward, 0).sum(1) / cnt.clamp(min=1)).cpu().numpy(),
                comp=(torch.where(m, node_comp, 0).sum(1) / cnt.clamp(min=1)).cpu().numpy())
        extra = dict(
            raw_prior=raw_root_policy.cpu().numpy(), prior=pr.cpu().numpy(),
            N=v.cpu().numpy(), Q=q.cpu().numpy(), R=r.cpu().numpy(),
            EV=ev.cpu().numpy(), D=dch.cpu().numpy(), comp=comp.cpu().numpy(),
            comp_init=comp_init.cpu().numpy(), norm=norm.cpu().numpy(),
            U=u.cpu().numpy(), minv=minv.cpu().numpy(), maxv=maxv.cpu().numpy(),
            root_ev=root_value.cpu().numpy(), root_q=root_q,
            min_depth=depth[rows, min_node].cpu().numpy(),
            max_depth=depth[rows, max_node].cpu().numpy(),
            min_rew=reward[rows, min_node].cpu().numpy(),
            max_rew=reward[rows, max_node].cpu().numpy(),
            min_ev=evalue[rows, min_node].cpu().numpy(),
            max_ev=evalue[rows, max_node].cpu().numpy(),
            n_nodes_visited=vis.sum(1).cpu().numpy(),
            max_tree_depth=torch.where(vis, dep, 0).max(1).values.cpu().numpy(),
            prof=prof)
        return pol, root_q, counts.cpu().numpy(), extra


def birth_nets(run):
    return {"representation": run.trainer.representation,
            "dynamics": run.trainer.dynamics,
            "prediction": run.trainer.prediction,
            "value_prefix": run.trainer.value_prefix}


def parity_check(run, obs):
    """The instrumented copy must reproduce the real engine bit for bit at
    the same global RNG state (noise + tie-breaks)."""
    nets = birth_nets(run)
    real = muzero_vtree.VectorizedPUCT(nets, ablation.ACTIONS, "cpu")
    mine = InstrumentedPUCT(nets, ablation.ACTIONS, "cpu")
    out = {}
    for noise in (True, False):
        torch.manual_seed(123)
        p1, v1, c1 = real.search(obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                                 c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                                 root_noise=noise)
        torch.manual_seed(123)
        p2, v2, c2, _ = mine.search(obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                                    c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                                    root_noise=noise)
        out[noise] = (bool(np.array_equal(c1, c2)), float(np.abs(v1 - v2).max()))
    return out


VARIANTS = {
    "noise": dict(root_noise=True),          # the birth search on acted steps
    "nonoise": dict(root_noise=False),
    "noise_rootrange": dict(root_noise=True, range_mode="root"),
    # counterfactuals (probe-only, never the engine's behaviour):
    "noise_prefixoff": dict(root_noise=True),   # zero edge rewards
    "noise_stepbn": dict(root_noise=True),      # per-step prefix BN stats
}


def step_calibrated_prefix(run, obs, acts, steps=None):
    """Value-prefix copies with BN statistics re-estimated PER UNROLL STEP
    (k = 1..horizon) on the probe's real states -- the batch statistics each
    per-step training call normalizes with -- for the stepbn counterfactual.
    Step 1 follows the played token, later steps the prior's argmax. The
    loaded nets are never modified (deep copies)."""
    import copy
    steps = steps or int(muzero.PREFIX_LSTM_HORIZON)
    rep, dyn, pred, vp = (run.trainer.representation, run.trainer.dynamics,
                          run.trainer.prediction, run.trainer.value_prefix)
    nets = {}
    with torch.no_grad():
        lat = rep(torch.tensor(obs, dtype=torch.float32) / 255.0)
        a = torch.tensor(acts, dtype=torch.long)
        state = vp.initial_state(len(obs), "cpu")
        for k in range(1, steps + 1):
            lat, _, _ = dyn(lat, a)
            net = copy.deepcopy(vp)
            for m in net.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.reset_running_stats(); m.momentum = None
            net.train()
            _, _, state_next = net(lat, state)     # one batch -> its stats
            net.eval()
            _, _, state = net(lat, state)          # chain on with eval copy
            nets[k] = net
            a = pred(lat)[0].argmax(1)
    return nets


def search(iters, variants=("noise", "nonoise"), batch=32):
    st = np.load(HERE / "states.npz")
    obs = st["obs"]
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    print(f"search: {len(obs)} states, A {ablation.ACTIONS}, sims "
          f"{muzero.ACT_SEARCH_SIMS}, c_puct {muzero.C_PUCT}, discount "
          f"{muzero.DISCOUNT}, alpha {muzero.DIRICHLET_ALPHA}, max_depth "
          f"{muzero.MAX_DEPTH}, prefix {muzero.USE_VALUE_PREFIX_IN_TREE}, "
          f"horizon {muzero.PREFIX_LSTM_HORIZON}, floor "
          f"{muzero.MINMAX_RANGE_FLOOR}, edge disc "
          f"{muzero.token_edge_discounts()}", flush=True)
    for it in iters:
        d = load(run, snap(it))
        ceiling = muzero.reward_ceiling(d["training_steps"])
        nets = birth_nets(run)
        for n in nets.values():
            n.eval()
        if it == iters[0]:
            print(" parity (counts equal, max |dv|):", parity_check(run, obs[:8]),
                  flush=True)
        eng = InstrumentedPUCT(nets, ablation.ACTIONS, "cpu")
        # deployed (acting_states) prior on the same states, for reference
        dd = torch.load(str(snap(it)), map_location="cpu", weights_only=False)
        for variant in variants:
            kw = VARIANTS[variant]
            t0 = time.time()
            recs = []
            saved_prefix = muzero.USE_VALUE_PREFIX_IN_TREE
            eng.step_nets = None
            if variant == "noise_prefixoff":
                muzero.USE_VALUE_PREFIX_IN_TREE = False
            if variant == "noise_stepbn":
                eng.step_nets = step_calibrated_prefix(
                    run, obs, st["meta"][:, 3].astype(int))
            for b0 in range(0, len(obs), batch):
                torch.manual_seed(10_000 * it + b0 + (0 if kw.get("root_noise") else 7))
                np.random.seed(b0)
                _, _, _, ex = eng.search(obs[b0:b0 + batch],
                                         num_simulations=muzero.ACT_SEARCH_SIMS,
                                         c_puct=muzero.C_PUCT,
                                         discount=muzero.DISCOUNT, **kw)
                recs.append(ex)
            muzero.USE_VALUE_PREFIX_IN_TREE = saved_prefix
            eng.step_nets = None
            out = {}
            for k in recs[0]:
                if k == "prof":
                    out["prof"] = {
                        dk: {f: np.concatenate([r["prof"][dk][f] for r in recs])
                             for f in recs[0]["prof"][dk]}
                        for dk in recs[0]["prof"]}
                else:
                    out[k] = np.concatenate([r[k] for r in recs])
            flat = {k: v for k, v in out.items() if k != "prof"}
            for dk, dv in out["prof"].items():
                for f, arr in dv.items():
                    flat[f"prof{dk}_{f}"] = arr
            np.savez_compressed(HERE / f"root-iter{it:03d}-{variant}.npz",
                                ceiling=ceiling,
                                training_steps=d["training_steps"], **flat)
            print(f" iter {it} {variant}: {time.time() - t0:.0f}s", flush=True)
        # deployed prior
        with torch.no_grad():
            saved = {n: {k: t.clone() for k, t in getattr(run.trainer, n).state_dict().items()}
                     for n in dd["acting_states"]}
            for n, sd in dd["acting_states"].items():
                getattr(run.trainer, n).load_state_dict(sd)
            run.set_train_mode(False)
            ps = []
            for b0 in range(0, len(obs), 64):
                x = torch.tensor(obs[b0:b0 + 64], dtype=torch.float32) / 255.0
                lg, val, _ = run.trainer.prediction(run.trainer.representation(x))
                ps.append(torch.softmax(lg.float(), 1).numpy())
            for n, sd in saved.items():
                getattr(run.trainer, n).load_state_dict(sd)
        np.save(HERE / f"acting-prior-iter{it:03d}.npy", np.concatenate(ps))
    print("DONE", flush=True)


def main(argv):
    make_run()
    with ablation.arm_config(ARM), ablation._run_length_globals(ITERS, TRAIN_STEPS), \
            ablation.token_actions():
        assert ablation.ACTIONS == len(muzero.TOKEN_VOCAB) == 14, ablation.ACTIONS
        assert muzero.BIRTH_SEARCH and not muzero.USE_SEARCH_ACTING
        if argv[0] == "collect":
            collect()
        elif argv[0] == "search":
            its = [int(a) for a in argv[1:] if a.isdigit()]
            variants = [a for a in argv[1:] if not a.isdigit()] or ["noise", "nonoise"]
            search(its, variants)
        else:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
