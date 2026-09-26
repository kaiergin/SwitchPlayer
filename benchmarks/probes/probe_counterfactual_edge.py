"""Does action discrimination fall when the projector collapses?

Time series over the edge run's per-eval snapshots (both seeds), scored on the
SAME counterfactual states/rollouts as probe_counterfactual_dynamics.py and
probe_counterfactual_latent.py (identical collection seeds, so the archfix and
control rows reproduce those logs). Per model:

  raw disc@k   argmin_a' ||g^k(s,a) - repr(true_k(a'))|| == a, over states
               whose three true futures differ in pixels (chance 1/3).
               PRIMARY: measured in the raw latent, which never collapses, so
               it cannot fall merely because the projector did.
  proj disc@k  same argmax over SimSiam cosines (eval-mode BN) -- what the
               consistency loss sees; expected to track the projector.
  spread p/t   predicted / true action spread in the raw latent (0 = the
               dynamics ignores the action).
  bs-rank      projector effective rank with BATCH statistics (throwaway
               train-mode copy), the collapse measure of probe_erank_artifact.
  dead         fraction of final projection-BN gammas with |gamma| < 0.01.

CPU only (CUDA hidden) so the live GPU run is untouched."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import sys, pathlib, pickle, time, copy
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
from muzero import initial_frame_stack, push_frame, preprocess_frame
torch.set_num_threads(6)

ARM = "effzero-repro-refresh5x-communal"
EDGE = "effzero-repro-refresh5x-communal-edge"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
EFF = (0, 2, 3); K = 5
R = ROOT / "benchmarks/results"
ARCHFIX = R / "pong-archfix-2026-08-30/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"
CONTROL = R / "pong-archfix-seq-s1-2026-09-13/weights-pong-skip4-effzero-repro-refresh5x-communal-seed1.pth"
MODELS = []
for s in (0, 1):
    snaps = sorted((R / f"pong-edge-s{s}-2026-09-21/snapshots-pong-skip4-{EDGE}-seed{s}").glob("weights-iter*.pth"))
    MODELS += [(f"edge s{s} @{int(p.stem[-3:])}", p) for p in snaps]
MODELS += [("archfix @90 +17.1", ARCHFIX), ("control s1 @90 -14.0", CONTROL)]
# Snapshot paths on the command line replace the edge series (the two
# references stay), e.g. a new arm's weights-iterNNN.pth files.
if len(sys.argv) > 1:
    MODELS = [(f"{pathlib.Path(q).parent.parent.name[5:]} @{int(pathlib.Path(q).stem[-3:])}", pathlib.Path(q))
              for q in sys.argv[1:]] + MODELS[-2:]


def load_nets(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    nets = {n: d[n] for n in ablation.NET_NAMES}
    # LATENT_RENORMALIZE is a plain attribute, not in the state_dict: take it
    # from the file's own arm spec, or a no-renorm (faithful) snapshot would
    # load cleanly and be silently min-max renormalised.
    nets["_renorm"] = d.get("spec", {}).get("LATENT_RENORMALIZE", True)
    return nets


def erank(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    p = s / s.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


with ablation.arm_config(ARM):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    nets = run.trainer
    dev = ablation.DEVICE
    assert dev.type == "cpu", dev
    print(f"device {dev}; {len(MODELS)} models", flush=True)

    def put(state):
        for n in ablation.NET_NAMES:
            getattr(nets, n).load_state_dict(state[n])
        nets.representation.renormalize_output = state["_renorm"]
        nets.dynamics.renormalize_output = state["_renorm"]
        run.set_train_mode(False)

    # ---- shared state set: identical to the two original counterfactual probes
    env = ablation.build_env("Pong-v4", 4)
    ale = env.unwrapped.ale
    rng = np.random.default_rng(0)

    def collect(policy, n, seed, every=4, warm=16):
        out = []
        obs, _ = env.reset(seed=seed)
        t = 0
        stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        while len(out) < n:
            stack = push_frame(stack, preprocess_frame(obs, muzero.IMAGE_RESHAPE))
            if t >= warm and t % every == 0:
                out.append((ale.cloneState(), stack.copy()))
            a = policy(stack)
            obs, _, term, trunc, _ = env.step(a)
            t += 1
            if term or trunc:
                obs, _ = env.reset()
                t = 0
                stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        return out

    def prior_policy(stack):
        p, _ = run.searcher.network_manager.initial_inference(stack)
        return int(rng.choice(6, p=p / p.sum()))

    states = []
    for label, src, n, seed in (("archfix-prior", ARCHFIX, 160, 11), ("control-prior", CONTROL, 160, 12), ("random", None, 120, 13)):
        if src is None:
            pol = lambda s: int(rng.integers(6))
        else:
            put(load_nets(src))
            pol = prior_policy
        got = collect(pol, n, seed)
        states += [(label, s, st) for s, st in got]
        print(f"collected {len(got)} states from {label}", flush=True)

    S0, S1, distinct = [], [], []
    for label, snap, st0 in states:
        seqs, ok = [], True
        for a in EFF:
            ale.restoreState(snap)
            st = st0.copy()
            seq = []
            for k in range(K):
                o, _, term, trunc, _ = env.step(a)
                st = push_frame(st, preprocess_frame(o, muzero.IMAGE_RESHAPE))
                seq.append(st)
                if term or trunc:
                    ok = False
                    break
            if not ok:
                break
            seqs.append(np.stack(seq))
        if not ok:
            continue
        seqs = np.stack(seqs)
        S0.append(st0); S1.append(seqs)
        newest = seqs[:, :, -3:]
        distinct.append([all(np.any(newest[i, k] != newest[j, k]) for i, j in ((0, 1), (0, 2), (1, 2))) for k in range(K)])
    S0 = np.stack(S0); S1 = np.stack(S1); distinct = np.array(distinct)
    dmask = np.repeat(distinct[:, None, :], 3, 1)
    print(f"usable states {len(S0)}; pixel-distinct futures at k=1..{K}: {distinct.mean(0).round(3).tolist()}", flush=True)

    @torch.no_grad()
    def score(bs=64):
        N = len(S0)
        D = np.zeros((N, 3, 3, K))    # raw L2: predicted(a_pred) -> true(a_true)
        C = np.zeros((N, 3, 3, K))    # SimSiam cosine, eval-mode BN
        Sp = np.zeros((N, K)); St = np.zeros((N, K))
        roots = []
        pairs = ((0, 1), (0, 2), (1, 2))
        for i in range(0, N, bs):
            x0 = torch.tensor(S0[i:i + bs], dtype=torch.float32, device=dev) / 255.0
            n = len(x0)
            lat0 = nets.representation(x0)
            roots.append(lat0)
            pr = torch.zeros(3, K, n, lat0[0].numel()); tr = torch.zeros_like(pr)
            on, tg = [[None] * K for _ in range(3)], [[None] * K for _ in range(3)]
            for ai, a in enumerate(EFF):
                lat = lat0
                for k in range(K):
                    lat, _, _ = nets.dynamics(lat, torch.full((n,), a, dtype=torch.long, device=dev))
                    xt = torch.tensor(S1[i:i + n, ai, k], dtype=torch.float32, device=dev) / 255.0
                    tl = nets.representation(xt)
                    pr[ai, k], tr[ai, k] = lat.flatten(1), tl.flatten(1)
                    on[ai][k] = F.normalize(nets.projector(lat, with_prediction=True), dim=1)
                    tg[ai][k] = F.normalize(nets.projector(tl, with_prediction=False), dim=1)
            for p in range(3):
                for q in range(3):
                    D[i:i + n, p, q] = (pr[p] - tr[q]).norm(dim=2).T.numpy()
                    C[i:i + n, p, q] = torch.stack([(on[p][k] * tg[q][k]).sum(1) for k in range(K)], 1).numpy()
            Sp[i:i + n] = np.mean([(pr[p] - pr[q]).norm(dim=2).T.numpy() for p, q in pairs], 0)
            St[i:i + n] = np.mean([(tr[p] - tr[q]).norm(dim=2).T.numpy() for p, q in pairs], 0)
        roots = torch.cat(roots)
        tp = copy.deepcopy(nets.projector).train()           # batch statistics, throwaway copy
        bs_rank = erank(tp(roots, with_prediction=False).numpy())
        return D, C, Sp, St, bs_rank

    def disc(pick):
        hit = np.stack([pick[:, a] == a for a in range(3)], 1)
        return [hit[:, :, k][dmask[:, :, k]].mean() for k in range(K)]

    print(f"\n{'model':22s} {'raw disc@1':>10s} {'@3':>6s} {'@5':>6s}   {'proj disc@1':>11s} {'@5':>6s}   "
          f"{'spread p/t@1':>12s} {'@5':>6s}   {'bs-rank':>7s} {'dead':>6s}  (chance disc 0.333)", flush=True)
    for label, src in MODELS:
        t0 = time.time()
        state = load_nets(src)
        put(state)
        D, C, Sp, St, bs_rank = score()
        rd, pd = disc(D.argmin(2)), disc(C.argmax(2))
        ratio = Sp.mean(0) / St.mean(0)
        g = state["projector"]
        gk = [k for k in g if k.startswith("projection.") and k.endswith(".weight") and g[k].ndim == 1][-1]
        dead = float((g[gk].abs() < 0.01).float().mean())
        print(f"{label:22s} {rd[0]:10.3f} {rd[2]:6.3f} {rd[4]:6.3f}   {pd[0]:11.3f} {pd[4]:6.3f}   "
              f"{ratio[0]:12.3f} {ratio[4]:6.3f}   {bs_rank:7.1f} {dead:6.3f}  [{time.time()-t0:.0f}s]", flush=True)
    print("\nDONE")
