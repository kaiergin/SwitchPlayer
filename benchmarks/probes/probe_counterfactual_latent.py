"""FOLLOW-UP (latent space + collapse diagnostics) to probe_counterfactual_dynamics.
Early-collapse hypothesis, model side: does the dynamics net know what
the actions DO -- including actions the policy rarely picks?

Counterfactual ground truth from the emulator: at each sampled state, clone
the ALE state, hold each effective action (NOOP=0, UP=2, DOWN=3) for K=5
decisions, restore. Every model is scored on the SAME states and rollouts.
Measured in each model's own SimSiam space, exactly as training's
consistency loss (online = projector+predictor on the dynamics latent,
target = projection of representation(true stack)):
  match      cos(online(g^k(s,a)), target(true_k(a)))
  identity   same with g^k replaced by the root latent (a dynamics that
             ignores the action AND time)
  disc       argmax_a' cos(online(g^k(s,a)), target(true_k(a'))) == a,
             over states whose three true futures differ in pixels
             (chance 1/3) -- can the model tell which action it took?
  sep_true   mean pairwise cos between the three TARGET projections (does
             the representation even encode the paddle difference?)
  sep_pred   mean pairwise cos between the three ONLINE predictions (1.0 =
             action-blind dynamics)
Split by the model's own prior mass on the action (p_eff(a|s)) to test
"rarely-taken actions are unmodeled"."""
import sys, pathlib, pickle, time
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
from muzero import initial_frame_stack, push_frame, preprocess_frame

ARM = "effzero-repro-refresh5x-communal"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
EFF = (0, 2, 3); GROUPS = ((0, 1), (2, 4), (3, 5)); K = 5
R = ROOT / "benchmarks/results"
MODELS = [  # label, final return, source
    ("archfix s0 @90", "+17.1", R / "pong-archfix-2026-08-30/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"),
    ("checkD s0 @90", "-11.4", R / "pong-archfix-ray-2026-09-11/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"),
    ("control s1 @90", "-14.0", R / "pong-archfix-seq-s1-2026-09-13/weights-pong-skip4-effzero-repro-refresh5x-communal-seed1.pth"),
    ("r12 s0 @90", "-11.4", R / "pong-r12-2026-09-14/weights-pong-skip4-effzero-repro-vtree-communal8-r12-seed0.pth"),
    ("keepnew90 @87", "-9.6", R / "pong-keepnew90-2026-09-08/weights-pong-skip4-effzero-repro-async-keepnew-seed1.pth"),
    ("vsat ckpt @10", "-19.0", R / "pong-vsat-2026-09-03/resume-pong-skip4-effzero-repro-vtree-saturate-seed0.ckpt"),
    ("vtree ckpt @26", "-18.9", R / "pong-vtree-2026-09-02/resume-pong-skip4-effzero-repro-vtree-refresh-seed0.ckpt"),
    ("arch1 ckpt @48", "-15.3", R / "pong-arch1-2026-08-29/resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt"),
]


def load_nets(path):
    path = str(path)
    if path.endswith(".pth"):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {n: d[n] for n in ablation.NET_NAMES}
    with open(path, "rb") as f:  # streamed container: head pickle only, skip the replay frames
        assert f.read(len(ablation.CKPT_STREAM_MAGIC)) == ablation.CKPT_STREAM_MAGIC
        head = pickle.load(f)
    return head["nets"]


def eff(p):
    return np.stack([p[:, list(g)].sum(1) for g in GROUPS], 1)


with ablation.arm_config(ARM):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    nets = run.trainer
    dev = ablation.DEVICE

    def put(state):
        for n in ablation.NET_NAMES:
            getattr(nets, n).load_state_dict(state[n])
        run.set_train_mode(False)

    # ---- shared state set: raw-prior play of the winner, of a plateau run, and random play
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
    for label, src_idx, n, seed in (("archfix-prior", 0, 160, 11), ("control-prior", 2, 160, 12), ("random", None, 120, 13)):
        if src_idx is None:
            pol = lambda s: int(rng.integers(6))
        else:
            put(load_nets(MODELS[src_idx][2]))
            pol = prior_policy
        got = collect(pol, n, seed)
        states += [(label, s, st) for s, st in got]
        print(f"collected {len(got)} states from {label}", flush=True)

    # ---- counterfactual rollouts
    S0, S1, SRC, distinct = [], [], [], []
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
        S0.append(st0); S1.append(seqs); SRC.append(label)
        newest = seqs[:, :, -3:]  # newest frame: (3 actions, K, 3, 84, 84)
        distinct.append([all(np.any(newest[i, k] != newest[j, k]) for i, j in ((0, 1), (0, 2), (1, 2))) for k in range(K)])
    S0 = np.stack(S0); S1 = np.stack(S1); SRC = np.array(SRC); distinct = np.array(distinct)
    print(f"usable states {len(S0)}; pixel-distinct futures at k=1..{K}: {distinct.mean(0).round(3).tolist()}", flush=True)
    for lab in ("archfix-prior", "control-prior", "random"):
        print(f"  {lab}: {np.sum(SRC == lab)} states", flush=True)


    def erank(X):
        X = X - X.mean(0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False) ** 2
        p = s / s.sum()
        return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))

    @torch.no_grad()
    def score_latent(bs=64):
        N = len(S0)
        D = np.zeros((N, 3, 3, K))            # L2 distance g^k(s,a_pred) -> repr(true_k(a_true))
        Dsame = np.zeros((N, K))              # mean pairwise distance between the three TRUE latents
        Dpred = np.zeros((N, K))              # mean pairwise distance between the three PREDICTED latents
        Dstep = np.zeros((N, K))              # distance root latent -> true_k latent (how far the state moves)
        proj_all, lat_all, prd_all = [], [], []
        pairs = ((0, 1), (0, 2), (1, 2))
        for i in range(0, N, bs):
            x0 = torch.tensor(S0[i:i + bs], dtype=torch.float32, device=dev) / 255.0
            n = len(x0)
            lat0 = nets.representation(x0)
            proj_all.append(nets.projector(lat0, with_prediction=False).cpu().numpy())
            lat_all.append(lat0.flatten(1).cpu().numpy())
            pr = torch.zeros(3, K, n, lat0[0].numel(), device=dev); tr = torch.zeros_like(pr)
            for ai, a in enumerate(EFF):
                lat = lat0
                for k in range(K):
                    lat, _, _ = nets.dynamics(lat, torch.full((n,), a, dtype=torch.long, device=dev))
                    pr[ai, k] = lat.flatten(1)
                    xt = torch.tensor(S1[i:i + n, ai, k], dtype=torch.float32, device=dev) / 255.0
                    tr[ai, k] = nets.representation(xt).flatten(1)
            prd_all.append(pr[1, 0].cpu().numpy())
            for p in range(3):
                for q in range(3):
                    D[i:i + n, p, q] = (pr[p] - tr[q]).norm(dim=2).T.cpu().numpy()
            Dsame[i:i + n] = np.mean([(tr[p] - tr[q]).norm(dim=2).T.cpu().numpy() for p, q in pairs], 0)
            Dpred[i:i + n] = np.mean([(pr[p] - pr[q]).norm(dim=2).T.cpu().numpy() for p, q in pairs], 0)
            Dstep[i:i + n] = np.mean([(tr[p] - lat0.flatten(1)[None]).norm(dim=2).T.cpu().numpy() for p in range(3)], 0)
        return D, Dsame, Dpred, Dstep, np.concatenate(proj_all), np.concatenate(lat_all), np.concatenate(prd_all)

    for label, ret, src in MODELS:
        put(load_nets(src))
        D, Dsame, Dpred, Dstep, PJ, LT, PD = score_latent()
        pick = D.argmin(2)
        hit = np.stack([pick[:, a] == a for a in range(3)], 1)
        dmask = np.repeat(distinct[:, None, :], 3, 1)
        own = np.stack([D[:, a, a] for a in range(3)], 1)
        pjn = PJ / np.linalg.norm(PJ, axis=1, keepdims=True)
        cc = pjn @ pjn.T
        cross = cc[~np.eye(len(cc), dtype=bool)].mean()
        print(f"\n### {label}  (final return {ret})", flush=True)
        print("  k:                 " + "".join(f"{k+1:>8d}" for k in range(K)))
        print("  latent disc acc    " + "".join(f"{hit[:, :, k][dmask[:, :, k]].mean():8.3f}" for k in range(K)) + "   (chance 0.333)")
        print("  pred err ||g-true||" + "".join(f"{own[:, :, k].mean():8.3f}" for k in range(K)))
        print("  true action spread " + "".join(f"{Dsame[:, k].mean():8.3f}" for k in range(K)) + "   (distance between the 3 true futures)")
        print("  pred action spread " + "".join(f"{Dpred[:, k].mean():8.3f}" for k in range(K)) + "   (distance between the 3 predicted futures)")
        print("  true motion        " + "".join(f"{Dstep[:, k].mean():8.3f}" for k in range(K)) + "   (root -> true_k)")
        print(f"  projection: mean cos between DIFFERENT states {cross:.3f}; eff. rank proj {erank(PJ):.1f}/{PJ.shape[1]}  "
              f"latent {erank(LT):.1f}  dyn-latent {erank(PD):.1f}   latent norm {np.linalg.norm(LT, axis=1).mean():.2f}", flush=True)
    print("\nDONE")
