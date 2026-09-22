"""Is the rows' proj_erank collapse real, or an eval-mode BatchNorm artifact?

evaluate_diagnostics projects in EVAL mode (running statistics). If a
projector dimension's running_var is stale and tiny, that dimension's
outputs explode, dominate the SVD, and read as rank ~1 -- a measurement
artifact, not collapse. This compares, on the SAME states from a snapshot:
eval-mode projections, batch-statistics (train-mode BN, on a throwaway copy)
projections, per-dimension spread, and the raw latent's effective rank.
CPU only, low priority (the edge runs own the GPU)."""
import copy, os, sys, pathlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import psutil
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
from muzero import initial_frame_stack, push_frame, preprocess_frame
torch.set_num_threads(4)

ARM = "effzero-repro-refresh5x-communal-edge"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
R = ROOT / "benchmarks/results"
SNAPS = [(f"s{s} @{it}", R / f"pong-edge-s{s}-2026-09-21/snapshots-pong-skip4-{ARM}-seed{s}/weights-iter{it:03d}.pth")
         for s, its in ((1, (10, 15, 20, 25)), (0, (10, 20))) for it in its]
SNAPS += [("archfix +17.1", R / "pong-archfix-2026-08-30/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"),
          ("checkD -11.4", R / "pong-archfix-ray-2026-09-11/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"),
          ("ctrl s1 -14", R / "pong-archfix-seq-s1-2026-09-13/weights-pong-skip4-effzero-repro-refresh5x-communal-seed1.pth"),
          ("r12 -11.4", R / "pong-r12-2026-09-14/weights-pong-skip4-effzero-repro-vtree-communal8-r12-seed0.pth")]
SNAPS += [("vsat ck @10", R / "pong-vsat-2026-09-03/resume-pong-skip4-effzero-repro-vtree-saturate-seed0.ckpt"),
          ("vtree ck @26", R / "pong-vtree-2026-09-02/resume-pong-skip4-effzero-repro-vtree-refresh-seed0.ckpt"),
          ("throttle @26", R / "pong-throttle-s1-2026-09-07/resume-pong-skip4-effzero-repro-async-throttle-seed1.ckpt"),
          ("satguard @34", R / "pong-satguard-s1-2026-09-05/resume-pong-skip4-effzero-repro-vtree-sat-guard-seed1.ckpt"),
          ("arch1 ck @48", R / "pong-arch1-2026-08-29/resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt"),
          ("satstale @50", R / "pong-satstale-2026-09-02/resume-pong-skip4-effzero-repro-vtree-sat-stale-seed0.ckpt")]


def erank(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


with ablation.arm_config(ARM):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    nets = run.trainer
    env = ablation.build_env("Pong-v4", 4)
    rng = np.random.default_rng(0)

    def load(path):
        if str(path).endswith(".ckpt"):   # streamed container: head pickle only
            import pickle
            with open(path, "rb") as f:
                assert f.read(len(ablation.CKPT_STREAM_MAGIC)) == ablation.CKPT_STREAM_MAGIC
                d = pickle.load(f)["nets"]
        else:
            d = torch.load(path, map_location="cpu", weights_only=False)
        for n in ablation.NET_NAMES:
            getattr(nets, n).load_state_dict(d[n])
        run.set_train_mode(False)

    # One shared state set: raw-prior play of the latest seed-1 snapshot.
    load(SNAPS[3][1])
    states, obs = [], env.reset(seed=21)[0]
    stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE)); t = 0
    while len(states) < 384:
        stack = push_frame(stack, preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        if t >= 16 and t % 3 == 0:
            states.append(stack.copy())
        p, _ = run.searcher.network_manager.initial_inference(stack)
        obs, _, term, trunc, _ = env.step(int(rng.choice(6, p=p / p.sum()))); t += 1
        if term or trunc:
            obs = env.reset()[0]; t = 0
            stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    X = torch.tensor(np.stack(states), dtype=torch.float32) / 255.0
    print(f"states {len(X)}", flush=True)

    for label, path in SNAPS:
        if not path.exists():
            print(f"{label}: missing {path.name}"); continue
        load(path)
        with torch.no_grad():
            lat = nets.representation(X)
            pe = nets.projector(lat, with_prediction=False).numpy()
            tp = copy.deepcopy(nets.projector).train()      # batch statistics, throwaway copy
            pt = tp(lat, with_prediction=False).numpy()
        bns = [m for m in nets.projector.projection if isinstance(m, torch.nn.BatchNorm1d)]
        rv = bns[-1].running_var.numpy()
        sd_e = pe.std(0)
        top = np.sort(sd_e)[::-1]
        print(f"{label:13s} erank eval {erank(pe):6.2f}  batch-stats {erank(pt):6.2f}  latent {erank(lat.flatten(1).numpy()):6.1f}  "
              f"| eval per-dim std: top1 {top[0]:.3g} top2 {top[1]:.3g} median {np.median(sd_e):.3g}  "
              f"| final-BN running_var min {rv.min():.3g} median {np.median(rv):.3g}  "
              f"| dead |g|<0.01 {(bns[-1].weight.abs() < 0.01).float().mean():.3f}", flush=True)
    print("DONE")
