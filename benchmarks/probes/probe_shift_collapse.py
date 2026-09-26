"""Audit H1 (2026-09-24): did the projector collapse onto the augmentation shift?

augment_observations shares ONE (dx, dy, intensity) draw between the online
root and every SimSiam target; EfficientZero's released code draws the root
and the targets independently (core/train.py:73-75). With a shared draw the
shift is perfectly predictable from online to target, so non-contrastive
training may grow exactly those directions. If the collapsed projector's ~2
surviving dimensions ARE (dx, dy), then on a grid of states x shifts the
shift factor should explain most of the projection variance -- and far more
than in a pre-collapse or healthy projector.

Batch-statistics projections (throwaway train-mode copy), as the loss sees
them. No intensity jitter, so shift is the only nuisance factor.
CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import sys, pathlib, copy
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
from muzero import initial_frame_stack, push_frame, preprocess_frame
torch.set_num_threads(6)

ARM = "effzero-repro-refresh5x-communal"
EDGE = "effzero-repro-refresh5x-communal-edge"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
R = ROOT / "benchmarks/results"
snap = lambda s, it: R / f"pong-edge-s{s}-2026-09-21/snapshots-pong-skip4-{EDGE}-seed{s}/weights-iter{it:03d}.pth"
MODELS = [
    ("edge s0 @10 (pre-collapse)", snap(0, 10)),
    ("edge s0 @20", snap(0, 20)),
    ("edge s0 @50", snap(0, 50)),
    ("edge s1 @10 (pre-collapse)", snap(1, 10)),
    ("edge s1 @55", snap(1, 55)),
    ("archfix @90 +17.1 (healthy)", R / "pong-archfix-2026-08-30/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"),
    ("control s1 @90 -14.0", R / "pong-archfix-seq-s1-2026-09-13/weights-pong-skip4-effzero-repro-refresh5x-communal-seed1.pth"),
]
if len(sys.argv) > 1:   # snapshot paths replace the edge series; references stay
    MODELS = [(f"{pathlib.Path(q).parent.parent.name[5:]} @{int(pathlib.Path(q).stem[-3:])}", pathlib.Path(q))
              for q in sys.argv[1:]] + MODELS[-2:]
N_STATES, SHIFT, GRID = 120, 4, (0, 2, 4, 6, 8)


def erank(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    p = s / s.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def decompose(Z):
    """Z: (N states, S shifts, D). Fractions of total variance explained by the
    shift factor (per-shift means) and the state factor (per-state means), over
    all dims and within the top-2 principal components."""
    N, S, D = Z.shape
    Zc = Z - Z.reshape(-1, D).mean(0)
    tot = (Zc ** 2).sum() / (N * S)
    shift_part = (Zc.mean(0) ** 2).sum() / S
    state_part = (Zc.mean(1) ** 2).sum() / N
    _, _, Vt = np.linalg.svd(Zc.reshape(-1, D), full_matrices=False)
    P = Zc @ Vt[:2].T                                    # (N, S, 2)
    ptot = (P ** 2).sum() / (N * S)
    pshift = (P.mean(0) ** 2).sum() / S
    pstate = (P.mean(1) ** 2).sum() / N
    return shift_part / tot, state_part / tot, pshift / ptot, pstate / ptot


with ablation.arm_config(ARM):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    nets = run.trainer
    assert ablation.DEVICE.type == "cpu", ablation.DEVICE

    env = ablation.build_env("Pong-v4", 4)
    rng = np.random.default_rng(0)
    states, t = [], 0
    obs, _ = env.reset(seed=21)
    stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    while len(states) < N_STATES:
        stack = push_frame(stack, preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        if t >= 16 and t % 6 == 0:
            states.append(stack.copy())
        obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
        t += 1
        if term or trunc:
            obs, _ = env.reset(); t = 0
            stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    X = torch.tensor(np.stack(states), dtype=torch.float32) / 255.0      # (N, C, H, W)
    H, W = X.shape[-2:]
    padded = F.pad(X, (SHIFT,) * 4, mode="replicate")
    shifts = [(dx, dy) for dx in GRID for dy in GRID]
    Xs = torch.stack([padded[:, :, dy:dy + H, dx:dx + W] for dx, dy in shifts], 1)  # (N, S, C, H, W)
    S = len(shifts)
    print(f"{N_STATES} states x {S} shifts (dx, dy in {GRID})\n", flush=True)
    print(f"{'model':30s} {'proj: shift':>11s} {'state':>6s}  {'top-2 PCs: shift':>16s} {'state':>6s}   "
          f"{'latent: shift':>13s} {'state':>6s}   {'bs-rank unaug':>13s} {'aug':>6s}", flush=True)
    for label, path in MODELS:
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        for n in ablation.NET_NAMES:
            getattr(nets, n).load_state_dict(d[n])
        # Plain attribute, not in the state_dict: honour the file's own spec.
        renorm = d.get("spec", {}).get("LATENT_RENORMALIZE", True)
        nets.representation.renormalize_output = renorm
        nets.dynamics.renormalize_output = renorm
        run.set_train_mode(False)
        tp = copy.deepcopy(nets.projector).train()                        # batch statistics
        with torch.no_grad():
            lat = nets.representation(Xs.flatten(0, 1))                    # (N*S, ...)
            proj = tp(lat, with_prediction=False).numpy().reshape(N_STATES, S, -1)
            latn = lat.flatten(1).numpy().reshape(N_STATES, S, -1)
            unaug = tp(nets.representation(X), with_prediction=False).numpy()
        sh, st, psh, pst = decompose(proj)
        lsh, lst, _, _ = decompose(latn)
        print(f"{label:30s} {sh:11.3f} {st:6.3f}  {psh:16.3f} {pst:6.3f}   {lsh:13.3f} {lst:6.3f}   "
              f"{erank(unaug):13.1f} {erank(proj.reshape(N_STATES * S, -1)):6.1f}", flush=True)
    print("\nshift/state = fraction of variance explained by that factor (the rest is interaction)."
          "\nDONE")
