"""Faithful-arm prior flattening (2026-09-25): is it the BatchNorm train/eval
mismatch that no-renorm latents create?

Without min-max renormalisation, training-mode latents grow with unroll
depth (~5x over K=5), and every head's BN running statistics are averaged
over depths 0..K. At acting time the ROOT latent (depth 0) is normalised by
those depth-averaged statistics, which would shrink the policy logits and
flatten the prior even while the search targets it trains on stay sharp.

Per snapshot, on the same root states (random-play Pong frames):
  H eval      prior entropy, eval-mode BN (what acting and search see)
  H batch     prior entropy with batch statistics of a root-only batch
              (throwaway train-mode copies: what a depth-0-only BN would give)
  lat0 / lat5 root latent RMS vs a 5-step dynamics rollout's latent RMS
              (train mode, the scale spread the running stats average over)
  bn var      prediction head's first BN: running_var mean / root batch var mean
  |W_pi|      policy-logit layer weight norm (weight-decay shrink check)
CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import sys, pathlib, copy
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
from muzero import initial_frame_stack, push_frame, preprocess_frame
torch.set_num_threads(6)

ARM = "effzero-repro-refresh5x-communal"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
R = ROOT / "benchmarks/results"
PATHS = [pathlib.Path(p) for p in sys.argv[1:]]


def entropy(logits):
    p = F.softmax(logits, 1)
    return float(-(p * torch.log(p + 1e-12)).sum(1).mean())


with ablation.arm_config(ARM):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    nets = run.trainer
    assert ablation.DEVICE.type == "cpu", ablation.DEVICE
    env = ablation.build_env("Pong-v4", 4)
    rng = np.random.default_rng(0)
    states, t = [], 0
    obs, _ = env.reset(seed=31)
    stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    while len(states) < 256:
        stack = push_frame(stack, preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        if t >= 16 and t % 5 == 0:
            states.append(stack.copy())
        obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
        t += 1
        if term or trunc:
            obs, _ = env.reset(); t = 0
            stack = initial_frame_stack(preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    X = torch.tensor(np.stack(states), dtype=torch.float32) / 255.0
    acts = torch.tensor(rng.integers(6, size=(5, len(X))))

    print(f"{'model':34s} {'H eval':>7s} {'H batch':>8s} {'lat0':>7s} {'lat5':>7s} "
          f"{'bn var':>7s} {'|W_pi|':>7s}   (uniform H = {np.log(6):.3f})", flush=True)
    for path in PATHS:
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        for n in ablation.NET_NAMES:
            getattr(nets, n).load_state_dict(d[n])
        renorm = d.get("spec", {}).get("LATENT_RENORMALIZE", True)
        nets.representation.renormalize_output = renorm
        nets.dynamics.renormalize_output = renorm
        run.set_train_mode(False)
        with torch.no_grad():
            lat = nets.representation(X)
            logits_eval, _, _ = nets.prediction(lat)
            rep_b = copy.deepcopy(nets.representation).train()
            pred_b = copy.deepcopy(nets.prediction).train()
            dyn_b = copy.deepcopy(nets.dynamics).train()
            lat_b = rep_b(X)
            logits_batch, _, _ = pred_b(lat_b)
            z = lat_b
            for k in range(5):
                z, _, _ = dyn_b(z, acts[k])
            lat0 = float(lat_b.pow(2).mean().sqrt())
            lat5 = float(z.pow(2).mean().sqrt())
            bns = [m for m in nets.prediction.modules() if isinstance(m, torch.nn.BatchNorm2d)]
            bn = bns[0]
            h = lat
            # batch variance of the root activations entering that first BN
            feats = {}
            hook = bn.register_forward_hook(lambda m, i, o: feats.setdefault("x", i[0]))
            pred_b2 = nets.prediction
            pred_b2(lat)
            hook.remove()
            xv = feats["x"].var(dim=(0, 2, 3)).mean()
            bn_ratio = float(bn.running_var.mean() / xv)
            wpi = [p for n_, p in nets.prediction.named_parameters()
                   if p.ndim == 2 and p.shape[0] == 6]
            wn = float(wpi[-1].norm()) if wpi else float("nan")
        label = f"{path.parent.parent.name[5:]} @{int(path.stem[-3:])}"
        print(f"{label:34s} {entropy(logits_eval):7.3f} {entropy(logits_batch):8.3f} "
              f"{lat0:7.2f} {lat5:7.2f} {bn_ratio:7.2f} {wn:7.2f}", flush=True)
    print("DONE")
