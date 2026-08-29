"""Watch a trained agent play, and write it to video.

    python experiments/render.py --checkpoint benchmarks/results/weights-per-step-seed0.pth
    python experiments/render.py --checkpoint ... --live      # window instead of a file

Note what this actually shows. The agent acts with a single forward pass through
the policy head -- there is no tree search on the acting path, which is the
project's core real-time adaptation (see README). MCTS runs offline during
training to produce better targets, so what you are watching is the policy head
that re-analysis taught, not a search.
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import muzero
from muzero import IMAGE_RESHAPE, initial_frame_stack, preprocess_frame, push_frame

ACTIONS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    spec = ck.get("spec", {})
    linear = spec.get("_linear_support")
    value_sup = muzero.VALUE_CEILING if linear else muzero.VALUE_SUPPORT
    value_bins = 51 if linear else muzero.VALUE_BINS

    # The weights file records its arm spec: apply its overrides (the
    # architecture knobs above all) before building, exactly as the harness
    # did when it trained them -- a smallnet-arm checkpoint builds 64-channel
    # nets here, not the module-default 128 (audit 2026-08-28: this builder
    # was missed by the knob wiring and import-time-captured the width).
    for key, value in spec.items():
        if not key.startswith("_") and hasattr(muzero, key):
            setattr(muzero, key, value)

    rep = muzero.build_representation().to(DEVICE)
    pred = muzero.build_prediction(ACTIONS, support_size=value_sup,
                                   num_bins=value_bins).to(DEVICE)
    rep.load_state_dict(ck["representation"])
    pred.load_state_dict(ck["prediction"])
    rep.eval()
    pred.eval()
    return rep, pred, ck


def act(rep, pred, stack, greedy):
    with torch.no_grad():
        x = torch.tensor(stack, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
        logits, value, _ = pred(rep(x))
        policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
    action = int(policy.argmax()) if greedy else int(np.random.choice(ACTIONS, p=policy))
    return action, policy, float(value.item())


def play(rep, pred, greedy=True, cap=3000, live=False, seed=0):
    """One full game -- all lives, not the per-life episode training uses."""
    import gymnasium as gym
    env = gym.make("MsPacman-v4", render_mode="human" if live else "rgb_array")

    obs, _ = env.reset(seed=seed)
    stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
    frames, actions, values = [], [], []
    score, lives, deaths, done, steps = 0.0, None, 0, False, 0

    while not done and steps < cap:
        stack = push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE))
        action, policy, value = act(rep, pred, stack, greedy)
        actions.append(action)
        values.append(value)

        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        score += float(reward)
        steps += 1

        now = env.unwrapped.ale.lives()
        if lives is not None and now < lives:
            deaths += 1
        lives = now

        if not live:
            frames.append(env.render())

    env.close()
    return dict(frames=frames, score=score, steps=steps, deaths=deaths,
                actions=actions, values=values)


def write_video(frames, path, fps=30, scale=3):
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * scale, h * scale))
    for f in frames:
        big = cv2.resize(f, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    writer.release()


def write_contact_sheet(frames, path, cols=5, rows=3, scale=2):
    """Evenly spaced stills, so the run can be inspected without playing video."""
    picks = np.linspace(0, len(frames) - 1, cols * rows).astype(int)
    h, w, _ = frames[0].shape
    sheet = np.zeros((rows * h * scale, cols * w * scale, 3), dtype=np.uint8)
    for i, idx in enumerate(picks):
        r, c = divmod(i, cols)
        tile = cv2.resize(frames[idx], (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"t={idx}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        sheet[r * h * scale:(r + 1) * h * scale, c * w * scale:(c + 1) * w * scale] = tile
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--sample", action="store_true", help="sample the policy instead of argmax")
    ap.add_argument("--live", action="store_true", help="open a window instead of writing a file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep, pred, ck = load(args.checkpoint)
    print(f"loaded {pathlib.Path(args.checkpoint).name}: arm={ck.get('arm')} "
          f"seed={ck.get('seed')} training_steps={ck.get('training_steps')}")
    print(f"acting: {'sampled' if args.sample else 'greedy (argmax)'}, single forward pass, no search")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        r = play(rep, pred, greedy=not args.sample, live=args.live, seed=args.seed + ep)
        counts = np.bincount(r["actions"], minlength=ACTIONS)
        names = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]
        print(f"\nepisode {ep}: score {r['score']:.0f} over {r['steps']} steps, "
              f"{r['deaths']} deaths")
        print("  action mix: " + "  ".join(
            f"{n}={c / max(len(r['actions']), 1):.0%}" for n, c in zip(names, counts)))
        print(f"  predicted value: mean {np.mean(r['values']):.1f}, "
              f"range {np.min(r['values']):.1f}..{np.max(r['values']):.1f}")

        if not args.live and r["frames"]:
            tag = f"{ck.get('arm', 'agent')}-ep{ep}"
            write_video(r["frames"], out / f"{tag}.mp4")
            write_contact_sheet(r["frames"], out / f"{tag}.png")
            print(f"  wrote {tag}.mp4 and {tag}.png ({len(r['frames'])} frames)")


if __name__ == "__main__":
    main()
