"""Curriculum feasibility scan: attainable Pong return vs frameskip.

    python experiments/skip_scan.py [--episodes 5] [--skips 4 8 12 20 30]

For each skip, plays episodes with (a) a random policy and (b) a scripted
paddle-tracker that reads ball/paddle positions from the raw frame. The
tracker approximates "competent reactive play"; where even IT floors near
-21, that skip is unlearnable and the curriculum must start lower. The
highest skip where the tracker clearly beats random is the recommended
first rung. CPU-only: no networks, safe to run beside a training job.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from curriculum_env import PONG_MINIMAL_ACTIONS, make_curriculum_env

NOOP, UP, DOWN = 0, 1, 2   # indices into PONG_MINIMAL_ACTIONS


def tracker_action(raw_rgb):
    """Move the right paddle toward the ball's row, from raw pixels.
    Playfield rows 34:194; right paddle lives in columns 140:144 (green
    92,186,92); the ball is the white (236,236,236) blob left of it."""
    field = raw_rgb[34:194]
    ball = np.argwhere((field[:, 20:140] == (236, 236, 236)).all(axis=2))
    paddle = np.argwhere((field[:, 140:144] == (92, 186, 92)).all(axis=2))
    if len(ball) == 0 or len(paddle) == 0:
        return NOOP
    ball_y = ball[:, 0].mean()
    paddle_y = paddle[:, 0].mean()
    if ball_y < paddle_y - 4:
        return UP
    if ball_y > paddle_y + 4:
        return DOWN
    return NOOP


def play(env, policy, rng, cap=20000):
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    total, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < cap:
        if policy == "random":
            a = int(rng.integers(env.action_space.n))
        else:
            a = tracker_action(env.last_raw_rgb)
        obs, r, terminated, truncated, _ = env.step(a)
        total += r
        steps += 1
    return total, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--skips", type=int, nargs="+", default=[4, 8, 12, 20, 30])
    ap.add_argument("--env", default="Pong-v4")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    print(f"{'skip':>5} {'policy':>8} {'return':>14} {'decisions/ep':>13}")
    for skip in args.skips:
        for policy in ("random", "tracker"):
            env = make_curriculum_env(args.env, skip=skip,
                                      action_map=PONG_MINIMAL_ACTIONS)
            rets, lens = zip(*[play(env, policy, rng)
                               for _ in range(args.episodes)])
            env.close()
            print(f"{skip:>5} {policy:>8} {np.mean(rets):>8.1f} ± {np.std(rets):<4.1f}"
                  f" {np.mean(lens):>11.0f}", flush=True)


if __name__ == "__main__":
    main()
