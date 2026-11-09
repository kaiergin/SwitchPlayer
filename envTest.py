"""Drive the live Tetris environment with random actions, for smoke-testing
the capture card, the serial link and the mask detectors together.

The --reward / --env / --pregame / --epochs flags are gone: they trained the
three CNNs, which `tetris_masks` replaced. To check the detectors, run
`python tetris_masks.py --eval` (against the labelled corpus, no hardware) or
`--demo` (frame-by-frame with the ROIs drawn on).
"""
import argparse

import numpy as np

from tetris import Tetris

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--episodes', type=int, default=3)
parser.add_argument('--capture-device', type=int, default=0)
parser.add_argument('--quiet', action='store_true',
                    help='suppress the per-event log')
args = parser.parse_args()

env = Tetris(capture_device=args.capture_device, debug=not args.quiet)
n_actions = env.action_space.n

try:
    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        # randint's upper bound is exclusive, so n_actions covers every legal
        # action. The previous `randint(0, n_actions - 1)` here was applied to
        # an already-decremented space and never sampled BTN_R at all.
        done, total, steps, events = False, 0.0, 0, []
        while not done:
            obs, reward, done, info = env.step(np.random.randint(0, n_actions))
            total += reward
            steps += 1
            if info['event']:
                events.append(info['event'])
        print(f"episode {episode}: {steps} steps, reward {total:+.2f}, "
              f"events {events or '[]'}")
finally:
    env.close()
