"""Live viewer: watch the CURRENT agent of a running (or finished)
harness run play, reloading the newest checkpoint between episodes.

    python watch_agent.py                        # newest satstale ckpt
    python watch_agent.py --dir benchmarks/results/pong-archfix-2026-08-30
    python watch_agent.py --sims 0               # raw policy head (fast)
    python watch_agent.py --sims 50              # full paper search (slow-mo)

Unlike experiments/render.py (final .pth weights, raw-frame pipeline,
policy-only acting), this reads the SPCHKPT2 resume checkpoint's head
mid-run, applies the ARM's overrides so net geometry and search knobs
match training, feeds observations through the SAME wrapper stack the
run trains on (noop starts + max-skip), and acts with the real greedy
search at --sims simulations (default 30: ~5 decisions/s on CPU, a
watchable slight slow-mo; 0 = raw policy at full speed). CPU-pinned so
a live training run is never contended. Window: q quits, any other key
is ignored; a new episode starts automatically and picks up the newest
checkpoint, so the window tracks the agent AS IT LEARNS.
"""
import argparse
import asyncio
import pathlib
import pickle
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "experiments"))

torch.set_num_threads(4)  # leave cores for a live run's actor

import muzero
from atari_wrappers import MaxAndSkipEnv, NoopResetEnv
import ablation

MAGIC = b"SPCHKPT2\n"
DEFAULT_DIR = "benchmarks/results/pong-satstale-2026-09-02"


def load_head(ckpt_path, retries=5):
    """The checkpoint head only (nets + counters; replay frames are
    never read). Retries cover the writer's os.replace window."""
    for attempt in range(retries):
        try:
            with open(ckpt_path, "rb") as f:
                if f.read(len(MAGIC)) != MAGIC:
                    raise ValueError(f"{ckpt_path} is not an SPCHKPT2 file")
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            if attempt == retries - 1:
                raise
            time.sleep(2)


def build_helper(head, actions):
    """Nets with the ARM's geometry/search knobs applied, exactly as the
    harness built them. Viewer process: overrides are not restored."""
    spec = ablation.ARMS[head["arm"]]
    for key, value in spec.items():
        if not key.startswith("_") and hasattr(muzero, key):
            setattr(muzero, key, value)
    helper = muzero.MCTSHelper(action_space=actions,
                               device=torch.device("cpu"),
                               network_ref=None, storage_ref=None,
                               enable_logging=False)
    nm = helper.network_manager
    for name in ("representation", "dynamics", "prediction", "value_prefix"):
        getattr(nm, name).load_state_dict(head["nets"][name])
        getattr(nm, name).eval()
    nm.training_steps = head.get("training_steps", 0)
    return helper


def build_env(env_name, skip):
    """paper_atari_env's exact stack, with a renderable base env."""
    env = gym.make(env_name, frameskip=1, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    return env


def hud(frame, lines, scale=4):
    h, w, _ = frame.shape
    big = cv2.resize(frame, (w * scale, h * scale),
                     interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
    for i, text in enumerate(lines):
        cv2.putText(big, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="results dir holding the resume-*.ckpt")
    ap.add_argument("--sims", type=int, default=30,
                    help="search sims per decision (0 = raw policy head)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="headless: run N decisions and exit (self-test)")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.dir)
    ckpts = sorted(run_dir.glob("resume-*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no resume-*.ckpt under {run_dir}")
    ckpt = ckpts[0]

    head = load_head(ckpt)
    env_name = head.get("env", "Pong-v4")
    skip = head.get("frameskip", 4) or 4
    probe = build_env(env_name, skip)
    actions = probe.action_space.n
    probe.close()
    helper = build_helper(head, actions)
    print(f"watching {head['arm']} @ iter {head.get('iteration')} "
          f"({head.get('training_steps')} train steps), {env_name} "
          f"A={actions}, sims={args.sims}", flush=True)

    env = build_env(env_name, skip)
    episode = 0
    while True:
        episode += 1
        obs, _ = env.reset(seed=args.seed + episode)
        stack = muzero.initial_frame_stack(
            muzero.preprocess_frame(obs, muzero.IMAGE_RESHAPE))
        score, steps, done, value = 0.0, 0, False, 0.0
        while not done:
            stack = muzero.push_frame(
                stack, muzero.preprocess_frame(obs, muzero.IMAGE_RESHAPE))
            if args.sims > 0:
                action, _, value = asyncio.run(helper.act_search(
                    stack, num_simulations=args.sims, greedy=True))
            else:
                nm = helper.network_manager
                _, pols, vals = nm.batch_inference(np.asarray(stack)[None])
                action, value = int(np.argmax(pols[0])), float(vals[0])
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            score += float(reward)
            steps += 1
            if args.smoke:
                if steps >= args.smoke:
                    print(f"SMOKE OK: {steps} decisions, score {score}, "
                          f"value {value:+.2f}", flush=True)
                    return
                continue
            frame = env.render()
            big = hud(frame, [
                f"iter {head.get('iteration')}  "
                f"train {head.get('training_steps')}",
                f"ep {episode}  step {steps}  score {score:+.0f}",
                f"V={value:+5.2f}  sims={args.sims}",
            ])
            cv2.imshow("SwitchPlayer -- live agent", big)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                env.close()
                cv2.destroyAllWindows()
                return
        print(f"episode {episode}: score {score:+.0f} in {steps} decisions",
              flush=True)
        # Pick up the newest weights for the next episode.
        try:
            new_head = load_head(ckpt)
            if new_head.get("training_steps") != head.get("training_steps"):
                head = new_head
                helper = build_helper(head, actions)
                print(f"reloaded: iter {head.get('iteration')} "
                      f"({head.get('training_steps')} steps)", flush=True)
        except Exception as e:
            print(f"reload skipped ({type(e).__name__}); keeping current "
                  f"weights", flush=True)


if __name__ == "__main__":
    main()
