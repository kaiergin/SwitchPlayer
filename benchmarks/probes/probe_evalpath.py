"""Check A: do the sequential GPU evaluator (archfix's +17.1 path) and the
CPU eval actor (every parallel run's path) score the SAME weights the same?
Same arm, same weights, same episode seeds, same cap, greedy 50-sim search."""
import sys, time, pathlib
ROOT = pathlib.Path(r"Y:\SwitchPlayer")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch
import muzero, ablation, ray_actors

path = sys.argv[1]            # "gpu" (sequential Run.play_episode on cuda) | "cpu" (SelfPlayActor force_cpu)
n_eps = int(sys.argv[2]); cap = int(sys.argv[3])
W = ROOT / "benchmarks/results/pong-archfix-2026-08-30/weights-pong-skip4-effzero-repro-refresh5x-communal-seed0.pth"
d = torch.load(W, map_location="cpu", weights_only=False)
arm = d["arm"]; spec = ablation.ARMS[arm]
overrides = {k: v for k, v in spec.items() if not k.startswith("_")}
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, cap, 6
seeds = [0 * 100003 + 90 * 613 + 500000 + e for e in range(n_eps)]  # production eval seeds, seed 0, row 90
nets = ("representation", "dynamics", "prediction", "value_prefix")
with ablation.arm_config(arm):
    if path == "gpu":
        run = ablation.Run(arm, 0, spec)
        for n in nets + ("projector",):
            getattr(run.trainer, n).load_state_dict(d[n])
        run.trainer.training_steps = d["training_steps"]
        run.searcher.network_manager.training_steps = d["training_steps"]
        run.set_train_mode(False)
        dev = ablation.DEVICE
        def play(s):
            run.env.reset(seed=int(s))
            _, ret, steps, term = run.play_episode(greedy=True, cap=cap)
            return float(ret), int(steps), bool(term)
    else:
        schedule = {"TEMPERATURE_TOTAL_STEPS": muzero.TEMPERATURE_TOTAL_STEPS,
                    "OFFPOLICY_DECAY_STEPS": muzero.OFFPOLICY_DECAY_STEPS}
        actor = ray_actors.SelfPlayActor(overrides, 6, 0 * 8191 + 5000, schedule,
                                         "Pong-v4", 4, cap, server=None,
                                         server_set="eval", force_cpu=True)
        states = {n: {k: v.clone() for k, v in d[n].items()} for n in nets}
        actor.set_weights(states, d["training_steps"])
        dev = actor.device
        def play(s):
            return actor.eval_episodes([s])[0]
    print(f"[{path}] device={dev} sims={muzero.ACT_SEARCH_SIMS} cap={cap} seeds={seeds}", flush=True)
    out = []
    for s in seeds:
        t0 = time.time()
        ret, steps, term = play(s)
        out.append((ret, steps, term))
        print(f"[{path}] seed {s}: return {ret:+.0f} steps {steps} terminated {term} ({time.time()-t0:.0f}s)", flush=True)
    rets = [r for r, _, _ in out]
    print(f"[{path}] SUMMARY mean {np.mean(rets):+.2f} max {max(rets):+.0f} min {min(rets):+.0f} "
          f"len_mean {np.mean([s for _, s, _ in out]):.0f} capped {sum(1 for _, _, t in out if not t)}", flush=True)
