"""Self-play actor for the concurrent (Ray) harness mode.

The actor owns its OWN network copies and environment, applies the arm's
overrides inside its own process (a Ray worker is a fresh interpreter --
the driver's arm_config setattrs never reach it), and plays episodes while
the driver trains. Weights arrive by explicit set_weights() sync at
iteration boundaries, so the acting policy's staleness is identical to the
sequential harness (episode k plays with end-of-iteration-(k-1) weights);
what changes is data incorporation, which lags one iteration (episode k
first trains in iteration k+1).

The class is deliberately UNDECORATED: the driver wraps it with
ray.remote(num_gpus=...) at creation, and tests run it in ray local mode.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch


class SelfPlayActor:
    def __init__(self, overrides, action_space, seed, run_length_steps,
                 env_name, frameskip, episode_cap):
        import muzero as mz
        self.mz = mz
        # The arm's world, rebuilt locally: constants first (network
        # factories and the search read them at call time), then the
        # run-length-keyed schedule globals the driver's
        # _run_length_globals would otherwise own.
        for key, value in overrides.items():
            if hasattr(mz, key):
                setattr(mz, key, value)
        mz.TEMPERATURE_TOTAL_STEPS = run_length_steps
        mz.OFFPOLICY_DECAY_STEPS = max(1, int(0.3 * run_length_steps))
        np.random.seed(seed * 7919 + 17)
        torch.manual_seed(seed * 7919 + 17)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.episode_cap = episode_cap

        self.searcher = mz.MCTSHelper(action_space=action_space,
                                      device=self.device, network_ref=None,
                                      storage_ref=None, enable_logging=False)
        nm = self.searcher.network_manager
        for net in (nm.representation, nm.prediction, nm.dynamics,
                    nm.value_prefix):
            net.eval()

        # Mirror Run.__init__'s env stack exactly.
        import gymnasium as gym
        from EpisodicLifeEnv import EpisodicLifeEnv
        if mz.USE_ATARI_WRAPPERS:
            from atari_wrappers import paper_atari_env
            self.env = EpisodicLifeEnv(paper_atari_env(env_name,
                                                       skip=frameskip or 4))
        else:
            kwargs = {"frameskip": frameskip} if frameskip else {}
            self.env = EpisodicLifeEnv(gym.make(env_name, **kwargs))

        # A minimal Run shell so ablation.Run.play_episode runs UNMODIFIED
        # here -- one episode implementation, not two drifting copies. Its
        # dependency surface is pinned by tests/test_ray_harness.py: the
        # self.* attributes (env, searcher, env_steps) AND the ablation
        # module globals it reads (ACTIONS, EPISODE_CAP; IMAGE_RESHAPE is a
        # muzero re-export), which a fresh worker interpreter would
        # otherwise leave at their MsPacman defaults -- the first test run
        # crashed exactly there, 5-action choice against a 6-action policy.
        import ablation
        ablation.ACTIONS = action_space
        ablation.EPISODE_CAP = episode_cap
        ablation.ENV_NAME = env_name
        ablation.FRAMESKIP = frameskip
        self._ablation = ablation
        self.shell = object.__new__(ablation.Run)
        self.shell.env = self.env
        self.shell.searcher = self.searcher
        self.shell.env_steps = 0

    def set_weights(self, state_dicts, training_steps):
        nm = self.searcher.network_manager
        for name, state in state_dicts.items():
            net = getattr(nm, name)
            net.load_state_dict(state)
            net.eval()
        # The staged acting temperature keys on training progress.
        nm.training_steps = training_steps
        return training_steps

    def play(self, episode_seed):
        """One self-play episode; returns (trajectory, terminated, steps)."""
        self.env.reset(seed=int(episode_seed))
        before = self.shell.env_steps
        traj, _, _, terminated = self._ablation.Run.play_episode(
            self.shell, greedy=False, cap=self.episode_cap)
        return traj, terminated, self.shell.env_steps - before

    def eval_episodes(self, seeds):
        """Greedy evaluation episodes; returns their returns. Env steps from
        eval are protocol-exempt (the driver never counts them) and the
        shell's counter is scratch here."""
        returns = []
        for s in seeds:
            self.env.reset(seed=int(s))
            returns.append(self._ablation.Run.play_episode(
                self.shell, greedy=True, cap=self.episode_cap)[1])
        return returns

    def get_constant(self, name):
        """Test probe: what this PROCESS's muzero constant resolves to --
        counterfactual-proof that overrides crossed the process boundary."""
        return getattr(self.mz, name)
