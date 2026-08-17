"""The benchmark protocol's remaining env wrappers, baselines order.

EfficientZero (and the whole SimPLe lineage) builds its Atari env as
raw (no frameskip) -> NoopReset(30) -> MaxAndSkip(4) -> EpisodicLife.
This repo previously used ALE's internal frameskip, which returns only the
LAST frame of the skip -- no max-pool over the final two frames, so Atari
2600 sprite flicker (ghosts, pellets, the ball) is randomly absent from
observations -- and no no-op starts, which after the fixed-skip change would
make greedy evaluation fully deterministic.
"""
import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """Take a random number (1..noop_max) of no-op actions after reset, for
    start-state diversity. Standard baselines wrapper."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat the action `skip` raw frames, sum rewards, and return the
    element-wise MAX of the last two frames -- the flicker defence ALE's
    internal frameskip (which returns only the last frame) lacks."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._buffer = np.zeros((2, *env.observation_space.shape),
                                dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._buffer[0] = observation
        self._buffer[1] = observation
        return observation, info

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if i >= self._skip - 2:
                self._buffer[i - (self._skip - 2)] = observation
            if terminated or truncated:
                break
        return self._buffer.max(axis=0), total, terminated, truncated, info


def paper_atari_env(env_name, skip=4):
    """The lineage's env stack, minus EpisodicLife (callers add it so the
    training/eval distinction stays theirs to make)."""
    env = gym.make(env_name, frameskip=1)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    return env
