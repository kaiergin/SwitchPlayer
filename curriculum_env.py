"""Curriculum env wrappers for the iterative-learning branch.

The observation is TWO grayscale channels instead of the repro line's
12-channel RGB stack:

  ch0  current state -- element-wise max of the last two RAW frames (the
       flicker defence MaxAndSkip exists for, doing only that one job)
  ch1  motion trail -- age-decayed |frame difference| accumulated over a
       window spanning ~2 decision intervals: moving objects appear as a
       fading streak whose bright end touches their current position, so
       position, direction, and speed (trail-segment spacing) are all
       explicit in the input instead of inferred from stacked frames.
       Motion-history images (Bobick & Davis) are the precedent.

Action semantics (the curriculum's contract): a selected action is held
for ALL `skip` raw frames of the decision, and the step reward is the SUM
of every raw-frame reward inside the window. As the curriculum lowers
`skip`, the trail window shortens and temporal resolution rises with no
code change.

Discounts are defined PER RAW FRAME and derived per decision with
`decision_discount` -- the only way value functions remain comparable
across curriculum stages with different skips (and the same convention
EfficientZero's released code uses: discount ** frame_skip).
"""
import collections

import cv2
import gymnasium as gym
import numpy as np

# Same guarded registration muzero.py performs: gymnasium >= 1.0 needs
# explicit ALE env registration; earlier versions register on import.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass

# ALE Pong full set is [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]; the
# game is fully playable with three (serve is automatic). NOTE: the
# harness's --actions N takes the FIRST N, which for Pong drops LEFT --
# use an explicit map instead.
PONG_MINIMAL_ACTIONS = (0, 2, 3)   # NOOP, UP, DOWN


def decision_discount(frame_discount, skip):
    """Per-decision gamma from a per-raw-frame gamma. Values learned at
    different curriculum skips are only comparable through this."""
    return frame_discount ** skip


class CurriculumFrameEnv(gym.Wrapper):
    """Action-repeat + two-channel grayscale observation, any skip.

    Wrap a RAW env (frameskip=1); this wrapper owns all temporal
    structure. `window_factor` decision-intervals of raw frames feed the
    trail; `trail_floor` sets the weight the oldest diff retains, from
    which the per-frame decay follows.
    """

    def __init__(self, env, skip=4, resize=(84, 84), window_factor=2,
                 trail_floor=0.1, action_map=None):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip
        self._resize = resize
        self._window = max(2, skip * window_factor)
        self._decay = trail_floor ** (1.0 / max(self._window - 1, 1))
        self._frames = collections.deque(maxlen=self._window)
        self._action_map = tuple(action_map) if action_map else None
        if self._action_map:
            self.action_space = gym.spaces.Discrete(len(self._action_map))
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(2, resize[1], resize[0]), dtype=np.uint8)
        self.last_raw_rgb = None   # for pixel-reading probes (skip_scan)

    @property
    def skip(self):
        return self._skip

    def _gray(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, self._resize, interpolation=cv2.INTER_AREA)

    def _observation(self):
        frames = list(self._frames)
        current = np.maximum(frames[-1], frames[-2]) if len(frames) > 1 else frames[-1]
        trail = np.zeros_like(frames[0], dtype=np.float32)
        # Newest diff carries weight 1, each step older multiplies by the
        # decay -- the streak fades toward the past, so its bright end
        # marks "now" and direction of travel is monotone in intensity.
        for age, i in enumerate(range(len(frames) - 1, 0, -1)):
            diff = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))
            trail += (self._decay ** age) * diff.astype(np.float32)
        trail = np.clip(trail, 0, 255).astype(np.uint8)
        return np.stack([current, trail])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_raw_rgb = obs
        self._frames.clear()
        # A buffer full of one frame: diffs are zero, the trail starts
        # empty rather than lighting the whole first scene as "motion".
        first = self._gray(obs)
        for _ in range(2):
            self._frames.append(first)
        return self._observation(), info

    def step(self, action):
        if self._action_map:
            action = self._action_map[action]
        total = 0.0
        terminated = truncated = False
        info = {}
        # The action is held for EVERY raw frame of the decision and the
        # step reward is the sum over the window -- the curriculum's
        # contract at any skip.
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_raw_rgb = obs
            total += float(reward)
            self._frames.append(self._gray(obs))
            if terminated or truncated:
                break
        return self._observation(), total, terminated, truncated, info


def make_curriculum_env(env_name, skip, noop_max=30, action_map=None,
                        window_factor=2):
    """Raw no-skip env -> NoopReset -> CurriculumFrameEnv, mirroring the
    paper stack's ordering (EpisodicLife stays the caller's choice)."""
    from atari_wrappers import NoopResetEnv
    env = gym.make(env_name, frameskip=1)
    env = NoopResetEnv(env, noop_max=noop_max)
    return CurriculumFrameEnv(env, skip=skip, action_map=action_map,
                              window_factor=window_factor)
