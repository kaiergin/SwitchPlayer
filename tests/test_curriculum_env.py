"""CurriculumFrameEnv: action-repeat contract, two-channel observation."""
import gymnasium as gym
import numpy as np
import pytest

from curriculum_env import (CurriculumFrameEnv, PONG_MINIMAL_ACTIONS,
                            decision_discount)


class SpriteEnv(gym.Env):
    """A white 8x8 sprite marching right 6px per raw frame on black.
    Records every action it receives; 1.0 reward per raw frame."""

    observation_space = gym.spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, end_at=None, flicker=False):
        self.t = 0
        self.actions = []
        self.end_at = end_at
        self.flicker = flicker

    def _frame(self):
        f = np.zeros((210, 160, 3), dtype=np.uint8)
        if not (self.flicker and self.t % 2 == 1):
            x = min(8 + 6 * self.t, 150)
            f[100:108, x:x + 8] = 255
        return f

    def reset(self, **kwargs):
        self.t = 0
        self.actions = []
        return self._frame(), {}

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        done = self.end_at is not None and self.t >= self.end_at
        return self._frame(), 1.0, done, False, {}


def test_action_held_for_all_skip_frames_and_rewards_summed():
    """The curriculum's contract: one decision = the action repeated for
    every raw frame of the skip, reward = the sum over the window."""
    base = SpriteEnv()
    env = CurriculumFrameEnv(base, skip=6)
    env.reset()
    obs, reward, *_ = env.step(3)
    assert base.actions == [3] * 6
    assert reward == pytest.approx(6.0)
    assert obs.shape == (2, 84, 84) and obs.dtype == np.uint8


def test_termination_mid_skip_stops_the_repeat():
    base = SpriteEnv(end_at=3)
    env = CurriculumFrameEnv(base, skip=6)
    env.reset()
    _, reward, terminated, *_ = env.step(0)
    assert terminated and reward == pytest.approx(3.0)
    assert base.actions == [0] * 3


def test_trail_marks_motion_and_fades_toward_the_past():
    env = CurriculumFrameEnv(SpriteEnv(), skip=4, window_factor=2)
    env.reset()
    obs, *_ = env.step(0)
    trail = obs[1].astype(np.float32)
    assert trail.max() > 0, "moving sprite left no trail"
    # The newest diff outweighs the oldest: intensity at the sprite's
    # most recent transition exceeds intensity at its earliest one.
    row = trail[40:44, :].max(axis=0)   # sprite row after 84/210 resize
    lit = np.nonzero(row > 5)[0]
    assert len(lit) >= 2
    assert row[lit[-1]] > row[lit[0]], "trail does not fade toward the past"


def test_static_scene_leaves_the_trail_empty():
    class StaticEnv(SpriteEnv):
        def _frame(self):
            f = np.zeros((210, 160, 3), dtype=np.uint8)
            f[100:108, 40:48] = 255   # bright but motionless
            return f

    env = CurriculumFrameEnv(StaticEnv(), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    assert obs[1].max() == 0, "static scene produced phantom motion"
    assert obs[0].max() > 0, "current channel lost the static sprite"


def test_flickering_sprite_survives_the_current_channel():
    env = CurriculumFrameEnv(SpriteEnv(flicker=True), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    # Present in only one of the last two raw frames: the 2-frame max
    # keeps it in ch0 regardless of flicker phase.
    assert obs[0].max() == 255


def test_action_map_translates_and_shrinks_the_space():
    base = SpriteEnv()
    env = CurriculumFrameEnv(base, skip=2, action_map=PONG_MINIMAL_ACTIONS)
    assert env.action_space.n == 3
    env.reset()
    env.step(1)   # agent action 1 -> ALE action 2 (UP)
    env.step(2)   # agent action 2 -> ALE action 3 (DOWN)
    assert base.actions == [2, 2, 3, 3]


def test_decision_discount_aligns_across_skips():
    # 0.997 per raw frame: skip 4 gives EffZero's 0.98807; two stages of
    # the ladder describe the same horizon in raw-frame time.
    assert decision_discount(0.997, 4) == pytest.approx(0.997 ** 4)
    assert decision_discount(0.997, 30) == pytest.approx(0.997 ** 30)
    assert decision_discount(0.997, 30) == pytest.approx(
        decision_discount(decision_discount(0.997, 5), 6))
