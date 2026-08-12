"""EpisodicLifeEnv against a scripted fake ALE."""
import gymnasium as gym
import numpy as np
import pytest

from EpisodicLifeEnv import EpisodicLifeEnv

OBS = np.zeros((2, 2, 3), dtype=np.uint8)


class FakeALE:
    def __init__(self, lives=3):
        self._lives = lives

    def lives(self):
        return self._lives


class FakeAtari(gym.Env):
    """Steps through a script of (reward, terminated, truncated, lives_after)."""

    def __init__(self, script=None):
        self.observation_space = gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(5)
        self.ale = FakeALE()
        self.script = list(script or [])
        self.reset_calls = 0
        self.step_calls = 0
        self.actions = []

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.ale._lives = 3
        return OBS, {"source": "reset"}

    def step(self, action):
        self.actions.append(action)
        self.step_calls += 1
        if self.script:
            reward, term, trunc, lives = self.script.pop(0)
            self.ale._lives = lives
        else:
            reward, term, trunc = 0.0, False, False
        return OBS, reward, term, trunc, {"source": "step"}


def wrap(script=None):
    inner = FakeAtari(script)
    env = EpisodicLifeEnv(inner)
    env.reset()
    return env, inner


# --------------------------------------------------------------------------
# step
# --------------------------------------------------------------------------
def test_losing_a_life_terminates_the_training_episode():
    env, _ = wrap([(0.0, False, False, 2)])
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    assert not env.was_real_done, "life loss is not a real game over"


def test_surviving_step_does_not_terminate():
    env, _ = wrap([(10.0, False, False, 3)])
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 10.0 and not terminated and not truncated


def test_real_game_over_sets_was_real_done():
    env, _ = wrap([(0.0, True, False, 0)])
    _, _, terminated, _, _ = env.step(0)
    assert terminated and env.was_real_done


def test_losing_the_last_life_is_a_real_game_over():
    """lives > 0 guards the life-loss branch, so dropping to zero must fall
    through to the underlying terminated flag rather than being masked."""
    env, _ = wrap([(0.0, True, False, 0)])
    env.step(0)
    assert env.was_real_done


def test_truncation_is_propagated_and_counts_as_done():
    """Regression: step unpacked the 5-tuple as (obs, reward, done, info, x), so
    truncated was bound to `info` and the real info dict to a spare. Time-limit
    expiry was never treated as the end of an episode."""
    env, _ = wrap([(0.0, False, True, 3)])
    _, _, terminated, truncated, info = env.step(0)
    assert truncated is True
    assert env.was_real_done, "truncation ends the episode too"
    assert isinstance(info, dict) and info["source"] == "step"


def test_info_is_a_dict_not_the_truncated_flag():
    env, _ = wrap([(0.0, False, False, 3)])
    *_, info = env.step(0)
    assert isinstance(info, dict)


def test_reset_returns_a_dict_info():
    env, _ = wrap()
    _, info = env.reset()
    assert isinstance(info, dict)


# --------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------
def test_reset_after_life_loss_does_not_restart_the_game():
    env, inner = wrap([(0.0, False, False, 2)])
    env.step(0)
    resets_before = inner.reset_calls
    env.reset()
    assert inner.reset_calls == resets_before, "should step through, not reset"
    assert inner.actions[-1] == 0, "continues with a no-op"


def test_reset_after_real_game_over_restarts():
    env, inner = wrap([(0.0, True, False, 0)])
    env.step(0)
    resets_before = inner.reset_calls
    env.reset()
    assert inner.reset_calls == resets_before + 1


def test_noop_step_that_ends_the_game_forces_a_reset():
    """Regression: reset discarded the result of its no-op step, so if that step
    consumed the last life or hit the time limit, the wrapper handed back a
    terminal observation and the env stayed stuck there."""
    env, inner = wrap([(0.0, False, False, 2),   # lose a life
                       (0.0, True, False, 0)])   # the no-op ends the game
    env.step(0)
    resets_before = inner.reset_calls
    env.reset()
    assert inner.reset_calls == resets_before + 1
    assert env.was_real_done


def test_noop_step_truncation_also_forces_a_reset():
    env, inner = wrap([(0.0, False, False, 2),
                       (0.0, False, True, 2)])
    env.step(0)
    resets_before = inner.reset_calls
    env.reset()
    assert inner.reset_calls == resets_before + 1


def test_life_counter_tracks_the_underlying_env():
    env, _ = wrap([(0.0, False, False, 1)])
    env.step(0)
    assert env.lives == 1


def test_life_counter_survives_a_mid_game_reset():
    """Resetting after a life loss continues the same game, so the life count
    must NOT jump back to full -- that would hide the remaining lives from the
    wrapper and break subsequent life-loss detection."""
    env, _ = wrap([(0.0, False, False, 1)])
    env.step(0)
    env.reset()
    assert env.lives == 1


def test_life_counter_is_refreshed_after_a_real_restart():
    env, _ = wrap([(0.0, True, False, 0)])
    env.step(0)
    env.reset()
    assert env.lives == 3
