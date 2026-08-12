import gymnasium as gym


class EpisodicLifeEnv(gym.Wrapper):
    """Ends the training episode when a life is lost, but only resets the
    underlying environment on a real game over.

    This is the standard Atari treatment: the agent gets a terminal signal for
    dying without the rest of the game being thrown away.
    """

    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        # Only reset environment when the game is truly over.
        if self.was_real_done:
            observation, info = self.env.reset(**kwargs)
        else:
            # No reset when a life is lost, just propagate the current state.
            observation, _, terminated, truncated, info = self.env.step(0)
            # That no-op can itself end the game -- it may have consumed the
            # last life, or hit the time limit. Previously the result was
            # discarded, leaving the env stuck in a terminal state.
            if terminated or truncated:
                self.was_real_done = True
                observation, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Truncation (the TimeLimit wrapper) ends the episode just as much as
        # termination does. The previous unpacking was positionally wrong --
        # it bound truncated to `info` and the real info dict to a spare -- so
        # time-limit expiry was never treated as an end of episode.
        self.was_real_done = terminated or truncated
        # Check current life count to see if a life was lost.
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Mark as terminated if a life is lost, but not if game over.
            terminated = True
        self.lives = lives
        return observation, reward, terminated, truncated, info
