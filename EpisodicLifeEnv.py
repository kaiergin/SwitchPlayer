import gymnasium as gym
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        # Only reset environment when the game is truly over.
        if self.was_real_done:
            observation, _ = self.env.reset(**kwargs)
        else:
            # No reset when life is lost, but just propagate the current state.
            observation, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return observation, _

    def step(self, action):
        observation, reward, done, info, x = self.env.step(action)
        self.was_real_done = done
        # Check current life count to see if a life was lost.
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Mark as done if a life is lost, but not if game over.
            done = True
        self.lives = lives
        return observation, reward, done, info, x