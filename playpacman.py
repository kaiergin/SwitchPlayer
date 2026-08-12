import gymnasium as gym
import pygame
import pickle
import numpy as np
import cv2
import muzero
from EpisodicLifeEnv import EpisodicLifeEnv

# Create the Gym environment for MsPacman
env = gym.make("MsPacman-v4")
env = EpisodicLifeEnv(env)
IMAGE_RESHAPE = (84,84)

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

'''
# Initialize the replay buffer (a list of game trajectories)
try:
    with open("replay_buffer.pkl", "rb") as pkfile:
        replay_buffer = pickle.load(pkfile)
except:
    replay_buffer = []
'''
replay_buffer = []

# Initialize Pygame and create a window sized to the environment frame
pygame.init()
action_space_size = env.action_space.n
# Get an initial frame from the environment to determine size.
frame, _ = env.reset()
height, width, _ = frame.shape
scale = 3
screen = pygame.display.set_mode((width * scale, height * scale))
pygame.display.set_caption("MsPacman")

# Set up a clock to control the frame rate
clock = pygame.time.Clock()
FPS = 30

# Map Pygame arrow keys to environment actions.
# Note: You might need to adjust these mappings depending on your environment.
# Here we assume:
#   Left arrow  -> action 3
#   Right arrow -> action 2
#   Up arrow    -> action 1
#   Down arrow  -> action 4
key_to_action = {
    pygame.K_LEFT: 3,
    pygame.K_RIGHT: 2,
    pygame.K_UP: 1,
    pygame.K_DOWN: 4
}

def get_action_from_keys():
    """Return the action corresponding to the currently pressed arrow key.
    Defaults to 0 (no-op) if no arrow key is pressed."""
    keys = pygame.key.get_pressed()
    for key, action in key_to_action.items():
        if keys[key]:
            return action
    return 0  # no-op action if no key is pressed

running = True

try:
    while running:
        # Start a new game episode
        current_game_trajectory = []
        observation, _ = env.reset()
        preprocessed_obs = muzero.preprocess_frame(observation, resize_shape=IMAGE_RESHAPE)
        preprocessed_obs = np.transpose(preprocessed_obs, (2, 1, 0))
        done = False
        max_reward = 0.0
        prev_reward = 0.0
        frame_stack = np.repeat(preprocessed_obs[:, :, :], 12, axis=0)
        while not done:
            # Handle Pygame events (including quit events)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

            preprocessed_obs = muzero.preprocess_frame(observation, resize_shape=IMAGE_RESHAPE)
            preprocessed_obs = np.transpose(preprocessed_obs, (2, 1, 0))
            frame_stack = np.concatenate([frame_stack[3:, :, :], preprocessed_obs[:, :, :]], axis=0)

            # Get the action from the arrow keys (default is 0 if none are pressed)
            action = get_action_from_keys()
            # Step the environment using the chosen action
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Save the (frame, action) pair for the current game trajectory
            current_game_trajectory.append((preprocessed_obs, action, prev_reward))

            prev_reward = reward
            if reward > max_reward:
                max_reward = reward
            
            # Convert the observation (a NumPy array) into a Pygame surface.
            surface = pygame.surfarray.make_surface(preprocessed_obs)
            # Scale the surface to make it bigger.
            surface = pygame.transform.scale(surface, (width * scale, height * scale))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            clock.tick(FPS)

        print(max_reward)
        discounted_max_backup = 0.0
        discounted_backup = 0.0
        for x in reversed(current_game_trajectory):
            discounted_backup = x[2] + 0.997*discounted_backup
            if discounted_backup > discounted_max_backup:
                discounted_max_backup = discounted_backup
        print(discounted_max_backup)
        # End of episode: add the game trajectory to the replay buffer
        replay_buffer.append(current_game_trajectory)
        
        '''
        # Save (overwrite) the entire replay buffer to a pickle file.
        with open("replay_buffer.pkl", "wb") as f:
            pickle.dump(replay_buffer, f)
        '''

except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting gracefully...")

finally:
    pygame.quit()
    env.close()
