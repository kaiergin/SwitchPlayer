import gymnasium as gym
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# See muzero.py -- gymnasium >= 1.0 needs the ale-py plugin registered explicitly.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass

# -----------------------------------------------------------------
# Example RepresentationNetwork (dummy). Replace with your own.
# -----------------------------------------------------------------
import torch.nn as nn
class RepresentationNetwork(nn.Module):
    def __init__(self, input_channels=1, latent_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, latent_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x expected shape: (batch_size, input_channels, 84, 84)
        return self.relu(self.conv(x))

# -----------------------------------------------------------------
# Preprocessing function
# -----------------------------------------------------------------
def preprocess_frame(frame, resize_shape=(96, 96)):
    """
    Convert RGB to grayscale, then resize the image to (84, 84).
    """
    # Convert from RGB to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(gray, resize_shape, interpolation=cv2.INTER_AREA)
    return resized

# -----------------------------------------------------------------
# Main script
# -----------------------------------------------------------------
def main():
    # Create your environment. If you have gymnasium or older Gym, adjust accordingly.
    env = gym.make("MsPacman-v4", render_mode="rgb_array")

    # Reset and get the initial observation
    obs, info = env.reset()
    # obs is typically a NumPy array of shape (height, width, 3) if render_mode="rgb_array"

    # Preprocess the frame
    preprocessed = preprocess_frame(obs)

    # Display the shape before you feed it into the network
    print("Raw frame shape:", obs.shape)
    print("Preprocessed frame shape:", preprocessed.shape)

    # Display the preprocessed frame using matplotlib
    plt.imshow(preprocessed, cmap='gray')
    plt.title("Preprocessed Frame (Grayscale 84×84)")
    plt.show()

    # Turn preprocessed frame into a PyTorch tensor and pass it into your representation network
    # Representation network expects shape (batch, channels, height, width).
    # Because we used grayscale, channels=1:
    x_tensor = torch.tensor(preprocessed, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    # Create the representation network (replace with your own instance if needed).
    representation_network = RepresentationNetwork(input_channels=1)
    latent = representation_network(x_tensor)
    print("Latent shape:", latent.shape)

    # Clean up environment if you want
    env.close()

if __name__ == "__main__":
    main()
