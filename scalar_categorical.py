import torch
import numpy as np

def scalar_to_categorical_batch(scalars, support_size, num_bins, device):
    """
    Converts a batch of scalar values into a categorical distribution over a fixed support.
    
    Args:
        scalars (torch.Tensor): A tensor of scalar values, shape can be (B,) or (B, unroll_steps).
        support_size (float): The maximum value of the support (v_max). For non-negative support, v_min is 0.
        num_bins (int): Number of bins in the support.
        device: Torch device for creating the target distribution.
        
    Returns:
        torch.Tensor: A tensor of shape (*original_shape, num_bins) representing the target distributions.
    """
    v_min = 0.0
    v_max = support_size

    # Save the original shape (e.g., (B,) or (B, unroll_steps))
    original_shape = scalars.shape

    # Flatten the scalars into a 1D tensor.
    scalars = scalars.view(-1)
    scalars = scalars.clamp(v_min, v_max)
    
    # Compute the bin width.
    delta = (v_max - v_min) / (num_bins - 1)
    
    # Map scalars to continuous indices in the support.
    indices = (scalars - v_min) / delta  # Shape: (N,)
    lower_idx = torch.floor(indices).long()  # Shape: (N,)
    upper_idx = lower_idx + 1  # Shape: (N,)
    
    # Compute interpolation weights.
    upper_weight = indices - lower_idx.float()  # Shape: (N,)
    lower_weight = 1.0 - upper_weight             # Shape: (N,)
    
    # Ensure that the upper index does not exceed num_bins - 1.
    upper_idx = torch.where(upper_idx < num_bins, upper_idx, lower_idx)
    
    N = scalars.shape[0]
    target = torch.zeros(N, num_bins, device=device)
    
    # scatter_add_, not scatter_: at the top of the support the clamp above
    # forces upper_idx == lower_idx, and two plain scatters alias -- the second
    # overwrites the first with upper_weight, which is 0.0 there. That produced
    # an all-zero "distribution" for any scalar at or above support_size, and
    # cross-entropy against an all-zero target contributes exactly no gradient.
    # Since gameplay clips rewards with min(reward, 400) and the reward support
    # is also 400, every large reward landed in that hole.
    target.scatter_add_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_add_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))
    
    # Reshape back to the original shape with an added dimension for the bins.
    target = target.view(*original_shape, num_bins)
    return target

# The former in-module expected_distribution() helper and its test lived here.
# They mirrored the implementation line for line -- including the aliasing bug
# fixed above -- so they would have agreed with the broken behaviour had anyone
# run them. Coverage moved to tests/test_scalar_categorical.py, which checks
# independent properties (mass sums to 1, expected value round-trips) rather
# than comparing against a second copy of the same arithmetic.

