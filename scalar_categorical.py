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
    
    # Scatter the lower and upper weights into the target tensor.
    target.scatter_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))
    
    # Reshape back to the original shape with an added dimension for the bins.
    target = target.view(*original_shape, num_bins)
    return target

# --- Helper function for unit testing ---
def expected_distribution(scalar, support_size, num_bins):
    """
    Computes the expected categorical distribution for a given scalar.
    Assumes v_min=0, v_max=support_size and uniform spacing.
    """
    v_min = 0.0
    v_max = support_size
    delta = (v_max - v_min) / (num_bins - 1)
    index = (scalar - v_min) / delta
    lower_idx = int(np.floor(index))
    upper_idx = lower_idx + 1
    upper_weight = index - lower_idx
    lower_weight = 1.0 - upper_weight
    if upper_idx >= num_bins:
        upper_idx = lower_idx
    target = np.zeros(num_bins, dtype=np.float32)
    target[lower_idx] = lower_weight
    target[upper_idx] = upper_weight
    return target

# --- Unit Test ---
def test_scalar_to_categorical_batch_correct_positions():
    device = torch.device("cpu")
    support_size = 500
    num_bins = 51
    # Create a test tensor with shape (B, unroll_steps), for example (2, 3).
    # We choose specific scalars to test various cases:
    # Case 1: scalar exactly 0 → should put full weight on bin 0.
    # Case 2: scalar exactly equals a bin center (e.g., 10 if delta=10) → full weight on that bin.
    # Case 3: scalar halfway between bins (e.g., 5) → half weight on bin 0 and half on bin 1.
    # Case 4: scalar equals support_size → full weight on the last bin.
    scalars = torch.tensor([[0.0, 5.0, 10.0],
                            [500.0, 250.0, 5.0]], device=device)  # shape (2, 3)
    
    # Compute the target distributions using our function.
    target_dists = scalar_to_categorical_batch(scalars, support_size, num_bins, device)
    
    # Now compute the expected distributions for each scalar.
    B, unroll_steps = scalars.shape
    expected = []
    for s in scalars.view(-1).cpu().numpy():
        expected.append(expected_distribution(s, support_size, num_bins))
    expected = np.array(expected).reshape(B, unroll_steps, num_bins)
    expected_tensor = torch.tensor(expected, device=device, dtype=torch.float32)
    
    # Check that each distribution matches the expected output.
    assert torch.allclose(target_dists, expected_tensor, atol=1e-5), (
        f"Target distributions do not match expected.\nGot: {target_dists}\nExpected: {expected_tensor}"
    )
    print("Test passed: All outputs are in the correct positions and the reshaping is correct.")

