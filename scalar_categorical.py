"""Encoding scalars (rewards, values, value prefixes) as categorical distributions.

This is the single home for the encode/decode contract. Two pieces:

1. An invertible transform h() that compresses large magnitudes, so a fixed
   number of bins can cover a wide range while still resolving small values.
2. A two-hot projection onto a symmetric support in transformed space.

Both directions must stay paired: targets go through h() before projection, and
predictions come back through h_inverse() after decoding, so everything outside
this module works in raw units.
"""
import torch
import torch.nn.functional as F

# The linear term in h(). Keeps the transform invertible and stops the gradient
# vanishing far from the origin.
H_EPSILON = 0.001


def h_transform(x, eps=H_EPSILON):
    """MuZero's invertible value transform (Pohlen et al. 2018, MuZero appendix F).

        h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x

    Square-root compression, so the useful range grows quadratically with the
    support while resolution near zero stays fine. h(2000) is about 46, which is
    why 51 bins can cover a full unroll of clipped rewards and still tell 1
    apart from 5 -- the resolution failure measured in benchmarks/RESULTS.md,
    where a linear support left 40-wide bins and every realistic target landed
    in bin 0.
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def h_inverse(x, eps=H_EPSILON):
    """Exact inverse of h_transform.

    Solving eps*u^2 + u - (1 + eps + |h|) = 0 for u = sqrt(|x| + 1) gives

        |x| = ((sqrt(1 + 4*eps*(|h| + 1 + eps)) - 1) / (2*eps))^2 - 1
    """
    root = torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps))
    return torch.sign(x) * (((root - 1.0) / (2.0 * eps)) ** 2 - 1.0)


def create_support(support_size, num_bins, device):
    """A symmetric support from -support_size to +support_size, in h-space.

    Symmetric because the old non-negative support made negative values
    literally unrepresentable -- any negative reward or value clamped to zero,
    so a head could express "worth little" but never "bad".
    """
    return torch.linspace(-support_size, support_size, num_bins, device=device)


def scalar_from_logits(logits, support, invert=True):
    """Decode categorical logits to a scalar, back in raw units.

    Takes the expectation over the support, then undoes h(). Pass invert=False
    to stay in transformed space (only useful for diagnostics).
    """
    probs = F.softmax(logits, dim=1)
    expected = torch.sum(probs * support.unsqueeze(0).expand_as(probs), dim=1, keepdim=True)
    return h_inverse(expected) if invert else expected


def scalar_to_categorical_batch(scalars, support_size, num_bins, device, transform=True):
    """Project raw scalars onto a two-hot categorical target.

    Args:
        scalars: raw values, shape (B,) or (B, T).
        support_size: half-width of the support, in TRANSFORMED space.
        num_bins: number of bins across [-support_size, +support_size].
        device: torch device for the target.
        transform: apply h() first. Leave True unless the input is already
            transformed.

    Returns:
        (*scalars.shape, num_bins), each row summing to 1.
    """
    original_shape = scalars.shape
    scalars = scalars.reshape(-1)
    if transform:
        scalars = h_transform(scalars)

    v_min, v_max = -float(support_size), float(support_size)
    scalars = scalars.clamp(v_min, v_max)

    delta = (v_max - v_min) / (num_bins - 1)
    indices = (scalars - v_min) / delta
    lower_idx = torch.floor(indices).long()
    upper_idx = lower_idx + 1

    upper_weight = indices - lower_idx.float()
    lower_weight = 1.0 - upper_weight

    # Keep the upper index in range. At the very top of the support this makes
    # it equal the lower index, which is why the scatters below must accumulate.
    upper_idx = torch.where(upper_idx < num_bins, upper_idx, lower_idx)

    target = torch.zeros(scalars.shape[0], num_bins, device=device)

    # scatter_add_, not scatter_: where the clamp forces upper_idx == lower_idx,
    # two plain scatters alias and the second overwrites the first's weight of
    # 1.0 with upper_weight, which is 0.0 there. That produced an all-zero
    # "distribution" for anything at the edge of the support, and cross-entropy
    # against all zeros contributes exactly no gradient.
    target.scatter_add_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_add_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    return target.view(*original_shape, num_bins)


def support_for(raw_ceiling):
    """Half-width of support needed, in h-space, to cover +/- raw_ceiling."""
    return float(h_transform(torch.tensor(float(raw_ceiling))))
