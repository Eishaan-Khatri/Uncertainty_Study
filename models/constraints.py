import torch

def total_variation_loss(x):
    """
    Computes the Total Variation (TV) loss for 1D signals.
    Encourages smoothness by penalizing differences between adjacent time steps.

    Args:
        x: (Batch, Channel, Length) or (Batch, Length)

    Returns:
        Scalar tensor representing the mean TV loss.
    """
    # Ensure shape is (Batch, Channel, Length)
    if x.dim() == 2:
        x = x.unsqueeze(1)

    # Calculate difference between t and t+1
    # x[..., 1:] is t+1
    # x[..., :-1] is t
    diff = torch.abs(x[..., 1:] - x[..., :-1])

    # Mean over batch and length ensures scale independence
    return torch.mean(diff)
