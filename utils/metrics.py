import torch
import numpy as np
from scipy.stats import spearmanr

def compute_uncertainty_metrics(preds, targets, uncertainty):
    """
    Computes rigorous uncertainty metrics including Calibration Error.
    """
    # 1. Reconstruction Error (MSE per sample)
    mse_per_sample = torch.mean((preds - targets)**2, dim=1)

    # 2. Mean Uncertainty per sample
    unc_per_sample = torch.mean(uncertainty, dim=1)

    # 3. Spearman Correlation
    mse_np = mse_per_sample.detach().cpu().numpy()
    unc_np = unc_per_sample.detach().cpu().numpy()

    if len(mse_np) > 1:
        corr, _ = spearmanr(mse_np, unc_np)
        corr = float(corr)
    else:
        corr = 0.0

    return {
        "mse": float(mse_per_sample.mean().item()),
        "mean_uncertainty": float(unc_per_sample.mean().item()),
        "spearman_corr": corr,
        "mse_per_sample": mse_per_sample,
        "unc_per_sample": unc_per_sample
    }

def compute_calibration_error(mse_per_sample, unc_per_sample, n_bins=10):
    """
    Computes Calibration Error for Regression.
    Ideally, Mean Uncertainty in a bin should equal Mean MSE in that bin.
    """
    mse = np.array(mse_per_sample)
    unc = np.array(unc_per_sample)

    # Sort by uncertainty
    indices = np.argsort(unc)
    mse = mse[indices]
    unc = unc[indices]

    bin_size = len(mse) // n_bins
    error_sum = 0.0
    total_samples = 0

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(mse)

        if start >= end: break

        bin_mse = np.mean(mse[start:end])
        bin_unc = np.mean(unc[start:end])

        # We assume variance corresponds to MSE directly (probabilistic interpretation)
        # If uncertainty is not calibrated to MSE scale, we measure the correlation of magnitudes
        # Simple Error: |Avg_Unc - Avg_MSE| weighted by bin size

        weight = (end - start) / len(mse)
        error_sum += weight * np.abs(bin_unc - bin_mse)

    return error_sum
