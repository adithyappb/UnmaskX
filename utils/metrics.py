import numpy as np

def mse(pred, target):
    """Mean Squared Error."""
    return np.mean((pred - target) ** 2)

def psnr(pred, target):
    """Peak Signal-to-Noise Ratio."""
    mse_val = mse(pred, target)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

def evaluate_inpainting(pred, target):
    """Evaluates inpainting quality using MSE and PSNR."""
    return {
        'MSE': mse(pred, target),
        'PSNR': psnr(pred, target)
    }
