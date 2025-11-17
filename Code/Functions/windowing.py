import numpy as np
from collections import Counter

def mode(arr: np.ndarray, axis=1) -> np.ndarray:
    """Compute mode along a given axis."""
    mode_1d = lambda x: Counter(x.tolist()).most_common(1)[0][0]
    return np.apply_along_axis(mode_1d, axis, arr)


def windowing(arr: np.ndarray, samples_per_window: int, step: int) -> np.ndarray:
    """Window a 2D array using given window size and step (in samples)."""
    arr = arr[:, None] if arr.ndim == 1 else arr
    n_samples = arr.shape[0]

    n_windows = (n_samples - samples_per_window) // step + 1
    if n_windows <= 0:
        raise ValueError(f"Array too short ({n_samples}) for window size {samples_per_window}")

    windows = np.stack([arr[i*step : i*step + samples_per_window] for i in range(n_windows)])
    return windows
