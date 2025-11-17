""" 
Functions for brain plotting
"""
import numpy as np

def filter_to_roi(errorChannels, rois):
    """
    Filter error channels to only include those in specified ROIs.
    Return: Filtered channels to highlight, and maximum absolute t-value across all participants.
    """
    out = {}
    all_t = []

    for ppt, d in errorChannels.items():
        m = np.isin(d['roi'], rois)
        out[ppt] = {
            'channels': np.array(d['channels'])[m],
            'tvalues':  np.array(d['tvalues'])[m],
            'roi':      np.array(d['roi'])[m]
        }
        all_t.extend(np.abs(out[ppt]['tvalues']))

    max_val = max(all_t) if all_t else None
    min_val = min(all_t) if all_t else None
    return out, min_val, max_val


def get_min_max(motorChannels):
    """Get global min and max absolute t-values across all participants."""
    all_t = []
    for ppt, d in motorChannels.items():
        all_t.extend(np.abs(d['tvalues']))
    return min(all_t), max(all_t)


def tvals_to_sizes(tvals, tmin, tmax, min_size=2, max_size=6):
    """Log-scale t-values (using global tmin/tmax) to marker sizes."""
    v = np.log(np.abs(tvals) - tmin + 1)
    hi = np.log(tmax - tmin + 1)
    if hi == 0: 
        return np.full_like(v, min_size, float)
    v = v / hi
    return min_size + v * (max_size - min_size)


def tvals_to_color(tvals, tmin, tmax):
    """Log-scale t-values to 0–1 range for colormap"""
    v = np.log(np.abs(tvals) - tmin + 1)
    hi = np.log(tmax - tmin + 1)
    if hi == 0:
        return np.zeros_like(v, float)
    return np.clip(v / hi, 0, 1)
