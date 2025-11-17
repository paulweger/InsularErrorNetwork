"""
Minimal iEEG Quality Checker
Flags channels that are irrelevant or detect inconsistent timestamps.
"""

import re
import numpy as np

class QualityChecker:
    """Minimal class to flag irrelevant EEG channels and check timestamps."""

    def __init__(self):
        self.results = {}

    def consistent_timestamps(self, timestamps: np.array, expected_fs: float, max_allowed_diff: float = 1) -> np.array:
        """
        Check for inconsistent inter-sample intervals.
        Returns indices of invalid timesteps.
        """
        timesteps = np.diff(timestamps)
        diff_timesteps = np.diff(timesteps)
        invalid_idx = np.where(np.abs(diff_timesteps) > max_allowed_diff / 1000)[0]
        self.results['consistent_timestamps'] = {
            'invalid_timesteps': invalid_idx,
            'mean_fs': 1 / np.mean(timesteps)
        }
        return invalid_idx

    def get_disconnected_channels(self, eeg: np.array, channel_names: list = None) -> np.array:
        """
        Flag channels that are likely disconnected:
        - by name if provided
        - or by abnormal power spectrum
        """
        if channel_names is not None:
            pattern = r'(?<![A-Za-z])[Ee][l\d]'
            flagged = [i for i, name in enumerate(channel_names) if re.search(pattern, name)]
        else:
            ps_log_mean = np.log(np.abs(np.fft.rfft(eeg.T - eeg.T.mean(axis=0)))**2).T.mean(axis=0)
            flagged = np.where(ps_log_mean > ps_log_mean.mean())[0]

        self.results['disconnected_channels'] = flagged
        return np.array(flagged)

    def get_ekg_channel(self, eeg: np.array, channel_names: list = None) -> np.array:
        """
        Flag EKG channel either by name or by largest amplitude peaks.
        """
        if channel_names is not None:
            flagged = [i for i, ch in enumerate(channel_names) if 'EKG' in ch]
        else:
            n_max = 10000
            sorted_vals = np.sort(np.abs(eeg), axis=0)
            ch_max = np.argmax(sorted_vals[-n_max:, :], axis=1)
            unique, counts = np.unique(ch_max, return_counts=True)
            flagged = [unique[np.argmax(counts)]]
        self.results['ekg_channels'] = flagged
        return np.array(flagged)

    def get_marker_channels(self, eeg: np.array, channel_names: list = None) -> np.array:
        """
        Flag marker channels by name or binary toggling.
        """
        if channel_names is not None:
            flagged = [i for i, ch in enumerate(channel_names) if 'MKR' in ch]
        else:
            # Assume marker channels toggle between 2 values
            flagged = np.where([len(np.unique(eeg[:, ch])) <= 2 for ch in range(eeg.shape[1])])[0]

        self.results['marker_channels'] = flagged
        return np.array(flagged)
