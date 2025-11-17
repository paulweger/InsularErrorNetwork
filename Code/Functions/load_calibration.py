import bisect
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from os.path import getctime

# Local imports
from Functions.load_xdf import xdf_to_dict


@dataclass
class Session:
    """Stores EEG session information."""
    ppt_id: int
    kh_id: str
    datetime: str
    eeg: np.array
    ts: np.array
    fs: int
    eeg_ts: np.array
    marker_ts: np.array
    channels: np.array
    trial_nums: np.array
    trial_names: np.array
    raw_eeg: np.array


def _get_created_date(file, dt_format='%Y%m%d%H%M%S'):
    """Return formatted creation date of a file."""
    return datetime.fromtimestamp(getctime(file)).strftime(dt_format)


def _locate_pos(available_tss, target_ts):
    """Find closest index in a list of timestamps."""
    pos = bisect.bisect_right(available_tss, target_ts)
    if pos == 0:
        return 0
    if pos == len(available_tss):
        return len(available_tss) - 1
    # Choose the closer of two neighboring timestamps
    if abs(available_tss[pos] - target_ts) < abs(available_tss[pos - 1] - target_ts):
        return pos
    else:
        return pos - 1


def _get_trials_info_newMarkers(eeg, eeg_ts, markers, marker_ts):
    """Assign numeric labels (0=Rest, 1=Move) and trial numbers per sample."""
    # Find start and end markers
    trial_start_mask = [m[0].split(';')[0] == 'start' for m in markers]
    trial_end_mask = [m[0].split(';')[0] == 'end' for m in markers]

    # Get indices in EEG for trial starts and ends
    trial_idc_start = np.array([_locate_pos(eeg_ts, t) for t in marker_ts[trial_start_mask]])
    trial_idc_end = np.array([_locate_pos(eeg_ts, t) for t in marker_ts[trial_end_mask]])

    # Map trial types to numeric labels
    trial_labels = [1 if 'Move' in m[0] else 0 for m in markers if m[0].startswith('start')]

    # Initialize trial arrays
    trial_seq = [0] * eeg.shape[0]
    trial_nums = [0] * eeg.shape[0]

    # Assign labels and trial numbers from start to next start (or end)
    for i, idx_start in enumerate(trial_idc_start):
        idx_end = trial_idc_start[i + 1] if i + 1 < len(trial_idc_start) else len(trial_seq)
        trial_seq[idx_start:idx_end] = [trial_labels[i]] * (idx_end - idx_start)
        trial_nums[idx_start:idx_end] = [i] * (idx_end - idx_start)

    # Fill samples before first start marker
    if trial_idc_start.size > 0:
        trial_seq[:trial_idc_start[0]] = [trial_labels[0]] * trial_idc_start[0]

    return np.array(trial_seq), np.array(trial_nums)


def _get_experiment_data(result):
    """Extract EEG and marker data for the experiment period."""
    marker_idx_exp_start = result['GraspMarkerStream']['data'].index(['experimentStarted'])
    marker_idx_exp_end = result['GraspMarkerStream']['data'].index(['experimentEnded'])

    marker = result['GraspMarkerStream']['data'][marker_idx_exp_start:marker_idx_exp_end]
    marker_ts = result['GraspMarkerStream']['ts'][marker_idx_exp_start:marker_idx_exp_end]

    eeg_idx_exp_start = _locate_pos(result['Micromed']['ts'], result['GraspMarkerStream']['ts'][marker_idx_exp_start])
    eeg_idx_exp_end = _locate_pos(result['Micromed']['ts'], result['GraspMarkerStream']['ts'][marker_idx_exp_end])

    eeg = result['Micromed']['data'][eeg_idx_exp_start:eeg_idx_exp_end, :]
    eeg_ts = result['Micromed']['ts'][eeg_idx_exp_start:eeg_idx_exp_end]

    return eeg, eeg_ts, marker, marker_ts


def load_grasp_seeg(file):
    """Load XDF file and return dict with EEG, timestamps, trial labels, and metadata."""
    file = Path(file)
    logging.info(f'Loading file: {file}')

    result, _ = xdf_to_dict(file)

    # Extract experiment-specific EEG and markers
    eeg, eeg_ts, markers, markers_ts = _get_experiment_data(result)

    # Get numeric trial labels and trial numbers
    trials, trial_nums = _get_trials_info_newMarkers(eeg, eeg_ts, markers, markers_ts)
    label_map = {0: 'Rest', 1: 'Move'}
    trials = np.vectorize(label_map.get)(trials)

    # Determine subject info
    multiple_measurements = 'kh' not in file.parts[-2]
    subject = file.parts[-2] if not multiple_measurements else file.parts[-3]
    exp_date = file.parts[-2] if multiple_measurements else _get_created_date(file)

    seeg = {
        'subject': subject,
        'experiment_type': file.parts[-1].split('.xdf')[0],
        'experiment_date': exp_date,
        'channel_names': result['Micromed']['channel_names'],
        'eeg': eeg.astype(np.float64),
        'eeg_ts': eeg_ts,
        'marker_ts': markers_ts,
        'trial_labels': trials,
        'trial_numbers': trial_nums,
        'fs': result['Micromed']['fs'],
        'dtype': result['Micromed']['data_type'],
        'first_ts': result['Micromed']['first_ts'],
        'last_ts': result['Micromed']['last_ts'],
        'total_stream_time': result['Micromed']['total_stream_time'],
        'samplecount': result['Micromed']['sample_count']
    }

    return seeg


def load(paths):
    """Load EEG session and return a Session object."""
    seeg = load_grasp_seeg(paths)

    s = Session(
        ppt_id=None,
        kh_id=seeg['subject'],
        datetime=seeg['experiment_date'],
        eeg=seeg['eeg'],
        ts=seeg['eeg_ts'],
        fs=seeg['fs'],
        channels=seeg['channel_names'],
        trial_nums=seeg['trial_numbers'],
        trial_names=seeg['trial_labels'],
        eeg_ts=seeg['eeg_ts'],
        marker_ts=seeg['marker_ts'],
        raw_eeg=seeg['eeg'].copy()
    )
    return s
