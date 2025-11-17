import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from os.path import getctime
import json

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
    marker: np.array
    marker_ts: np.array
    channels: np.array
    trial_nums: np.array
    trial_names: np.array
    labels: dict
    raw_eeg: np.array

def _get_created_date(file, dt_format='%Y%m%d%H%M%S'):
    """Return formatted creation date of a file."""
    return datetime.fromtimestamp(getctime(file)).strftime(dt_format)


def _get_experiment_data(result):
    "Extract data and markers from the loaded xdf result."
    
    # EEG data 
    dataStreamName = 'Micromed'
    eeg = result[dataStreamName]['data']
    eeg_ts = result[dataStreamName]['ts']
    
    # Markers
    markerStreamName = 'GameMarkers'
    marker = result[markerStreamName]['data']
    marker_ts = result[markerStreamName]['ts']
    
    return eeg, eeg_ts, marker, marker_ts


def extract_markers(marker, marker_ts=None):
    """Parse JSON markers into dict of NumPy arrays (including timestamps)."""
    flat = [m[0] if isinstance(m, (list, np.ndarray)) else m for m in marker]
    parsed = [json.loads(m) for m in flat]
    keys = parsed[0].keys()
    out = {k: np.array([d.get(k, np.nan) for d in parsed]) for k in keys}
    if marker_ts is not None:
        out["timestamp"] = np.array(marker_ts)
    return out

def align_markers(marker_dict, data_ts):
    """
    Forward-fill markers to match data timestamps.
    
    marker_dict: dict of arrays with 'timestamp'
    data_ts: array of continuous timestamps
    """
    t_markers = marker_dict["timestamp"]
    idxs = np.searchsorted(t_markers, data_ts, side='right') - 1
    idxs[idxs < 0] = 0
    return {k: v[idxs] for k, v in marker_dict.items()}


def load_seeg(file):
    ''' Loads xdf file and returns a dict with all necessary information '''

    # Load data from file
    file = Path(file)
    logging.info(f'Loading file: {file}')
    result, _ = xdf_to_dict(file)
    eeg, eeg_ts, marker, marker_ts = _get_experiment_data(result)

    # Determine subject info
    multiple_measurements = 'kh' not in file.parts[-2]
    subject = file.parts[-2] if not multiple_measurements else file.parts[-3]
    exp_date = file.parts[-2] if multiple_measurements else _get_created_date(file)

    # Modify markers so they fit in array
    labels = extract_markers(marker, marker_ts)
    labels = align_markers(labels, eeg_ts)

    # Save data and markers in dict
    seeg = {
        'subject': subject,
        'experiment_type': file.parts[-1].split('.xdf')[0],
        'experiment_date': exp_date,
        'channel_names': result['Micromed']['channel_names'],
        'eeg': eeg.astype(np.float64),
        'eeg_ts': eeg_ts,
        'marker': marker,
        'marker_ts': marker_ts,
        'labels': labels,
        'trial_labels': None,
        'trial_numbers': None,
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
    
    seeg = load_seeg(paths)
    
    s = Session(
        ppt_id =   None,
        kh_id =    seeg['subject'],
        datetime = seeg['experiment_date'],
        eeg =      seeg['eeg'],
        ts =       seeg['eeg_ts'],
        fs =       seeg['fs'],
        channels = seeg['channel_names'],
        trial_nums = seeg['trial_numbers'],
        trial_names = seeg['trial_labels'],
        eeg_ts = seeg['eeg_ts'],
        marker_ts = seeg['marker_ts'],
        marker = seeg['marker'],
        labels = seeg['labels'],
        raw_eeg= seeg['eeg'].copy()
    )
    
    return s