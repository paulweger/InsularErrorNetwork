import numpy as np
import scipy.signal
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

# Local import
from Functions.check_quality import QualityChecker


def hilbert_transform(data, smoothing=True, fs=1024, kernel_ms=250):
    """
    Compute Hilbert transform to get the analytic signal (amplitude envelope).
    Optionally smooth with a Gaussian filter and normalize each channel (z-score).
    """
    # Compute analytic signal using next optimal FFT length
    n_fourier = next_fast_len(data.shape[0])
    y = np.abs(scipy.signal.hilbert(data, n_fourier, axis=0)[:data.shape[0]])

    # Smooth with Gaussian kernel if requested
    if smoothing:
        sigma = (kernel_ms / 1000) * fs / 2.355  # convert ms to sigma
        y = gaussian_filter1d(y, sigma=sigma, axis=0, mode='reflect')

    # Z-score normalization per channel
    return (y - np.mean(y, axis=0)) / np.std(y, axis=0)


def frequency_filters(session, filters, order=2):
    """
    Apply bandpass filter for each specified frequency band.
    Returns session with filtered EEG (channels stacked per band).
    """
    eeg = session.eeg
    fs = session.fs
    nyquist = 0.5 * fs
    filtered_data = []

    for band, (lowcut, highcut) in filters.items():
        # Design Butterworth bandpass filter
        b, a = scipy.signal.butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
        # Apply filter (axis=1 because we transpose for filtfilt)
        filtered_data.append(scipy.signal.filtfilt(b, a, eeg.T, axis=1))

    # Stack filtered bands and transpose back
    session.eeg = np.vstack(filtered_data).T
    return session

def bandpass_filter(data, lowcut=4, highcut=200, order=4):
    """
    Bandpass filter raw signal into desired range for frequency analysis
    data in shape (time x channels)
    """
    fs = 1024
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data, axis=0)  # Zero-phase filtering
    return y


def preprocess(session, reref=True, norm=True):
    """
    Rereference EEG channels per shaft and normalize (z-score).
    """
    eeg = session.eeg
    chan_names = session.channels

    # Rereference: subtract mean per shaft
    if reref:
        shafts = defaultdict(list)
        for i, name in enumerate(chan_names):
            shaft = ''.join(c for c in name if not c.isdigit())  # e.g., "A" from "A1"
            shafts[shaft].append(i)

        reref_eeg = np.copy(eeg)
        for indices in shafts.values():
            indices = np.array(indices)
            if len(indices) > 0:
                reref_eeg[:, indices] -= np.mean(eeg[:, indices], axis=1, keepdims=True)
    else:
        reref_eeg = eeg

    # Normalize each channel (z-score)
    if norm:
        reref_eeg = (reref_eeg - np.mean(reref_eeg, axis=0)) / np.std(reref_eeg, axis=0)

    session.eeg = reref_eeg
    return session


def signal_quality(session):
    """
    Remove bad or irrelevant channels using QualityChecker.
    Marks channels with '+' as irrelevant and stores relevant channel indices.
    """
    qc = QualityChecker()

    # Check timestamps
    if any(qc.consistent_timestamps(session.ts, session.fs)):
        return None, None

    # Combine all irrelevant channels
    ir_channels = np.hstack([
        qc.get_disconnected_channels(session.eeg, session.channels),
        qc.get_ekg_channel(session.eeg, session.channels),
        qc.get_marker_channels(session.eeg, session.channels)
    ]).astype(int)

    # Exclude channels with '+' in name
    ir_channels = np.append(ir_channels, [i for i, ch in enumerate(session.channels) if '+' in ch])
    ir_channels = np.array(ir_channels[ir_channels < session.eeg.shape[1]], dtype=int)

    # Keep only relevant channels
    relevant = [i for i in range(session.eeg.shape[1]) if i not in ir_channels]
    session.eeg = np.delete(session.eeg, ir_channels, axis=1)

    # Update channel names
    chan_names = np.array(session.channels)
    session.channels = np.delete(session.channels, ir_channels)
    chan_names[ir_channels] = 'XX'

    session.relevant_channels = relevant
    return session, chan_names


def process_session(session, reref=True, norm=True, band=[60, 200], smoothing=True, CL=False, window_smoothing=250):
    """
    Full preprocessing pipeline:
    1. Remove bad channels
    2. Rereference and normalize
    3. Bandpass filter
    4. Hilbert transform (smoothed)
    5. Trim data to game-related segment if CL session
    """
    filters = {'Band': [band[0], band[1]]}

    # Remove bad channels
    session, _ = signal_quality(session)
    if session is None:
        return 'invalid'

    # Rereference and normalize
    session = preprocess(session, reref, norm)
    session.raw_eeg = bandpass_filter(session.eeg)  # Store preprocessed EEG (time domain, 4-200Hz)
    
    # Bandpass filter to range of interest
    session = frequency_filters(session, filters)

    # Hilbert transform + smoothing
    session.eeg = hilbert_transform(session.eeg.copy(), smoothing, kernel_ms=window_smoothing)
    
    # Trim data of CL session to game-related segment only (no neurofeedback)
    if CL:
        move_prob = np.array(session.labels['MoveProb'])    # Movement probability only given during gameplay
        start_idx = np.argmax(move_prob != move_prob[0]) - 1            # First deviation from initial value
        end_idx = len(move_prob) - np.argmax(move_prob[::-1] != move_prob[-1]) - 1  # Last deviation
        session.eeg = session.eeg[start_idx:end_idx, :].copy()
        session.labels = {key: val[start_idx:end_idx] for key, val in session.labels.items()}
        session.raw_eeg = session.raw_eeg[start_idx:end_idx, :].copy()
        
    return session
