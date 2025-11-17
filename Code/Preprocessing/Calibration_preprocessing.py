#########################
# This script takes the raw calibration data and preprocesses it into high-frequency and low-frequency power.
# Written by P. Weger (paul.weger@maastrichtuniversity.nl)
#########################

import os
import numpy as np

# Local import of functions
from Functions.load_calibration import load
from Functions.process_session import process_session

#########################
# Settings
#########################

REREFERENCING = True
NORMALIZATION = True
SMOOTHING = True
HFA_BAND = np.array([60, 200])   # High-frequency
LFA_BAND = np.array([8, 30])     # Low-frequency

#########################
# Path handling
#########################

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Raw data path
path_calibdata = os.path.join(project_root, 'Data', 'Raw_data', 'Calibration')

# Output path
save_dir = os.path.join(project_root, 'Data', 'Preprocessed_data', 'Calibration')
os.makedirs(save_dir, exist_ok=True)

# Get all calibration filenames
files = os.listdir(path_calibdata)
calibration_files = [
    f for f in files 
    if f.startswith("calib_P") and f.endswith(".xdf")
]
calibration_files = sorted(
    calibration_files, 
    key=lambda x: int(os.path.splitext(x)[0].split("_P")[1])
)


#########################
# Processing
#########################

for filename in calibration_files:
    file_path = os.path.join(path_calibdata, filename)
    
    # Extract participant ID (e.g., P01)
    participant = os.path.splitext(filename)[0].split("_")[1]  # "P01"
    print(f"\n=== Processing {filename} ({participant}) ===")

    # ------ HFA ------
    session = load(file_path)
    session_hfa = process_session(session, REREFERENCING, NORMALIZATION, HFA_BAND, smoothing=SMOOTHING, window_smoothing=250)
    data_hfa = session_hfa.eeg           # numpy array (time x channels)
    labels = np.array(session_hfa.trial_names, dtype=object)
    channels = np.array(session_hfa.channels)      # list of channel names
    
    # Save HFA
    np.save(os.path.join(save_dir, f"calib_{participant}_hfa.npy"), data_hfa)
    print(f"Saved HFA -> calib_{participant}_hfa.npy")


    # ------ LFA ------
    session = load(file_path)
    session_lfa = process_session(session, REREFERENCING, NORMALIZATION, LFA_BAND, smoothing=SMOOTHING, window_smoothing=250)
    data_lfa = session_lfa.eeg

    # Save LFA
    np.save(os.path.join(save_dir, f"calib_{participant}_lfa.npy"), data_lfa)
    print(f"Saved LFA -> calib_{participant}_lfa.npy")

    # ------ Save labels and channel names ------
    np.save(os.path.join(save_dir, f"calib_{participant}_labels.npy"), labels)
    print(f"Saved labels -> calib_{participant}_labels.npy")
    np.save(os.path.join(save_dir, f"calib_{participant}_channels.npy"), channels)
    print(f"Saved channels -> calib_{participant}_channels.npy")

print("\n✅ Done. All participants processed.")
