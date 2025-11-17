#########################
# This script takes the raw closedLoop data and preprocesses it.
# Written by P. Weger (paul.weger@maastrichtuniversity.nl)
#########################

# Libaries
import os
import numpy as np
import re

# Local import of functions
from Functions.load_CL import load
from Functions.process_session import process_session
from Functions.helper_functions_error import label_alignment

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
path_CL = os.path.join(project_root, 'Data', 'Raw_data', 'ClosedLoop')
save_dir = os.path.join(project_root, 'Data', 'Preprocessed_data', 'ClosedLoop')
os.makedirs(save_dir, exist_ok=True)

# Get all closedloop filenames
files = os.listdir(path_CL)
CL_files = [
    f for f in files 
    if f.startswith("CL_P") and f.endswith(".xdf")
]
CL_files = sorted(
    CL_files,
    key=lambda x: os.path.splitext(x)[0].replace("CL_", "")
)
participants = sorted(set(re.findall(r"(P\d+)", "_".join(CL_files))))       # Participant IDs, independant of sessions

##################################
# Processing of closed-loop files
##################################
for ppt in participants:
    
    # Find all sessions for this participant
    ppt_files = sorted([f for f in CL_files if f"_{ppt}_" in f or f.endswith(f"{ppt}.xdf")])
    print(f"\n=== Processing participant {ppt} ({len(ppt_files)} sessions) ===")


    # ---- Process each session individually, then stich together ----
    hfa_list, lfa_list, rawEEG_list, labels_list, channels = [], [], [], [], None
    for s_idx, filename in enumerate(ppt_files):
        if len(ppt_files) > 1: print(f"Session {s_idx+1}/{len(ppt_files)}: {filename}")
        file_path = os.path.join(path_CL, filename)
        session = load(file_path)
        
        # ---- HFA ----
        session_hfa = process_session(session, REREFERENCING, NORMALIZATION, HFA_BAND,
                                      smoothing=SMOOTHING, CL=True, window_smoothing=250)
        hfa_list.append(session_hfa.eeg.copy())     
        rawEEG_list.append(session_hfa.raw_eeg.copy())

        # ---- LFA ----
        session = load(file_path)
        session_lfa = process_session(session, REREFERENCING, NORMALIZATION, LFA_BAND,
                                      smoothing=SMOOTHING, CL=True, window_smoothing=250)
        lfa_list.append(session_lfa.eeg.copy())

        # ---- Labels ----
        labels_list.append(session_hfa.labels.copy())
        channels = np.array(session_hfa.channels.copy())
        print(f"Session {(session_hfa.eeg.shape[0]/1024)/60:.2f} minutes long")
        
    # ---- Concatenate sessions ----
    data_hfa = np.concatenate(hfa_list, axis=0)
    data_lfa = np.concatenate(lfa_list, axis=0)
    data_raw = np.concatenate(rawEEG_list, axis=0)
    labels = label_alignment(labels_list) 
    
    # ---- Save stitched participant data ----
    np.save(os.path.join(save_dir, f"CL_{ppt}_hfa.npy"), data_hfa)
    np.save(os.path.join(save_dir, f"CL_{ppt}_lfa.npy"), data_lfa)
    np.save(os.path.join(save_dir, f"CL_{ppt}_rawEEG.npy"), data_raw)
    np.save(os.path.join(save_dir, f"CL_{ppt}_labels.npy"), labels)
    np.save(os.path.join(save_dir, f"CL_{ppt}_channels.npy"), channels)
    print(f"→ Saved data for {ppt}: {data_hfa.shape[0]} samples, {data_hfa.shape[1]} channels")

    
    
    
#######################################
# Same processing of error-correction files
#######################################
# Raw data path
path_CL = os.path.join(project_root, 'Data', 'Raw_data', 'ErrorCorrection')
save_dir = os.path.join(project_root, 'Data', 'Preprocessed_data', 'ErrorCorrection')
os.makedirs(save_dir, exist_ok=True)

# Get all closedloop filenames
files = os.listdir(path_CL)
CL_files = [
    f for f in files 
    if f.startswith("CL_P") and f.endswith(".xdf")
]
CL_files = sorted(
    CL_files,
    key=lambda x: os.path.splitext(x)[0].replace("CL_", "")
)

for filename in CL_files:
    file_path = os.path.join(path_CL, filename)
    
    # Extract participant ID (e.g., P01)
    participant = os.path.splitext(filename)[0].replace("CL_", "")
    print(f"\n=== Processing {filename} ({participant}) ===")

    # ------ HFA ------
    session = load(file_path)
    session_hfa = process_session(session, REREFERENCING, NORMALIZATION, HFA_BAND, smoothing=SMOOTHING, CL=True, window_smoothing=200)
    data_hfa = session_hfa.eeg           # numpy array (time x channels)
    labels = session_hfa.labels                     # dict of labels, upsampled to iEEG sampling rate
    channels = np.array(session_hfa.channels)      # list of channel names
    data_raw = session_hfa.raw_eeg
    
    # Save HFA
    np.save(os.path.join(save_dir, f"CL_{participant}_hfa.npy"), data_hfa)
    np.save(os.path.join(save_dir, f"CL_{participant}_rawEEG.npy"), data_hfa)
    print(f"Saved HFA -> CL_{participant}_hfa.npy")


    # ------ LFA ------
    session = load(file_path)
    session_lfa = process_session(session, REREFERENCING, NORMALIZATION, LFA_BAND, smoothing=SMOOTHING, CL=True, window_smoothing=200)
    data_lfa = session_lfa.eeg

    # Save LFA
    np.save(os.path.join(save_dir, f"CL_{participant}_lfa.npy"), data_lfa)
    print(f"Saved LFA -> CL_{participant}_lfa.npy")

    # ------ Save labels and channel names ------
    np.save(os.path.join(save_dir, f"CL_{participant}_labels.npy"), labels)
    print(f"Saved labels -> CL_{participant}_labels.npy")
    np.save(os.path.join(save_dir, f"CL_{participant}_channels.npy"), channels)
    print(f"Saved channels -> CL_{participant}_channels.npy")


print("\n✅ Done. All participants processed.")
