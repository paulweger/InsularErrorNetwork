#########################
# This script takes the preprocessed calibration data and analyzes movement results and decoder training.
# Written by P. Weger (paul.weger@maastrichtuniversity.nl)
# 17th of November 2025
#########################

# Libaries
import os
import csv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

# Function imports
from Functions.windowing import windowing, mode
from Functions.Plotting_functions.plot_motorCalib import *
from Functions.helper_functions_motor import *


# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
preproc_path = os.path.join(project_root, 'Data', 'Preprocessed_data', 'Calibration')
elecLoc_path = os.path.join(project_root, 'Data', 'Electrode_locations')

# Participants
all_files = os.listdir(preproc_path)
participants = sorted(set(f.split('_')[1] for f in all_files if f.startswith('calib_P')))

# Window params (samples)
WINDOW_LENGTH, WINDOW_SHIFT = 256, 256      # Non-overlapping
fs = 1024

# Keywords for anatomical regions
anatomical_locations = {
    'Precentral gyrus': ['precentral'],
    'Postcentral gyrus': ['postcentral'],
    'Posterior Insula': ['ctx_rh_g_ins_lg_and_s_cent_ins', 'ctx_rh_s_circular_insula_inf']
}
pooled_hfa = {region: [] for region in anatomical_locations}

# Loop over participants
print(f"\n--------- Starting movement analysis ---------")
all_hfa_motorRelevant, all_lfa_motorRelevant, motorHFA, motorLFA, motorInsula, X, Y = [], [], {}, {}, {}, [], []
for ppt in participants:
    
    # Load preprocessed data
    hfa = np.load(os.path.join(preproc_path, f'calib_{ppt}_hfa.npy'))
    lfa = np.load(os.path.join(preproc_path, f'calib_{ppt}_lfa.npy'))
    labels = np.load(os.path.join(preproc_path, f'calib_{ppt}_labels.npy'), allow_pickle=True)
    channels = np.load(os.path.join(preproc_path, f'calib_{ppt}_channels.npy'))
    
    # Load channel locations
    with open(os.path.join(elecLoc_path, f'electrode_locations_{ppt}.csv'), encoding='utf-8') as f:
        r = list(csv.DictReader(f))
        elec_names, elec_locs, roi = (np.array([row[k] for row in r]) for k in ('electrode_name_1', 'location', 'ROI'))
    
    # Find anatomically relevant channels in 3 target areas of anatomical_locations
    anat_idx = {}
    for region, keywords in anatomical_locations.items():
        matched_elecs = [name for name, loc in zip(elec_names, elec_locs)
                        if any(kw.lower() in str(loc).lower() for kw in keywords)]
        idx = np.where(np.isin(channels, matched_elecs))[0]
        anat_idx[region] = idx

    # Window and average per window
    hfa_win = windowing(hfa, WINDOW_LENGTH, WINDOW_SHIFT).mean(axis=1)
    lfa_win = windowing(lfa, WINDOW_LENGTH, WINDOW_SHIFT).mean(axis=1)
    trial_labels = np.array([1 if t=='Move' else 0 for t in mode(windowing(labels, WINDOW_LENGTH, WINDOW_SHIFT))])
    
    # Save features/trials and labels for decoder training
    features_windowed = np.hstack([hfa_win, lfa_win])            
    labels_windowed = trial_labels                                
    X.append(features_windowed)
    Y.append(labels_windowed)

    # t-test per electrode
    move, rest = trial_labels==1, trial_labels==0
    hfa_t, hfa_p = ttest_ind(hfa_win[move], hfa_win[rest], axis=0, equal_var=False)
    lfa_t, lfa_p = ttest_ind(lfa_win[move], lfa_win[rest], axis=0, equal_var=False)

    # Get significant electrodes (p < 0.05) & motor relevant electrodes (|t|>4)
    hfa_sig_idx = np.where(hfa_p<0.05)[0]
    hfa_motor_idx = np.where((hfa_p<0.05) & (np.abs(hfa_t)>4))[0]       
    lfa_sig_idx = np.where(lfa_p<0.05)[0]
    lfa_motor_idx = np.where((lfa_p<0.05) & (np.abs(lfa_t)>4))[0]
    print(f"\n{ppt}: HFA {len(hfa_sig_idx)}/{len(hfa_p)} sig., {len(hfa_motor_idx)} highly motor-rel.")
    print(f"LFA {len(lfa_sig_idx)}/{len(lfa_p)} sig., {len(lfa_motor_idx)} highly motor-rel.")
    
    # Save values for insula distribution
    roi_map = {name: int(r) if r.isdigit() else np.nan for name, r in zip(elec_names, roi)}
    roi_for_channels = np.array([roi_map.get(ch, np.nan) for ch in channels])
    motorInsula[ppt] = {
        'channels': channels.tolist(),                    # all recorded electrodes
        'tvalues': np.round(hfa_t, 2).tolist(),           # t-values for all electrodes
        'roi': roi_for_channels.tolist()                  # corresponding ROI numbers
    }
    
    # Save motor relevant channels for brainplot
    hfa_sig_chans = channels[hfa_sig_idx]
    lfa_sig_chans = channels[lfa_sig_idx]
    ppt_id = int(ppt.replace("P", ""))
    motorHFA[ppt_id] = {
        'channels': channels[hfa_sig_idx].tolist(),
        'tvalues': np.round(hfa_t[hfa_sig_idx], 2).tolist(),
        'roi': roi_for_channels[hfa_sig_idx].tolist()
    }
    motorLFA[ppt_id] = {
        'channels': channels[lfa_sig_idx].tolist(),
        'tvalues': np.round(lfa_t[lfa_sig_idx], 2).tolist(),
        'roi': roi_for_channels[lfa_sig_idx].tolist()
    }

    # Movement-onset average for motor relevant channels
    interval = 2*fs
    movement = (labels=='Move').astype(int)
    onsets = np.where(np.diff(movement)==1)[0]
    hfa_move = np.array([hfa[o-interval:o+interval, :] for o in onsets])
    hfa_init = hfa_move.mean(axis=0)
    if len(hfa_motor_idx): all_hfa_motorRelevant.append(hfa_init[:,hfa_motor_idx])
    lfa_move = np.array([lfa[o-interval:o+interval, :] for o in onsets])
    lfa_init = lfa_move.mean(axis=0)
    if len(lfa_motor_idx): all_lfa_motorRelevant.append(lfa_init[:,lfa_motor_idx])
    
    # Pool electrodes in target areas
    for region, idx in anat_idx.items():
        if len(idx) == 0: continue  # skip if no channels in this region
        pooled_hfa[region].append(hfa_move[:, :, idx])


# Insula's spatial distribution of motor relevant channels
print("\nInsula spatial distribution of motor-relevant channels:")
coord_vals, kde_all, kde_t2, kde_t4 = compute_spatial_kdes(motorInsula)
plot_insula_kde(coord_vals, kde_all, kde_t2, kde_t4)


############ Regional HFA plot #############
# Stack regional HFA across patients
regional_hfa = {}
for region, region_data in pooled_hfa.items():          # Loop over region
    electrode_averaged_trials = []
    for pdata in region_data:                           # Loop over participants
        if pdata.shape[2]==0: continue
        avg = pdata.mean(axis=2)                        # average over electrodes per participant
        baseline = avg[:, int(0.5*fs):int(1.5*fs)].mean()
        electrode_averaged_trials.append(avg - baseline)          # Stack baseline corrected trials
    regional_hfa[region] = np.vstack(electrode_averaged_trials) if electrode_averaged_trials else np.array([])
    
# Plot regional HFA time-courses
regional_hfa_path = os.path.join(project_root, 'Plotting', 'Fig2', 'Regional_HFA.pdf')
plot_regional_activity(regional_hfa, save_path=regional_hfa_path)



############ HFA / LFA at movement onset plot #############

# Stack motor relevant electordes across patients
hfa_init = np.concatenate(all_hfa_motorRelevant, axis=1) if all_hfa_motorRelevant else np.array([])
lfa_init = np.concatenate(all_lfa_motorRelevant, axis=1) if all_lfa_motorRelevant else np.array([])
print(f"\nCombined motor-rel channels in HFA: {hfa_init.shape[1]}, LFA: {lfa_init.shape[1]}")
    
# Safe motor relevant channels and tvalues
save_path = os.path.join(project_root, 'Plotting', 'Plotting_values')
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'motorHFAchannels.pkl'), 'wb') as f:
    pickle.dump(motorHFA, f)
with open(os.path.join(save_path, 'motorLFAchannels.pkl'), 'wb') as f:
    pickle.dump(motorLFA, f)
    
# Plot HFA at movement onset
fig_path = os.path.join(project_root, 'Plotting', 'Fig2', 'HFA_average.pdf')
plot_motor_power(hfa_init, fs, ytick_vals=[0,50,100,150], title="HFA at movement onset", save_path=fig_path)

# Plot LFA at movement onset
fig_path = os.path.join(project_root, 'Plotting', 'Fig2', 'LFA_average.pdf')
plot_motor_power(lfa_init, fs, ytick_vals=[0,100,200,300], title="LFA at movement onset", save_path=fig_path)



############ Decoder training #############

# Incrementally train decoder for every participant
algorithm = 'LDA'                   # 'LDA' or 'RNN', RNN takes very long (hours-days)
print(f"\nIncremental decoder training: ")
print("Using algorithm:", algorithm, ". LDA computes fast, RNN takes longer but was using for our study")
decoding_results = {}
for i, ppt in enumerate(participants):
    x = X[i]            # [trials, features]
    y = Y[i]            # [trials]
    prior = [0.5, 0.5]
    
    # Run incremental training
    print(f"\n{ppt}:")
    fold_accuracies, fold_increments = train_incremental(x, y, prior = prior, algorithm = algorithm, runs=50, test_fraction=0.2)     # [runs, increments], [runs, increments]
    avg_accuracies = np.mean(fold_accuracies, axis=0)      # [increments]
    std_accuracies = np.std(fold_accuracies, axis=0)        # [increments]
    
    # Transform increments to seconds and accumulate participants
    total_seconds = 0.8 * X[i].shape[0] * WINDOW_LENGTH / fs
    x_vals = fold_increments[0] * WINDOW_LENGTH / fs 
    x_ticks = np.arange(0, total_seconds + 60, 60)        # Make seconds out of increments
    x_tick_labels = [f"{int(t // 60)}" for t in x_ticks]        # Make labels for major ticks
    decoding_results[ppt] = {
        'accuracy': avg_accuracies,
        'accuracy_std': std_accuracies,
        'x_vals': x_vals,
        'x_ticks': x_ticks,
        'x_tick_labels': x_tick_labels
    }
    
# Statistics of decoder training
onemin, twomin, top = zip(*[
    (d['accuracy'][8], d['accuracy'][11], d['accuracy'][-1])            # 8: 1min, 11: 2min, -1: final
    for d in decoding_results.values()
])
onemin = (np.array(onemin) - 0.5) / (np.array(top) - 0.5)               # subtract chance level to get improvement of top accuracy - chance level
twomin = (np.array(twomin) - 0.5) / (np.array(top) - 0.5)
print(f"\n1-min improvement: {100*onemin.mean():.1f} %")
print(f"2-min improvement: {100*twomin.mean():.1f} %")

# Plot incremental decoding results
fig_path = os.path.join(project_root, 'Plotting', 'Fig2', 'inc_training.pdf')
plot_incremental_training(decoding_results, save_path=fig_path)
    