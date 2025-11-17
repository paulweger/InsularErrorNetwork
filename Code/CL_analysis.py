#########################
# This script takes the preprocessed closedloop data and analyzes it for error and behavior.
# Written by P. Weger (paul.weger@maastrichtuniversity.nl)
# 17th of November 2025
#########################

# Libaries
import os
import csv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.signal import welch

# Function imports
from Functions.behavior import compute_behavior, plot_behavior, plot_error_vs_standard
from Functions.helper_functions_error import *            
from Functions.Plotting_functions.plot_errorPotentials import *  



# Paths of closed-loop game data
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
preproc_path = os.path.join(project_root, 'Data', 'Preprocessed_data', 'ClosedLoop')
elecLoc_path = os.path.join(project_root, 'Data', 'Electrode_locations')

# Laod participant info
all_files = os.listdir(preproc_path)
cl_files = [f for f in all_files if f.startswith("CL_P") and f.endswith(".npy")]
participants = sorted( set(os.path.splitext(f)[0].split("_")[1] for f in cl_files) )
fs = 1024

# Loop over participants
print(f"\n--------- Starting closed-loop error analysis ---------")
avgSuccess, time_behavior, rollingBehavior = {}, {}, {}             # For accumulating behavioral results
errorResults = {ppt: {} for ppt in participants}                    # Accumulate error analysis results
for ppt in participants:
    print(f"\nParticipant {ppt}")
    
    # Load preprocessed data
    hfa = np.load(os.path.join(preproc_path, f'CL_{ppt}_hfa.npy'))
    if ppt == 'P01': lfa = np.load(os.path.join(preproc_path, f'CL_{ppt}_lfa.npy'))
    rawEEG = np.load(os.path.join(preproc_path, f'CL_{ppt}_rawEEG.npy'))
    labels = np.load(os.path.join(preproc_path, f'CL_{ppt}_labels.npy'), allow_pickle=True).item()
    channels = np.load(os.path.join(preproc_path, f'CL_{ppt}_channels.npy'))
    
    # Load channel locations
    with open(os.path.join(elecLoc_path, f'electrode_locations_{ppt}.csv'), encoding='utf-8') as f:
        r = list(csv.DictReader(f))
        elec_names, elec_locs, roi = (np.array([row[k] for row in r]) for k in ('electrode_name_1', 'location', 'ROI'))
        roi = roi.astype(int)
    
    # Compute game behavior and save for plotting
    behavior = compute_behavior(labels)
    avgSuccess[ppt] = np.mean(behavior['trialSuccess'])
    time_behavior[ppt] = behavior['t']
    rollingBehavior[ppt] = behavior['avgPerformance']

    # Identify correct/erroneous lane changes
    player, obstacle, lanechange_command, moveprob = labels['Player_lane'], labels['Traffic_lane'], labels['Smoothed_command'], labels['MoveProb']
    changes = np.where(np.diff(player) != 0)[0] + 1                        # Player switches lane
    correct_moves = changes[player[changes] != obstacle[changes]]           # Player moves away from obstacle lane
    error_moves = changes[player[changes] == obstacle[changes]]             # Player moves into obstacle lane
    print(f"Correct switches: {len(correct_moves)}, Erroneous switches: {len(error_moves)}")
    
    # Get HFA segments around lane changes
    before, after = 1*fs, 3*fs    # samples before and after event
    valid_correct, valid_error = remove_uncompleted_trials(hfa, correct_moves, error_moves)  # Ensure that lane change didn't happen too early/late so that we can get full segment
    correctSeg = np.stack([hfa[i-before:i+after] for i in valid_correct])              # Shape: [trial, samples in time, channels] HFA power
    errorSeg  = np.stack([hfa[i-before:i+after] for i in valid_error])  
    correctTimeSeg = np.stack([rawEEG[i-before:i+after] for i in valid_correct])       # Shape: [trial, samples in time, channels] time-domain EEG
    errorTimeSeg  = np.stack([rawEEG[i-before:i+after] for i in valid_error])
        
    # Calculate average power and tvalue/pvalue between erroneous and correct trials
    powerCorrect = power_windowing(correctSeg[:,before:before+fs,:])         # shape: (trials, 3 windows, channels)
    powerError = power_windowing(errorSeg[:,before:before+fs,:])              # shape: (trials, 3 windows, channels)
    tvals, pvals, window_idx = ttest_best_window(powerError, powerCorrect)      # For every channel: tval, pval, window indices
    meanPowerCorrect = powerCorrect[:, window_idx, np.arange(powerCorrect.shape[2])].mean(axis=0)     # shape: (channels,) of average power during best performing 500-ms window
    meanPowerError   = powerError[:, window_idx, np.arange(powerError.shape[2])].mean(axis=0)         # shape: (channels,)
    powerTrialsCorrect = powerCorrect[:, window_idx, np.arange(powerCorrect.shape[2])]                  # shape: (trials, channels)     -> best-performing 500-ms window
    powerTrialsError = powerError[:, window_idx, np.arange(powerError.shape[2])]                        # shape: (trials, channels)    ️ -> best-performing 500-ms window
    featuresCorrect = powerCorrect[:, 1, np.arange(powerCorrect.shape[2])]                              # shape: (trials, channels)     -> central 500-ms window 
    featuresError = powerError[:, 1, np.arange(powerError.shape[2])]                                    # shape: (trials, channels)     -> central500-ms window 
    
    # Get significant channels and their tvalue
    sig_idx = np.where(pvals < 0.05)[0]
    sig_channel_names = [channels[i] for i in sig_idx]
    sig_tvals, sig_pvals = tvals[sig_idx], pvals[sig_idx]
    print(f"{len(sig_idx)} / {len(channels)} sig. channels, highest tval: {np.max(np.abs(tvals)):.1f} in ch {channels[np.argmax(np.abs(tvals))]}")
    
    # Find every electrodes region of interest 
    channel_roi = np.array([
        roi[elec_names == ch][0] if np.any(elec_names == ch) else 0
        for ch in channels
    ])
    n_elecs_per_roi = np.array([np.sum(roi == i) for i in range(5)])        # 5 ROIs (0: noROI, 1: aINS, 2: PFC, 3: centroparietal, 4: pINS)
    
    # Compute specturm over whole experimental session
    fs, nperseg, noverlap = 1024, 1024, 512
    freqs, psd = welch(rawEEG, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)  # shape psd: (freq_bins, channels)
    
    # Plot decoder prediction over time for one exemplary patient
    if ppt == 'P01':
        hfa_chans, lfa_chans = top_corr_channels(hfa, lfa, moveprob)
        plot_motor_heatmap(hfa[:, hfa_chans], lfa[:, lfa_chans], moveprob)
        print("-> Plotted motor heatmap")
    
    
    # Save results from this patient
    errorResults[ppt] = {
        'channels': np.array(channels),                           # [channels]
        'tvalues': np.array(tvals),                               # [channels]
        'pvalues': np.array(pvals),                               # [channels]
        'sig_idx': np.array(sig_idx),                             # [n_sig_channels]
        'roi': np.array(channel_roi),                             # [channels]
        'n_elecs_per_roi': n_elecs_per_roi,                       # [ROIs]
        'meanPowerCorrect': np.array(meanPowerCorrect),           # [channels]
        'meanPowerError': np.array(meanPowerError),               # [channels]
        'powerTrialsCorrect': np.array(powerTrialsCorrect),       # [trials, channels] best-performing 500-ms window
        'powerTrialsError': np.array(powerTrialsError),           # [trials, channels] best-performing 500-ms window
        'featuresCorrect': np.array(featuresCorrect),             # [trials, channels] 500-ms window in middle
        'featuresError': np.array(featuresError),                 # [trials, channels] 500-ms window in middle
        'correctSeg': np.array(correctSeg),                       # [trials, time-samples, channels]    HFA Power
        'errorSeg': np.array(errorSeg),                           # [trials, time-samples, channels]    HFA Power
        'correctTimeSeg': np.array(correctTimeSeg),               # [trials, time-samples, channels]    time-domain EEG
        'errorTimeSeg': np.array(errorTimeSeg),                   # [trials, time-samples, channels]    time-domain EEG
        'psd': np.array(psd),                                     # [freq_bins, channels]
        'freqs': np.array(freqs)                                  # [freq_bins]
    }
    
    
# Plot overall behavior
path_save_behavior = os.path.join(project_root, 'Plotting', 'Fig2', 'behavior.pdf')
plot_behavior(time_behavior, rollingBehavior, avgSuccess, save_path = path_save_behavior)
print("\n→ Saved behavior plot under Plotting/Fig2/behavior.pdf")

# Safe error-relevant channels and tvalues for brainplots 
errorChannels = {}
for ppt, res in errorResults.items():
    idx = int(ppt.replace("P", ""))
    sig = res['sig_idx']                       # boolean mask or index array per participant
    errorChannels[idx] = {
        'channels': res['channels'][sig],           # Only save significant channels
        'tvalues':  res['tvalues'][sig],            # and their tvalues
        'roi':      res['roi'][sig],                # and their ROIs
    }
save_path = os.path.join(project_root, 'Plotting', 'Plotting_values')
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'errorChannels.pkl'), 'wb') as f:
    pickle.dump(errorChannels, f, protocol=pickle.HIGHEST_PROTOCOL)

# Perfentage of sig. electrodes in ROI
total_elecs = sum(r['n_elecs_per_roi'] for r in errorResults.values())
total_sig_elecs = sum(np.bincount(np.array(r['roi'], int)[r['sig_idx']], minlength=5) 
                      for r in errorResults.values())
percent_sig_per_roi = 100 * total_sig_elecs / total_elecs     # Holds % of noROI, aINS, PFC, centroparietal, pINS
print(f"\nError responsive elecs in 3 ROIs:")
print(f"→ aINS: {percent_sig_per_roi[1]:.1f}%, PFC: {percent_sig_per_roi[2]:.1f}%, Centroparietal: {percent_sig_per_roi[3]:.1f}%")

# Plot significant electrode percentages in bar plot
plot_sig_electrodes(percent_sig_per_roi, total_sig_elecs, total_elecs,
                    save_path=os.path.join(project_root, 'Plotting', 'Fig3', 'sig_electrodes.pdf'))
print(f"→ Saved sig. electrode percentage plot under Plotting/Fig3/sig_electrodes.pdf")

# Group error results by ROI across participants, plot violin and group response
print(f"\nGrouping electrodes into ROI:")
groupedResults_allElecs = group_error_results_by_roi(errorResults, t_thresh=0)        # All electrodes
groupedResults_sig = group_error_results_by_roi(errorResults, t_thresh=2)        # Significant electrodes
groupedResults_t4 = group_error_results_by_roi(errorResults, t_thresh=4) 

# Error power comparisson and temporal dynamics
dataframe_linMixedEffects = build_lmm_dataframe(errorResults)               # Build dataframe for LMM analysis  
plot_roi_violin(groupedResults_allElecs, dataframe_linMixedEffects)                             # Plot spaghetti plot of HFA increases
print(f"\nTemporal dynamics of error processing:")
plot_grouped_activity(groupedResults_allElecs, mode='trialwise')     # Plot group activity over time for three ROIs

# Timing of error-related potentials: look at highly-responsive electrodes only
print(f"\nPeak latency of highly-responsive electrodes")
plot_error_hfa(groupedResults_t4, mode='trialwise')     # Plot closeup of highly responsive electrodes for peak timing
plot_peak_latencies(groupedResults_t4)

# Frequency analysis of highly-responsive electrodes
print(f"\nSpectral contributions:")
f, spectral_mean, spectral_sem, sig_mask = calculate_spectral_increase(groupedResults_t4)
plot_power_change(f, spectral_mean, spectral_sem, sig_mask)

# Get spatial distribution across insula
print("\nSpatial distribution of error-responsive electrodes in insula")
coord_vals, kde_all, kde_t2, kde_t4 = compute_spatial_kdes(errorResults)
plot_insula_kde(coord_vals, kde_all, kde_t2, kde_t4)

# Spectrogram of selected ant. Insula electrode (P05, ch RX7)
ch = 'RX7'
ppt_id = 'P05'
print(f"\nPlotting Spectrogram (tval) of ch {ch} from {ppt_id}")
idx_spectrogram_chan = np.where(errorResults[ppt_id]['channels'] == ch)
correctSegment = errorResults['P05']['correctTimeSeg'][:,:,idx_spectrogram_chan] # [4096 samples in time, trials
errorSegment = errorResults['P05']['errorTimeSeg'][:,:,idx_spectrogram_chan]
t_values, p_values, significant_mask, f, t = tval_spectrogram(np.squeeze(correctSegment), np.squeeze(errorSegment))
plot_tval_spectrogram(t_values, p_values, significant_mask, f, t)

# Decoding across regions
print("\nDecoding error responses across ROIs.")
roiDecoding = temporal_decoding(groupedResults_t4, n_runs=100, n_sig_runs=100)    
plot_temporal_decoding(roiDecoding)

# Decoding across all participants
print("\nDecoding error responses across participants.")
accuracy, std, sig = decode_participants(errorResults, n_repeats=100, n_shuffles=100)
plot_decoding_results(accuracy, std, sig)

# Confusion matrices for exemplary participant (P09)
participant_confusion = 'P09'
cm_neutral = decode_confusion_single(errorResults, participant_confusion, target_fpr=0.2)
plot_confusion_matrix(cm_neutral, 'Neutral error decoding')
cm_conservative = decode_confusion_single(errorResults, participant_confusion, target_fpr=0.05)
plot_confusion_matrix(cm_conservative, 'Conservative error decoding')








########################################
# Error correction behavioral results
########################################
print(f"\n--------- Error correction behavioral analysis ---------")

# Get error files from respective folder
errorCorr_path = os.path.join(project_root, 'Data', 'Preprocessed_data', 'ErrorCorrection')
all_files = os.listdir(errorCorr_path)
clError_files = [f for f in all_files if f.startswith("CL_P") and f.endswith("error_labels.npy")]
participants_error = sorted(set(f.split("_")[1] for f in clError_files))

# Loop over participants
avgSuccessError, time_behavior_error, rollingBehaviorError = {}, {}, {}
for ppt_err in participants_error:

    # Load error labels
    labels = np.load(os.path.join(errorCorr_path, f'CL_{ppt_err}_error_labels.npy'), allow_pickle=True).item()

    # Compute game behavior and save for plotting
    behavior = compute_behavior(labels)
    avgSuccessError[ppt_err] = np.mean(behavior['trialSuccess'])
    time_behavior_error[ppt_err] = behavior['t']
    rollingBehaviorError[ppt_err] = behavior['avgPerformance']
    
# Plot overall behavior with error correction
path_errorCorr = os.path.join(project_root, 'Plotting', 'Fig5', 'behavior_errorCorr.pdf')
plot_error_vs_standard(
    time_behavior, rollingBehavior, avgSuccess,                             # Normal behavior
    time_behavior_error, rollingBehaviorError, avgSuccessError,             # Error correction behavior
    save_path=path_errorCorr
)
print("→ Saved error-correction behavior plot under Plotting/Fig5/behavior_errorCorr.pdf")

                           