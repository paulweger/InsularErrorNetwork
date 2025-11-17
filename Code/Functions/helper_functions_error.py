import numpy as np
import copy
import pandas as pd
from pathlib import Path
from scipy.signal import welch, spectrogram
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_1samp, ttest_ind, gaussian_kde
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat


def label_alignment(labels):
    """
    Aligns labels across multiple sessions by concatenating elements and 
    subtrating offsets. Only applies to 4 game variables (2x time, points, deaths)+
    So 2 sessiosn are seemlessly united together without jumps in time or points.
    """
    if len(labels) == 1:
        return labels[0]

    labels_aligned = copy.deepcopy(labels[0])
    offset_vars = ['Time_game', 'timestamp', 'Points', 'Death_count']

    # Initialize offsets for continuous variables
    offsets = {key: labels_aligned[key][-1] for key in offset_vars}

    # Iterate through remaining sessions
    for sess in labels[1:]:
        for key, values in sess.items():
            if key in offset_vars:
                # Apply offset continuation
                adjusted = values - values[0] + offsets[key]
                labels_aligned[key] = np.concatenate([labels_aligned[key], adjusted])
                offsets[key] = adjusted[-1]
            else:
                # Just concatenate
                labels_aligned[key] = np.concatenate([labels_aligned[key], values])

    return labels_aligned



def power_windowing(seg, window_len=512):
    # --- Compute average power in best performing 500 ms window ---
    first   = seg[:, :window_len, :].mean(axis=1)
    middle  = seg[:, window_len//2 : window_len//2 + window_len, :].mean(axis=1)
    last    = seg[:, -window_len:, :].mean(axis=1)
    stacked_power = np.stack([first, middle, last], axis=1)  # shape: (trials, 3, channels)
    return stacked_power


def ttest_best_window(seg_correct, seg_error):
    """
    Compute t-test between error vs correct trials for multiple windows.
    seg_correct / seg_error: (trials, windows, channels)
    
    Returns:
        max_t      : t-value of best window for each channel
        p_best     : p-value of best window for each channel
        best_idx   : index of the best-performing window per channel
    """
    # t-test along trials (axis=0), result: (windows, channels)
    t_vals, p_vals = ttest_ind(seg_error, seg_correct, axis=0)
    
    # Best-performing window per channel = window with max absolute t-value
    best_idx = np.argmax(np.abs(t_vals), axis=0)  # shape: (channels,)
    max_t = t_vals[best_idx, np.arange(t_vals.shape[1])]
    p_best = p_vals[best_idx, np.arange(t_vals.shape[1])]
    
    return max_t, p_best, best_idx



def group_error_results_by_roi(errorResults, target_rois=(1, 2, 3), t_thresh=2.0):
    """
    Combine data across participants into ROIs above a certain t-value threshold.
    """
    # Initialize output files
    grouped = {
        roi: {
            'trialwise_correct': [],
            'trialwise_error': [],
            'electrodewise_correct': [],
            'electrodewise_error': [],
            'channels': [],
            'tvalues': [],
            'pvalues': [],
            'sig_idx': [],
            'participant_idx': [],
            'meanPowerCorrect': [],
            'meanPowerError': [],
            'generalSpectrum': [],
            'f_general': [],
            'errorSpectrum': [],
            'f_error': [],
            'correctSegments': [],
            'errorSegments': []
        } for roi in target_rois
    }

    # Loop over participants and extract data 
    for p_idx, r in enumerate(errorResults.values(), start=1):
        roi_all = np.array(r['roi'])
        tvals = np.array(r['tvalues'])
        pvals = np.array(r['pvalues'])
        channels = np.array(r['channels'])
        meanPowerCorrect = np.array(r['meanPowerCorrect'])
        meanPowerError = np.array(r['meanPowerError'])
        correctSeg = np.array(r['correctSeg'])  # (trials_correct, time, channels)
        errorSeg = np.array(r['errorSeg'])      # (trials_error, time, channels)
        errorTimeSeg = np.array(r['errorTimeSeg'])  # (trials_error, time, channels)
        spectrum = np.array(r['psd'])           # (frequencies, channels)
        freqs = np.array(r['freqs'])              # (frequencies,)
        
        # Loop over ROIs
        for roi in target_rois:
            # Skip if no electrode in this roi
            roi_mask = roi_all == roi
            if not np.any(roi_mask):
                continue
            
            # Skip if no electrode in roi above t
            sig_mask = np.abs(tvals[roi_mask]) > t_thresh
            if not np.any(sig_mask):
                continue
            sel_idx = np.where(roi_mask)[0][sig_mask]

            # Trialwise: average across selected electrodes for correct and error separately
            avg_corr = np.nanmean(correctSeg[:, :, sel_idx], axis=2)  # (n_trials_corr, n_time)
            avg_err  = np.nanmean(errorSeg[:, :, sel_idx], axis=2)    # (n_trials_err,  n_time)
            grouped[roi]['trialwise_correct'].append(avg_corr)
            grouped[roi]['trialwise_error'].append(avg_err)

            # Electrodewise: mean across correct trials per electrode and mean across error trials per electrode
            corr_avg_per_elec = np.nanmean(correctSeg[:, :, sel_idx], axis=0).T  # (n_elec_sel, n_time)
            err_avg_per_elec  = np.nanmean(errorSeg[:, :, sel_idx], axis=0).T    # (n_elec_sel, n_time)
            grouped[roi]['electrodewise_correct'].append(corr_avg_per_elec)
            grouped[roi]['electrodewise_error'].append(err_avg_per_elec)
            
            # Save all correct/error segments for later use
            grouped[roi]['correctSegments'].append(correctSeg[:, :, sel_idx])
            grouped[roi]['errorSegments'].append(errorSeg[:, :, sel_idx])

            # Metadata
            grouped[roi]['channels'].extend(channels[sel_idx].tolist())
            grouped[roi]['tvalues'].extend(tvals[sel_idx].tolist())
            grouped[roi]['pvalues'].extend(pvals[sel_idx].tolist())
            grouped[roi]['sig_idx'].extend(sel_idx.tolist())
            grouped[roi]['meanPowerCorrect'].extend(meanPowerCorrect[sel_idx].tolist())
            grouped[roi]['meanPowerError'].extend(meanPowerError[sel_idx].tolist())
            grouped[roi]['participant_idx'].extend([p_idx] * len(sel_idx))
            
            # Spectrum of erroneous trials & session-wide (general)
            elecs_error = []
            for elec_idx in sel_idx:
                
                # Calculate welch for every trial of every electrode
                psd_err = [welch(trial, fs=1024, nperseg=256, noverlap=128) for trial in errorTimeSeg[:, 1024:2048, elec_idx]]
                f = psd_err[0][0]
                mean_psd_err  = np.mean([p[1] for p in psd_err], axis=0)
                elecs_error.append(mean_psd_err)
            
            grouped[roi]['errorSpectrum'].extend(np.array(elecs_error))             # [n_elec_sel, n_freq_error]
            grouped[roi]['f_error'] = f                                  # [n_freq_error]
            grouped[roi]['f_general'] = freqs                             # [n_freq_general]
            grouped[roi]['generalSpectrum'].extend(spectrum[:, sel_idx].T)          # [n_elec_sel, n_freq_general]
                

    # Convert lists -> numpy arrays (vstack where appropriate)
    for roi in target_rois:
        g = grouped[roi]
        g['trialwise_correct'] = np.vstack(g['trialwise_correct']) if g['trialwise_correct'] else np.empty((0,))
        g['trialwise_error']   = np.vstack(g['trialwise_error'])   if g['trialwise_error']   else np.empty((0,))
        g['electrodewise_correct'] = np.vstack(g['electrodewise_correct']) if g['electrodewise_correct'] else np.empty((0,))
        g['electrodewise_error']   = np.vstack(g['electrodewise_error'])   if g['electrodewise_error']   else np.empty((0,))
        for key in ['channels','tvalues','pvalues','sig_idx','meanPowerCorrect','meanPowerError']:
            g[key] = np.array(g[key])
          
        # Stack session-wise/error-trial spectrum into [channels, frequencies]
        for key in ['errorSpectrum', 'generalSpectrum']:
            g[key] = np.stack(g[key], axis=0)

    return grouped

def build_lmm_dataframe(errorResults):
    """
    Convert errorResults dict into a long-format DataFrame suitable for
    linear mixed-effects modeling of HFA (power) values.

    Each row = one trial (correct or error)
    Columns: participant, electrode, roi, condition, HFA
    """
    rows = []

    for ppt_id, data in errorResults.items():
        rois = np.array(data['roi'])
        power_corr = np.array(data['powerTrialsCorrect'])   # [trials, channels]
        power_err  = np.array(data['powerTrialsError'])     # [trials, channels]

        n_ch = power_corr.shape[1]
        for ch_idx in range(n_ch):
            roi = rois[ch_idx]

            # Correct trials
            for val in power_corr[:, ch_idx]:
                rows.append({
                    "participant": ppt_id,
                    "electrode": f"{ppt_id}_ch{ch_idx+1}",
                    "roi": roi,
                    "condition": "correct",
                    "HFA": val
                })

            # Error trials
            for val in power_err[:, ch_idx]:
                rows.append({
                    "participant": ppt_id,
                    "electrode": f"{ppt_id}_ch{ch_idx+1}",
                    "roi": roi,
                    "condition": "error",
                    "HFA": val
                })

    df = pd.DataFrame(rows)
    return df



def remove_uncompleted_trials(rawEEG, correct_moves, error_moves, before=1024, after=3072):
    """
    Remove trials that were not completed / noisy through a jump in data
    Jumps are detected from large amplitude discontinuities in rawEEG.
    """
    n_samples = rawEEG.shape[0]
    fs_neural, fs_markers = 1024, 48
    correct_moves -= int(fs_neural / (2*fs_markers))    # Align markers, since lane change began in between two game markers of left/right lane
    error_moves -= int(fs_neural / (2*fs_markers))

    # Detect jumps (large discontinuities within data)
    diffs = np.abs(np.diff(rawEEG[:, 0]))
    jump_threshold = 500 * np.median(diffs)
    jumps = np.where(diffs > jump_threshold)[0] + 1
    jumps = np.concatenate(([0], jumps, [n_samples]))  # segment boundaries

    # Remove trials at beginnig / end if not sufficient data
    clean_correct, clean_error = [], []
    for i in correct_moves:
        if i - before < 0 or i + after > n_samples:
            continue
        for start, end in zip(jumps[:-1], jumps[1:]):
            if (i - before) >= start and (i + after) <= end:
                clean_correct.append(i)
                break
    for i in error_moves:
        if i - before < 0 or i + after > n_samples:
            continue
        for start, end in zip(jumps[:-1], jumps[1:]):
            if (i - before) >= start and (i + after) <= end:
                clean_error.append(i)
                break
    
    # print(f"-> Removed {len(correct_moves)-len(clean_correct)} correct & {len(error_moves)-len(clean_error)} error trials (unfinished / noisy)")

    return np.array(clean_correct), np.array(clean_error)



def calculate_spectral_increase(groupedResults):
    """
    Compute mean and SEM percent power increase (Error vs General) across all ROIs.
    """

    sigma = 1
    error_all, general_all = [], []

    for roi in [1, 2, 3]:
        d = groupedResults[roi]

        # Get session-wide and error-wide spectrum betw. 4 and 200 HZ
        f_err = d['f_error']
        f_gen = d['f_general']
        mask_err = (f_err >= 4) & (f_err <= 200)
        mask_gen = (f_gen >= 4) & (f_gen <= 200)
        f_err_sel = f_err[mask_err]
        f_gen_sel = f_gen[mask_gen]

        # Align general to error frequencies (different resolutions)
        gen_interp = np.array([
            np.interp(f_err_sel, f_gen_sel, g[mask_gen])
            for g in d['generalSpectrum']
        ])
        err_sel = np.array([e[mask_err] for e in d['errorSpectrum']])

        # Smooth per electrode (since error trials are short and have noisy spectra)
        gen_smooth = np.array([gaussian_filter1d(g, sigma=sigma) for g in gen_interp])
        err_smooth = np.array([gaussian_filter1d(e, sigma=sigma) for e in err_sel])
        general_all.append(gen_smooth)
        error_all.append(err_smooth)

    # Combine electrodes
    error_all = np.vstack(error_all)
    general_all = np.vstack(general_all)

    # Percent increase and mean/SEM
    percent_increase = ((error_all - general_all) / general_all) * 100
    mean_inc = np.mean(percent_increase, axis=0)
    sem_inc  = np.std(percent_increase, axis=0) / np.sqrt(percent_increase.shape[0])
    
    # Statistical testing (different from 0)
    t_stat, p_values = ttest_1samp(percent_increase, 0, axis=0)
    sig_mask = p_values < 0.05

    return f_err_sel, mean_inc, sem_inc, sig_mask



def tval_spectrogram(timeSegCorrect, timeSegError):
    """
    Compute tvalued spectrogram of correct vs. erroneous trials 
    Given time segments of correct and error trials of a single channel [samples in time, trials]
    """
    # Calculate spectrograms of all correct and error trials
    f, t, spectrogram_correct = compute_spectrograms(timeSegCorrect)
    _, _, spectrogram_error = compute_spectrograms(timeSegError)
    time_indices, freq_indices = (t >= 0.6) & (t < 2.4), (f >= 4) & (f <= 200)
    spectrogram_correct = spectrogram_correct[freq_indices, :, :]
    spectrogram_correct = spectrogram_correct[:, time_indices, :]
    spectrogram_error = spectrogram_error[freq_indices, :, :]
    spectrogram_error = spectrogram_error[:, time_indices, :]
    
    # Smooth spectrograms and normalize them across trials
    spectrogram_correct = gaussian_smooth(spectrogram_correct)
    spectrogram_error = gaussian_smooth(spectrogram_error)
    all_spectrograms = np.concatenate([spectrogram_correct, spectrogram_error], axis=-1)  # (time, freq, trials)
    spectrogram_correct = zscore_spectrograms(spectrogram_correct, all_spectrograms)          
    spectrogram_error = zscore_spectrograms(spectrogram_error, all_spectrograms)
    
    # Transfer to t-value spectrogram
    t_values, p_values, significant_mask = compute_ttest(spectrogram_error, spectrogram_correct)
    
    return t_values, p_values, significant_mask, f[freq_indices], t[time_indices]
    

def compute_spectrograms(segments, window_size = 256, overlap = 0.95, fft_size = 5*1024):
    """
    Compute spectrograms for multiple trials of a single channel
    """
    # Segment is (trials, samples in time) of one particular channel in time domain 
    Sxx_list = []
    for segment in segments:
        f, t, Sxx = spectrogram(
            segment, fs=1024, 
            nperseg=window_size,   
            noverlap=int(window_size * overlap),  
            nfft=fft_size  
        )
        Sxx_list.append(Sxx)
    return np.array(f), np.array(t), np.stack(Sxx_list, axis=-1)

def gaussian_smooth(spectrogram, sigma_time=4, sigma_freq=25): 
    """
    Apply Gaussian smoothing to spectrogram along time and frequency axes
    """
    spectrogram_smoothed_time = gaussian_filter1d(spectrogram, sigma=sigma_time, axis=1)
    spectrogram_smoothed = gaussian_filter1d(spectrogram_smoothed_time, sigma=sigma_freq, axis=0)
    return spectrogram_smoothed

def zscore_spectrograms(spectrogram, all_spectrogram_data):
    """
    Z-score normalize spectrogram across all trials and time
    """
    mean_all = np.mean(all_spectrogram_data, axis=(1, 2), keepdims=True)  
    std_all = np.std(all_spectrogram_data, axis=(1, 2), keepdims=True)    
    return (spectrogram - mean_all) / std_all

def compute_ttest(spectrogram_correct, spectrogram_error, significance_threshold=0.05):
    """
    Compute t-test between correct and erroneous trials for each time-frequency point
    """
    # spectrogram must have shape [Freq, Time, trials]
    time_bins = spectrogram_correct.shape[1]
    freq_bins = spectrogram_correct.shape[0]
    t_values = np.zeros((freq_bins, time_bins))
    p_values = np.ones((freq_bins, time_bins))  
    
    for freq_idx in range(freq_bins):
        for time_idx in range(time_bins):
            success_data = spectrogram_correct[freq_idx, time_idx, :]
            fail_data = spectrogram_error[freq_idx, time_idx, :]
            t_stat, p_val = ttest_ind(success_data, fail_data, equal_var=False)
            t_values[freq_idx, time_idx] = t_stat
            p_values[freq_idx, time_idx] = p_val

    significant_mask = p_values < significance_threshold
    return t_values, p_values, significant_mask
    


def run_cv_lda(X, y, n_splits=5):
    """
    Run stratified 5-fold cross-validation using LDA.
    Returns mean balanced accuracy across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        clf = LDA(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        accs.append(balanced_accuracy_score(y[test_idx], y_pred))
    return np.mean(accs)


def decode_participants(errorResults, n_repeats=100, n_shuffles=1000):
    """
    LDA decoding of correct vs. error per participant.
    Returns mean accuracy, std, and binary significance flag (above 95th percentile).
    """
    mean_accs, std_accs, sig_flags = [], [], []
    for ppt, pdata in errorResults.items():
        
        # Get data for participant
        X_corr, X_err = pdata['featuresCorrect'], pdata['featuresError']
        if X_corr.size == 0 or X_err.size == 0:
            mean_accs.append(np.nan)
            std_accs.append(np.nan)
            sig_flags.append(False)
            continue

        # Combine correct/error trials
        X = np.vstack([X_corr, X_err])
        y = np.concatenate([np.zeros(len(X_corr)), np.ones(len(X_err))])

        # --- Real decoding ---
        real_accs = []
        print(f"\n{ppt}: ", end="", flush=True)
        for _ in range(n_repeats):
            real_accs.append(run_cv_lda(X, y))
            print(".", end="", flush=True)
        real_accs = np.array(real_accs) * 100
        real_mean, real_std = np.mean(real_accs), np.std(real_accs)

        # --- Null distribution (Monte carlo style label shuffling) ---
        null_accs = np.array([run_cv_lda(X, np.random.permutation(y))
                              for _ in range(n_shuffles)]) * 100
        null_95 = np.percentile(null_accs, 95)
        sig = real_mean > null_95

        mean_accs.append(real_mean)
        std_accs.append(real_std)
        sig_flags.append(sig)
        print(f"-> {real_mean:.1f}±{real_std:.1f}% | {'sig' if sig else 'n.s.'}")

    return np.array(mean_accs), np.array(std_accs), np.array(sig_flags)


def train_lda_cv(X, y, priors):
    """
    Run 10-fold cross-validated LDA with given priors.
    Returns the confusion matrix in % and the false positive rate (FPR).
    
    FPR = proportion of true 'correct' trials predicted as 'error'.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        clf = LDA(solver='lsqr', shrinkage='auto', priors=priors)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all, normalize='true') * 100
    fpr = cm[0, 1] / 100.0  # false positive rate (true correct → predicted error)
    return cm, fpr


def decode_confusion_single(errorResults, participant_id, target_fpr=0.05):
    """10-fold LDA decoding for one participant.
    Tests all priors (0.01–0.99) and picks the one whose FPR 
    is closest to target_fpr but not above it (rounded allowed)."""
    
    # Load data
    pdata = errorResults[participant_id]
    X_corr, X_err = pdata['featuresCorrect'], pdata['featuresError']
    X = np.vstack([X_corr, X_err])
    y = np.concatenate([np.zeros(len(X_corr)), np.ones(len(X_err))])

    priors = np.arange(0.01, 1.00, 0.01)
    best_diff, best_cm, best_fpr, best_prior = np.inf, None, None, None

    for p in priors:
        cm, fpr = train_lda_cv(X, y, priors=[p, 1 - p])

        # Only accept if FPR <= target (allow equal if rounded)
        if fpr <= target_fpr or round(fpr, 2) == round(target_fpr, 2):
            diff = abs(fpr - target_fpr)
            if diff < best_diff:
                best_diff, best_cm, best_fpr, best_prior = diff, cm, fpr, p

    # If all FPRs > target, just take the closest overall
    if best_cm is None:
        for p in priors:
            cm, fpr = train_lda_cv(X, y, priors=[p, 1 - p])
            diff = abs(fpr - target_fpr)
            if diff < best_diff:
                best_diff, best_cm, best_fpr, best_prior = diff, cm, fpr, p
 
    # Print final confusion matrix
    print(f"\n{participant_id} — Confusion matrix (%):")
    print("             Predicted")
    print("          Correct   Error")
    print(f"True Corr   {best_cm[0,0]:3.0f}%   {best_cm[0,1]:3.0f}%")
    print(f"True Err    {best_cm[1,0]:3.0f}%   {best_cm[1,1]:3.0f}%")
    print(f"Achieved FPR={best_fpr*100:.1f}%  |  "
          f"Priors [{best_prior:.2f}, {1-best_prior:.2f}]")

    return best_cm


def prepare_data(corr_data, err_data, min_trials):
    """Sample equal # of trials per participant and concatenate across channels."""
    corr_list, err_list = [], []
    for c, e in zip(corr_data, err_data):
        corr_idx = np.random.choice(c.shape[0], min_trials, replace=False)
        err_idx  = np.random.choice(e.shape[0], min_trials, replace=False)
        corr_list.append(c[corr_idx])
        err_list.append(e[err_idx])
    corr =np.concatenate(corr_list, axis=2)  # (trials, time, all_chans)
    err  = np.concatenate(err_list,  axis=2)
    return corr, err


def decode_windowed(corr, err, win=512, step=32, fs=1024, sig_test=False):
    """Perform time-resolved LDA decoding for one dataset."""
    
    # Define data/labels, timestamps, and storage
    time_len = corr.shape[1]
    X = np.concatenate([corr, err], axis=0)
    y = np.array([0]*corr.shape[0] + [1]*err.shape[0])
    if sig_test: np.random.shuffle(y)
    timestamps = np.array([(i + win) / fs for i in range(0, time_len - win + 1, step)])     
    accs = []

    # Loop through timesteps, extract features, and decode
    for i in range(0, time_len - win + 1, step):
        feat = np.mean(X[:, i:i+win, :], axis=1)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        fold_accs = []
        
        # Training and test split
        for train, test in skf.split(feat, y):
            clf = MLPClassifier(hidden_layer_sizes=(10,), alpha=30.0, max_iter=2000)
            clf.fit(feat[train], y[train])
            preds = clf.predict(feat[test])
            fold_accs.append(accuracy_score(y[test], preds))
        accs.append(np.mean(fold_accs))
        
    return timestamps, np.array(accs)


def temporal_decoding(groupedResults, n_runs=100, n_sig_runs=100):
    """Run decoding for all ROIs (1=insula, 2=prefrontal, 3=central-parietal).
    Includes 95th percentile significance threshold from shuffled-label decoding.
    """
    # Parameters
    fs, win, step = 1024, 512, 32
    region_labels = {1: 'Anterior Insula', 2: 'Prefrontal cortex', 3: 'Central-parietal cortex'}
    results = {}

    # Loop over 3 regions
    for region_key, region_name in region_labels.items():
        print(f"\nDecoding {region_name}:", end="")
        corr_data = groupedResults[region_key]['correctSegments']
        err_data  = groupedResults[region_key]['errorSegments']
        
        # Filter out electrodes with less then 10 trials per condition
        min_required_trials = 10
        corr_filtered, err_filtered = [], []
        for c, e in zip(corr_data, err_data):
            if c.shape[0] >= min_required_trials and e.shape[0] >= min_required_trials:
                corr_filtered.append(c)
                err_filtered.append(e)
        corr_data, err_data = corr_filtered, err_filtered

        # Equalize trial counts
        min_trials = min(min(c.shape[0] for c in corr_data),
                         min(e.shape[0] for e in err_data))

        # --- Normal decoding ---
        all_accs = []
        for run in range(n_runs):
            print(".", end="")
            corr, err = prepare_data(corr_data, err_data, min_trials)
            _, accs = decode_windowed(corr, err, win, step, fs, sig_test=False)
            all_accs.append(accs)
        all_accs = np.array(all_accs)
        mean_accs = np.mean(all_accs, axis=0)
        std_accs  = np.std(all_accs, axis=0)
        timestamps = np.array([(i + win) / fs for i in range(0, corr.shape[1] - win + 1, step)])  

        # --- Significance decoding ---
        if n_sig_runs > 0:
            print("\n-> sig. testing: ", end="")
            sig_accs = []
            for run in range(n_sig_runs):
                print(".", end="")
                corr, err = prepare_data(corr_data, err_data, min_trials)
                _, accs = decode_windowed(corr, err, win, step, fs, sig_test=True)
                sig_accs.append(accs)
            sig_accs = np.array(sig_accs)

            # Compute 95th percentile and significance mask
            sig_threshold = np.percentile(sig_accs, 95, axis=0)
            sig_bool = mean_accs > sig_threshold
        else:
            sig_threshold = np.full_like(mean_accs, np.nan)
            sig_bool = np.full_like(mean_accs, False, dtype=bool)

        results[region_name] = (timestamps, mean_accs, std_accs, sig_threshold, sig_bool)

    return results



def kde_vals(samples, coord_vals, bw=0.35):
    """Compute a Gaussian KDE along coord_vals, scaled by sample count."""
    samples = np.asarray(samples)
    if samples.size < 2 or np.allclose(samples, samples[0]):
        return np.zeros_like(coord_vals)
    kde = gaussian_kde(samples, bw_method=bw)
    return kde(coord_vals) * len(samples)


def compute_spatial_kdes(errorResults):
    """
    Compute normalized spatial KDEs (all, t>2, t>4) across anterior/posterior insula electrodes.
    """
    coord_dir = Path("Data/Electrode_coordinates")
    T2, T4 = 2, 4
    ROIS = (1, 4)   # anterior + posterior insula
    bw = 0.35
    all_list, t2_list, t4_list = [], [], []

    for ppt, vals in errorResults.items():
        f = coord_dir / f"coords_{ppt}.mat"
        if not f.exists():
            print(f"⚠️ Missing: {f}")
            continue

        mat = loadmat(f)
        coords = mat['elecmatrix']             # (n_channels, 3)
        names = [x[0][0] for x in mat['anatomy']]

        # recorded data
        chans = [str(x) for x in np.ravel(vals['channels'])]
        tvals = np.ravel(vals['tvalues']).astype(float)
        rois  = np.ravel(vals['roi']).astype(int)

        # map recorded channels to their t-values and ROI
        ch_to_t = {ch: t for ch, t in zip(chans, tvals)}
        ch_to_roi = {ch: r for ch, r in zip(chans, rois)}

        # cycle through all anatomical names
        for name, coord in zip(names, coords):
            if name in ch_to_roi and ch_to_roi[name] in ROIS:
                t = np.abs(ch_to_t[name])
                all_list.append(coord)
                if t > T2:
                    t2_list.append(coord)
                if t > T4:
                    t4_list.append(coord)

    # --- Convert to arrays ---
    all_mat = np.array(all_list, dtype=float).reshape(-1, 3) if all_list else np.zeros((0,3))
    t2_mat  = np.array(t2_list,  dtype=float).reshape(-1, 3) if t2_list  else np.zeros((0,3))
    t4_mat  = np.array(t4_list,  dtype=float).reshape(-1, 3) if t4_list  else np.zeros((0,3))

    # --- y-axis (anterior–posterior) ---
    dim = 1
    all_y = all_mat[:, dim] if len(all_mat) else np.array([])
    t2_y  = t2_mat[:, dim]  if len(t2_mat)  else np.array([])
    t4_y  = t4_mat[:, dim]  if len(t4_mat)  else np.array([])

    # --- Coordinate range ---
    if len(all_y) + len(t2_y) + len(t4_y) > 0:
        c_min = min([arr.min() for arr in [a for a in [all_y, t2_y, t4_y] if len(a)]])
        c_max = max([arr.max() for arr in [a for a in [all_y, t2_y, t4_y] if len(a)]])
        span = (c_max - c_min) if c_max > c_min else 1.0
        coord_vals = np.linspace(c_min - 0.3*span, c_max + 0.3*span, 500)
    else:
        coord_vals = np.linspace(0, 1, 500)

    # --- Compute KDEs and normalize ---
    kde_all = kde_vals(all_y, coord_vals, bw)
    norm = np.max(kde_all) if np.max(kde_all) > 0 else 1.0
    kde_all = kde_all / norm
    kde_t2  = kde_vals(t2_y, coord_vals, bw) / norm
    kde_t4  = kde_vals(t4_y, coord_vals, bw) / norm

    return coord_vals, kde_all, kde_t2, kde_t4


def top_corr_channels(hfa, lfa, labels, n_hfa=15, n_lfa=35):
    """
    Return indices of top HFA (positively correlated) and LFA (negatively correlated)
    channels with movement labels.
    """
    n = min(len(hfa), len(lfa), len(labels))
    hfa, lfa, labels = hfa[:n], lfa[-n:], labels[:n]

    corr_hfa = np.array([np.corrcoef(hfa[:, i], labels)[0, 1] for i in range(hfa.shape[1])])
    corr_lfa = np.array([np.corrcoef(lfa[:, i], labels)[0, 1] for i in range(lfa.shape[1])])

    hfa_idx = np.argsort(np.abs(corr_hfa))[::-1][:n_hfa]
    lfa_idx = np.argsort(np.abs(corr_lfa))[::-1][:n_lfa]

    return hfa_idx, lfa_idx

