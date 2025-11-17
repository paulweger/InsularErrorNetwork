import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.formula.api as smf
import matplotlib.lines as mlines
import matplotlib.colors as colors
from pathlib import Path




def plot_sig_electrodes(percent_sig_per_roi, total_sig_elecs, total_elecs, save_path='Figures/Fig3.pdf'):
    """Plot % of significant electrodes per ROI and save as PDF."""
    regions = ['Anterior Insula', 'Prefrontal cortex', 'Central-parietal cortex']
    colors = ['tab:blue', '#00B3B3', 'forestgreen']

    percent_sig_per_roi = np.array(percent_sig_per_roi)
    total_sig_elecs = np.array(total_sig_elecs)
    total_elecs = np.array(total_elecs)

    # Select relevant ROIs (adjust indices if needed)
    roi_idx = [1, 2, 3]
    percent_sigElecs = percent_sig_per_roi[roi_idx]
    num_sigElecs = total_sig_elecs[roi_idx]
    num_roiElecs = total_elecs[roi_idx]

    # Sort
    sort_idx = np.argsort(percent_sigElecs)
    regions = [regions[i] for i in sort_idx]
    colors = [colors[i] for i in sort_idx]
    percent_sigElecs = percent_sigElecs[sort_idx]
    num_sigElecs = num_sigElecs[sort_idx]
    num_roiElecs = num_roiElecs[sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 2))
    y = np.arange(len(regions))
    ax.barh(y, percent_sigElecs, color=colors, height=0.5)
    ax.set(yticks=y, yticklabels=regions, xlabel='% of significant channels', xlim=(0, 60))
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=12, length=0)
    ax.tick_params(axis='y', length=0)
    for s in ['left', 'right', 'top']: ax.spines[s].set_visible(False)

    # Labels
    for i, (p, s, t) in enumerate(zip(percent_sigElecs, num_sigElecs, num_roiElecs)):
        ax.text(p - 1, i, f'{s} / {t}', va='center', ha='right', color='white', fontsize=12)
    ax.set_xlabel('% of significant channels', fontsize=14)
    ax.set_yticklabels(regions, fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf')


def plot_roi_violin(groupedResults, dataframe_linMixedEffects, save_path='Plotting/Fig3/HFA_violin.pdf'):
    """
    Plot mean HFA (correct vs error) for significant channels per ROI using meanPowerCorrect / meanPowerError.
    Only significant electrodes (sig_idx) are plotted.
    """

    roi_map = {
        1: ('Anterior Insula', 'tab:blue'),
        2: ('Prefrontal', '#00B3B3'),
        3: ('Central-parietal', 'forestgreen')
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 3))
    x = [0, 0.7]
    jitter = 0.1

    for ax, roi in zip(axes, [1, 2, 3]):
        title, color = roi_map[roi]
        g = groupedResults[roi]

        # Use only significant electrodes
        corr_vals = g['meanPowerCorrect']
        err_vals  = g['meanPowerError']
        participant_idx = g['participant_idx']

        # Violin plots
        parts = ax.violinplot([corr_vals, err_vals], positions=x, showmeans=False,
                              showmedians=False, showextrema=False)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(['grey', color][i])
            body.set_alpha(0.7)
            body.set_zorder(2)

        # Jittered points
        corr_jit = np.random.uniform(-jitter, jitter, len(corr_vals))
        err_jit  = np.random.uniform(-jitter, jitter, len(err_vals))
        ax.scatter(x[0] + corr_jit, corr_vals, color='grey', alpha=0.7, zorder=3, clip_on=False)
        ax.scatter(x[1] + err_jit, err_vals, color=color, alpha=0.7, zorder=3, clip_on=False)

        # Spaghetti lines
        for i in range(len(corr_vals)):
            ax.plot([x[0] + corr_jit[i], x[1] + err_jit[i]],
                    [corr_vals[i], err_vals[i]],
                    color='lightgray', alpha=0.5, linewidth=2, zorder=1)

        # Median lines
        for i, vals in enumerate([corr_vals, err_vals]):
            median = np.median(vals)
            ax.plot([x[i] - 0.1, x[i] + 0.1], [median, median], 'k-', linewidth=3, zorder=4)
        print(f"-> Median HFA {title} - Correct: {np.median(corr_vals):.2f}, Error: {np.median(err_vals):.2f}")
        
        
        # --- Significance testing: Linear Mixed-Effects Model ---
        df_roi = dataframe_linMixedEffects[dataframe_linMixedEffects["roi"] == roi]
        try:
            model = smf.mixedlm("HFA ~ C(condition)",
                                df_roi,
                                groups=df_roi["participant"],
                                vc_formula={"electrode": "0 + C(electrode)"})
            result = model.fit()
            p_val = result.pvalues.get('C(condition)[T.error]', 1.0)
        except Exception as e:
            print(f"LMM failed for {roi_map[roi][0]}: {e}")
            p_val = 1.0
        
        # Plot significance line and star
        star = '***' if p_val <= 0.001 else '**' if p_val <= 0.01 else '*' if p_val <= 0.05 else ''
        if title == 'Anterior Insula': line_y = 2 * 1.05
        elif title == 'Prefrontal': line_y = 1.2 * 1.05
        else: line_y = 1.5 * 1.05
        ax.plot([x[0], x[1]], [line_y, line_y], 'k-', linewidth=1.5, zorder=5, clip_on=False)
        if star:
            ax.text(0.35, line_y * 0.97, star, ha='center', va='bottom', fontsize=14, zorder=6)

        # Aesthetics
        ax.set_xticks(x)
        ax.set_xticklabels(['Correct', 'Error'], fontsize=14)
        ax.tick_params(axis='y', labelsize=12, length=0)
        for s in ['right', 'top']: ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=16, pad=20)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)
        if title == 'Anterior Insula':
            ax.set_ylim([-0.9, 2])
            ax.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])
            ax.set_yticklabels(['-0.5', '0', '0.5', '1', '1.5', '2'], fontsize=12)
        elif title == 'Prefrontal':
            ax.set_ylim([-0.7, 1.2])
            ax.set_yticks([-0.5, 0, 0.5, 1])
            ax.set_yticklabels(['-0.5', '0', '0.5', '1'], fontsize = 12)
        elif title == 'Central-parietal':
            ax.set_ylim([-1, 1.5])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5])
            ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1', '1.5'], fontsize=12)
        

    axes[0].set_ylabel('Average HFA (n.u.)', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    

def plot_grouped_activity(groupedResults, mode='trialwise', save_path='Plotting/Fig3/Pooled_hfa.pdf'):
    """
    Plot activity for ROIs 1=aINS, 2=PFC, 3=Centroparietal.

    Parameters
    ----------
    groupedResults : dict
        Output from group_error_results_by_roi.
    mode : str
        'trialwise' or 'electrodewise'.
    """
    roi_labels = {1: 'Anterior Insula', 2: 'Prefrontal cortex', 3: 'Central-parietal cortex'}
    colors = {1: 'tab:blue', 2: '#00B3B3', 3: 'forestgreen'}

    # Determine number of samples from first available dataset
    for roi in [1, 2, 3]:
        key = f'{mode}_correct'
        if key in groupedResults[roi] and len(groupedResults[roi][key]) > 0:
            n_samples = groupedResults[roi][key].shape[1]
            break
    else:
        raise ValueError(f"No {mode}_correct data found in groupedResults.")

    start_time = -1
    time_seconds = np.linspace(start_time, start_time + n_samples / 1024, n_samples)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 5.5), sharex=True)
    fig.suptitle("Pooled error-responsive electrodes", fontsize=16, y=0.95)

    for idx, roi in enumerate([1, 2, 3]):
        r = groupedResults[roi]
        key_corr = f'{mode}_correct'
        key_err = f'{mode}_error'
        if key_corr not in r or key_err not in r:
            continue

        correct_data = np.array(r[key_corr])
        error_data = np.array(r[key_err])
        if correct_data.size == 0 or error_data.size == 0:
            continue

        # Mean and SEM
        correct_mean = np.mean(correct_data, axis=0)
        correct_sem = np.std(correct_data, axis=0) / np.sqrt(correct_data.shape[0])
        error_mean = np.mean(error_data, axis=0)
        error_sem = np.std(error_data, axis=0) / np.sqrt(error_data.shape[0])

        # T-test across trials/electrodes
        t_stat, p_values = ttest_ind(error_data, correct_data, axis=0)
        sig_mask = p_values < 0.05
        
        # Print some values
        print(f"\n-> ", roi_labels[roi])
        try: print(f"Error peak: {np.max(error_mean):.2f} at {time_seconds[np.argmax(error_mean)]:.3f}s")
        except: print("Error peak: n/a")
        try: print(f"Error return to 0: {time_seconds[np.where(error_mean[1024:] <= 0)[0][0] + 1024]:.3f}s")
        except: print("Error return to 0: n/a")
        try: print(f"Correct minima: {np.min(correct_mean):.3f} at {time_seconds[np.argmin(correct_mean)]:.3f}s")
        except: print("Correct minima: n/a")
        try: print(f"Correct return to 0: {time_seconds[np.where(error_mean[2*1024:] <= 0)[0][0] + 1024]:.3f}s")
        except: print("Correct return to 0: n/a")
        try:
            sig_diff = np.diff(np.concatenate([[0], sig_mask.astype(int), [0]]))
            starts, ends = np.where(sig_diff == 1)[0], np.where(sig_diff == -1)[0]
            sig_intervals = [(np.round(time_seconds[s],3), np.round(time_seconds[e-1],3)) for s, e in zip(starts, ends)]
            print(f"Significance: {sig_intervals}")
        except:
            print("Significance: n/a")

        # Plot
        ax = axes[idx]
        error_color = colors[roi]
        ax.plot(time_seconds, correct_mean, label='Correct', color='grey', linewidth=2)
        ax.fill_between(time_seconds, correct_mean - correct_sem, correct_mean + correct_sem,
                        color='grey', alpha=0.2)
        ax.plot(time_seconds, error_mean, label='Error', color=error_color, linewidth=2)
        ax.fill_between(time_seconds, error_mean - error_sem, error_mean + error_sem,
                        color=error_color, alpha=0.2)
        ax.axvline(0, linestyle='--', color='grey', linewidth=1.5, zorder=0)

        # Significance bar + dots
        if np.any(sig_mask):
            y_bar = min(np.min(correct_mean - correct_sem), np.min(error_mean - error_sem)) - 0.05
            minmin = y_bar - 0.02
            maxmax = max(np.max(correct_mean + correct_sem), np.max(error_mean + error_sem)) + 0.05
            ax.scatter(time_seconds[sig_mask], np.full(np.sum(sig_mask), y_bar + 0.01),
                       color='black', s=0.5, marker='o', zorder=5, linewidth=0.5)

        # Titles & labels
        ax.text(0.5, 0.9, roi_labels[roi], transform=ax.transAxes, fontsize=14, va='bottom', ha='center')
        if roi == 1: y_error, y_correct = 0.06, -0.165
        elif roi == 2: y_error, y_correct = 0.065, -0.14
        elif roi == 3: y_error, y_correct = 0.12, -0.155
        ax.text(2.59, y_correct, 'Correct', color='grey', fontsize=10, fontweight=550)
        ax.text(2.705, y_error, 'Error', color=error_color, fontsize=10, fontweight=550)

        # Axes formatting
        ax.set_xlim([-0.3, time_seconds[-1]])
        ax.set_ylim([minmin, maxmax])
        ax.set_yticks([0, 0.5])
        ax.set_yticklabels(["0", "0.5"], fontsize=11)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=4 if idx < len(axes) - 1 else 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=12)

    # Shared labels
    fig.text(0.03, 0.5, 'High-frequency activity (n.u.)', va='center', rotation='vertical', fontsize=13)
    axes[-1].set_xlabel("Time (s)", fontsize=13)
    axes[-1].set_xticks([0, 1, 2, 3])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')



def plot_error_hfa(groupedResults, mode = 'trialwise', save_path='Plotting/Fig3/timing_hfa.pdf'):
    """
    Plot trial-wise error HFA for 3 ROIs and save the figure as PDF.
    Expects keys 1, 2, 3 in groupedResults for anterior insula, prefrontal, central-parietal.
    """
    # Time vector
    n_samples = groupedResults[1]['trialwise_error'].shape[1]
    time = np.linspace(-1, 3, n_samples)
    mask = (time >= -0.3) & (time <= 0.75)
    t_plot = time[mask]
    
    roi_order = [1, 2, 3]
    labels = ['Anterior insula', 'Prefrontal cortex', 'Central-parietal cortex']
    colors = ['tab:blue', '#00B3B3', 'forestgreen']
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    for roi, label, color in zip(roi_order, labels, colors):
        if mode == 'trialwise': data = groupedResults[roi]['trialwise_error'][:, mask]
        else: data = groupedResults[roi]['electrodewise_error'][:, mask]
        mean_vals = np.nanmean(data, axis=0)
        sem_vals  = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
        
        ax.plot(t_plot, mean_vals, color=color, linewidth=2, label=label)
        ax.fill_between(t_plot, mean_vals - sem_vals, mean_vals + sem_vals, color=color, alpha=0.2)
        
        peak_idx = np.argmax(mean_vals)
        ax.scatter(t_plot[peak_idx], mean_vals[peak_idx], color=color, s=50, zorder=10)
        print(f"-> {label}: trial-peak at {t_plot[peak_idx]:.3f}s with {mean_vals[peak_idx]:.2f} n.u.")
    
    ax.axvline(0, linestyle='--', color='grey', linewidth=1.5, zorder=0)
    
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('HFA (n.u.)', fontsize=14)
    ax.set_title('Timing of Error-related potentials', fontsize=16)
    ax.tick_params(axis='both', length=0, labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Axis
    ax.set_ylim([0, 1.1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=12)
    ax.set_xlim([-0.05, 0.55])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(['0', '100', '200', '300', '400', '500'], fontsize=12)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    
def plot_peak_latencies(groupedResults, save_path='Plotting/Fig3/peaklatency_boxplot.pdf'):
    """
    Calculate peak latency for each electrode in groupedResults['electrodewise_error']
    between 0-1s (samples 1024:2048), and plot as a horizontal boxplot with Mann-Whitney significance.
    """

    fs = 1024  # samples per second
    start_sample, end_sample = 1024, 2048  # 0-1s

    # Map ROIs to labels and colors
    roi_order = [1, 2, 3]
    labels = ['Anterior insula', 'Prefrontal cortex', 'Central-parietal cortex']
    colors = ['tab:blue', '#00B3B3', 'forestgreen']

    # Compute peak latencies per electrode
    latencies_by_region = {}
    for roi, label in zip(roi_order, labels):
        data = groupedResults[roi]['electrodewise_error']  # shape: (n_electrodes, n_time)
        peaks = np.argmax(data[:, start_sample:end_sample], axis=1)  # index within 0-1024
        latencies_by_region[label] = peaks / fs  # convert to seconds

    # Sort by median
    medians = {label: np.median(latencies_by_region[label]) for label in labels}
    sorted_labels = sorted(labels, key=lambda x: medians[x], reverse=True)  # descending median
    sorted_colors = [colors[labels.index(l)] for l in sorted_labels]
    data_box = [latencies_by_region[l] for l in sorted_labels]
    print("Median latencies per region:")
    for label in labels:
        print(f"-> {label}: {np.round(1000*medians[label],0)} ms")
    

    # Plot
    plt.figure(figsize=(4.5, 3))
    box = plt.boxplot(data_box, vert=False, patch_artist=True, widths=0.3,
                      boxprops=dict(linewidth=0),
                      whiskerprops=dict(color='grey'),
                      capprops=dict(linewidth=0, color='grey'),
                      medianprops=dict(color='black'), whis=[0, 100])  # no outliers

    for patch, color in zip(box['boxes'], sorted_colors):
        patch.set_facecolor(color)

    # Significance testing
    x_base = min(map(np.min, data_box))
    offset = (max(map(np.max, data_box)) - x_base) * 0.075

    print("\nSignificance:")
    for i, (a, b) in enumerate(itertools.combinations(range(len(data_box)), 2)):
        p = mannwhitneyu(data_box[a], data_box[b], alternative='two-sided').pvalue
        print(f"-> {sorted_labels[a]} vs {sorted_labels[b]}: p = {p:.3g}")
        if p >= 0.05: continue
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        y_min, y_max = min(a+1, b+1), max(a+1, b+1)
        x = x_base - offset * (i + 1) * 1.5
        plt.plot([x, x], [y_min, y_max], color='black', lw=1, clip_on=False)
        tick_length = offset * 0.2
        plt.plot([x, x + tick_length], [y_min, y_min], color='black', lw=1, clip_on=False)
        plt.plot([x, x + tick_length], [y_max, y_max], color='black', lw=1, clip_on=False)
        plt.text(x - tick_length, (y_min + y_max) / 2, star, ha='center', va='center',
                 fontsize=14, clip_on=False, rotation=90)

    # Clean look
    plt.title('Electrode-wise peak latency', fontsize=16, x=0.48)
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.xlim(0, 1)
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.yticks([])  # remove y-axis
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '200', '400', '600', '800', '1000'], fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis='both', length=0, labelsize=12)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()



def plot_power_change(freqs, spectral_mean, spectral_sem, sig_mask):
    """
    Plot mean ± SEM percent power change (Error vs Correct) across electrodes
    and save to Plotting/Fig3/spectral_change.pdf.
    """
    # Plot
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(freqs, spectral_mean, color='firebrick')
    ax.fill_between(freqs, spectral_mean - spectral_sem, spectral_mean + spectral_sem, color='firebrick', alpha=0.3)

    # Significance line
    y_sig = -10
    sig_freqs = freqs[sig_mask]
    for f in sig_freqs:
        ax.plot([f - 2, f + 2], [y_sig, y_sig], color='black', linewidth=1.5)
    starts = np.where(np.diff(np.concatenate(([0], sig_mask.astype(int), [0]))) == 1)[0]
    ends   = np.where(np.diff(np.concatenate(([0], sig_mask.astype(int), [0]))) == -1)[0] - 1
    for s, e in zip(starts, ends):
        print(f"Significant from {freqs[s]:.0f} Hz to {freqs[e]:.0f} Hz")
    print("Spectral peak: ", np.round(np.max(spectral_mean),1), "% at", freqs[np.argmax(spectral_mean)], "Hz")

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.5)

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xticks([4, 12, 30, 60, 100, 200])
    ax.set_xticklabels([4, 12, 30, 60, 100, 200], fontsize=12)
    ax.set_ylim(-11, 30)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_yticklabels([-10, 0, 10, 20, 30], fontsize=12)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    ax.set_ylabel('Power change (%)', fontsize=14)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_title('Frequency profile', fontsize=16, pad=10)
    plt.tight_layout()

    # Save
    save_dir = os.path.join("Plotting", "Fig3")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "spectral_change.pdf"), bbox_inches='tight')
    plt.close(fig)



def plot_tval_spectrogram(t_values, p_values, significant_mask, f, t):
    """
    Plot t-valued spectrogram of a selected electrode.
    Shows t-values (Error vs Correct) with significance contour overlay.
    """

    # --- Parameters ---
    shifted_time = t - 1
    max_t_value = 5  # Better for visualization in RX7, normally: np.ceil(np.max(np.abs(t_values)))
    freq_label_indices = [4, 30, 60, 100, 200]

    # --- Figure setup ---s
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # --- Main image ---
    img = ax.imshow(
        t_values, aspect='auto', cmap='RdBu_r', origin='lower',
        alpha=1,
        extent=[shifted_time[0], shifted_time[-1], f[0], f[-1]],
        vmin=-max_t_value, vmax=max_t_value
    )

    # --- Significance contour & masked overlay ---
    ax.contour(
        significant_mask, levels=[0.5], colors='k', linewidths=2,
        extent=[shifted_time[0], shifted_time[-1], f[0], f[-1]]
    )

    ax.imshow(
        np.ma.masked_where(~significant_mask, t_values),
        aspect='auto', cmap='RdBu_r', origin='lower', alpha=0.6,
        extent=[shifted_time[0], shifted_time[-1], f[0], f[-1]],
        vmin=-max_t_value, vmax=max_t_value
    )

    # --- Time zero line ---
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1.5)

    # --- Labels, ticks, style ---
    ax.set_title('Time-frequency distribution', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'], fontsize=12)
    ax.set_yticks(freq_label_indices)
    ax.set_yticklabels(freq_label_indices, fontsize=12)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Subtle colorbar ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    cbar.set_label('T-value', fontsize=12)

    plt.tight_layout()
    save_dir = os.path.join("Plotting", "Fig3")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "spectrogram_tval.pdf"), bbox_inches='tight')
    plt.close(fig)

    

def plot_decoding_results(mean_accs, std_accs, sig_flags, save_path='Plotting/Fig5/errorDecoding.pdf'):
    """
    Plot participant decoding accuracies with error bars and significance stars.
    """
    n_participants = len(mean_accs)
    participant_names = [f'P{i:02d}' for i in range(1, n_participants + 1)]
    bar_width, bar_spacing = 0.5, 0.8

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x_pos = np.arange(n_participants) * bar_spacing

    # Bars and error bars
    bars = ax.bar(x_pos, mean_accs, width=bar_width,
                  color='tab:blue', edgecolor='none', zorder=2)
    ax.errorbar(x_pos, mean_accs, yerr=std_accs,
                fmt='none', ecolor='lightgrey', capsize=0,
                linewidth=2, zorder=1)

    # Axes and style
    ax.set_xticks(x_pos)
    ax.set_xticklabels(participant_names, fontsize=14)
    ax.set_ylabel('Balanced accuracy (%)', fontsize=14)
    ax.set_ylim(45, 95)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title('Offline error-detection accuracies', fontsize=16, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

    # Chance level and baseline
    left_bar, right_bar = x_pos[0] - bar_width / 2, x_pos[-1] + bar_width / 2
    ax.set_xlim(left_bar, right_bar + 0.2)
    xmin_data, xmax_data = ax.get_xlim()
    xmax_frac = (right_bar - xmin_data) / (xmax_data - xmin_data)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1.5,
               zorder=1, xmin=0, xmax=xmax_frac)
    ax.add_line(mlines.Line2D([left_bar, right_bar], [45, 45],
                              color='black', linewidth=1.0, zorder=3))

    # Label percentage and significance stars
    sigstar_height = 92.5
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h - 4,
                    f'{h:.0f}%', va='bottom', ha='center', fontsize=12, color='white')
            if sig_flags[i]:
                ax.text(bar.get_x() + bar.get_width() / 2, sigstar_height,
                        '*', ha='center', va='bottom', fontsize=14,
                        color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')



def plot_confusion_matrix(cm, title):
    """Plot a 2x2 confusion matrix (in %) and save as PDF."""
    
    # --- Setup figure and color normalization ---
    norm = colors.Normalize(vmin=0, vmax=np.max(cm))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest', cmap='Blues', norm=norm)

    # --- Add the percentage values inside each cell ---
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            # Highlight correct classifications with color and bold text
            if (i, j) == (0, 0):
                color, weight = 'tab:orange', 'bold'
            elif (i, j) == (1, 1):
                color, weight = 'firebrick', 'bold'
            else:
                color, weight = 'black', 'normal'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=color, fontsize=22, fontweight=weight)

    # --- Axis labels and ticks ---
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Correct", "Error"], fontsize=18)
    yt = ax.set_yticklabels(["Correct", "Error"], fontsize=18, rotation=90)
    for y in yt:
        y.set_va('center')
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("True", fontsize=20)
    ax.set_title(title, fontsize=20)

    # --- Clean up plot style ---
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='both', length=0)

    # --- Determine filename and save ---
    os.makedirs("Plotting/Fig5", exist_ok=True)
    if "neutral" in title.lower():
        filename = "confusion_neutral.pdf"
    elif "conservative" in title.lower():
        filename = "confusion_conservative.pdf"
    else:
        filename = f"confusion_{title.lower().replace(' ', '_')}.pdf"

    plt.tight_layout()
    plt.savefig(f"Plotting/Fig5/{filename}", format='pdf', bbox_inches='tight')


def plot_temporal_decoding(roiDecoding, save_path='Plotting/Fig5/temp_decoding.pdf'):
    """Plot temporal decoding results for all ROIs in one figure."""

    # Extract data from results
    timestamps_d, mean_accs_d, std_accs_d, _, sig_mask_d = roiDecoding['Anterior Insula']           # d = deep
    timestamps_s, mean_accs_s, std_accs_s, _, sig_mask_s = roiDecoding['Prefrontal cortex']         # s = superficial
    timestamps_m, mean_accs_m, std_accs_m, _, sig_mask_m = roiDecoding['Central-parietal cortex']   # m = motor

    # --- Plot setup ---
    plt.figure(figsize=(5.5, 4))
    ax = plt.gca()

    # Central-parietal
    plt.plot(timestamps_m, mean_accs_m, label='Central-parietal cortex',
             color='forestgreen', linewidth=3, zorder=2)
    plt.fill_between(timestamps_m, mean_accs_m - std_accs_m, mean_accs_m + std_accs_m,
                     color='forestgreen', alpha=0.1, zorder=2)

    # Anterior Insula
    plt.plot(timestamps_d, mean_accs_d, label='Anterior Insula',
             color='tab:blue', linewidth=3, zorder=2)
    plt.fill_between(timestamps_d, mean_accs_d - std_accs_d, mean_accs_d + std_accs_d,
                     color='tab:blue', alpha=0.1, zorder=2)

    # Prefrontal cortex
    plt.plot(timestamps_s, mean_accs_s, label='Prefrontal cortex',
             color='#00B3B3', linewidth=3, zorder=2)
    plt.fill_between(timestamps_s, mean_accs_s - std_accs_s, mean_accs_s + std_accs_s,
                     color='#00B3B3', alpha=0.1, zorder=2)

    # --- Formatting ---
    plt.axhline(0.5, color='lightgrey', linestyle='--', linewidth=1.5, zorder=1)
    plt.axvline(1, color='grey', linestyle='--', linewidth=1.5, zorder=1)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0.42, 1)
    plt.xlim(0.5, 2)
    plt.xticks([0.5, 1, 1.5, 2], ['-0.5', '0', '0.5', '1'], fontsize=12)
    ax.tick_params(axis='x', which='both', length=0)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1],
               ['50', '60', '70', '80', '90', '100'], fontsize=12)
    ax.tick_params(axis='y', which='both', length=0)
    plt.title("Temporal decoding", fontsize=16, pad=10)
    plt.box(on=True)

    # --- Aesthetic tweaks ---
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)

    # --- Significance lines or dots ---
    y_sig_level = 0.45
    plot_sig = True
    dot_size = 10

    if plot_sig:
        if np.any(sig_mask_m):
            plt.hlines(y=y_sig_level, xmin=timestamps_m[sig_mask_m][0],
                       xmax=timestamps_m[sig_mask_m][-1],
                       colors='forestgreen', linewidth=2, alpha=1, zorder=3)
        if np.any(sig_mask_d):
            plt.hlines(y=y_sig_level - 0.01, xmin=timestamps_d[sig_mask_d][0],
                       xmax=timestamps_d[sig_mask_d][-1],
                       colors='tab:blue', linewidth=2, alpha=1, zorder=3)
        if np.any(sig_mask_s):
            plt.hlines(y=y_sig_level - 0.02, xmin=timestamps_s[sig_mask_s][0],
                       xmax=timestamps_s[sig_mask_s][-1],
                       colors='#00B3B3', linewidth=2, alpha=1, zorder=3)
    else:
        plt.scatter(timestamps_m[sig_mask_m], [y_sig_level]*np.sum(sig_mask_m),
                    color='forestgreen', s=dot_size, marker='.', alpha=0.8, zorder=3)
        plt.scatter(timestamps_d[sig_mask_d], [y_sig_level-0.01]*np.sum(sig_mask_d),
                    color='tab:blue', s=dot_size, marker='.', alpha=0.8, zorder=3)
        plt.scatter(timestamps_s[sig_mask_s], [y_sig_level-0.02]*np.sum(sig_mask_s),
                    color='#00B3B3', s=dot_size, marker='.', alpha=0.8, zorder=3)

    # Print summary of results (peak timing, significance, ...)
    for name, t, acc, sig in [
        ("Central-parietal", timestamps_m, mean_accs_m, sig_mask_m),
        ("Anterior Insula", timestamps_d, mean_accs_d, sig_mask_d),
        ("Prefrontal", timestamps_s, mean_accs_s, sig_mask_s)
    ]:
        peak_idx = np.argmax(acc[t >= 1])
        print(f"{name}: peak={acc[t >= 1][peak_idx]:.2f} at {t[t >= 1][peak_idx]:.2f}s", end="")
        if np.any(sig): print(f", sig {t[sig][0]:.2f}-{t[sig][-1]:.2f}s")
        else: print(", no sig.")

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    
def plot_insula_kde(coord_vals, kde_all, kde_t2, kde_t4):
    """Plot anterior–posterior KDE distributions for all, t>2, and t>4 electrodes."""

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.fill_between(coord_vals, 0, kde_all, color='lightgrey', alpha=0.4, zorder=1, edgecolor='none')
    ax.fill_between(coord_vals, 0, kde_t2, color='tab:blue', alpha=0.7, zorder=2, edgecolor='none')
    ax.fill_between(coord_vals, 0, kde_t4, color='firebrick', alpha=0.8, zorder=3, edgecolor='none')

    # --- Cosmetics ---
    ax.set_xlabel('Anterior–Posterior coordinate (mm)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout()

    # --- Save figure ---
    out = Path(f'Plotting/Fig4/insula_error_distribution.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()



def plot_motor_heatmap(hfa, lfa, moveprob, start_time=64.5, duration=20):
    """
    Plot binned HFA (top) + LFA (bottom) heatmap with movement probability overlay.
    Saves as 'Plotting/Fig2/predictions.pdf'.
    """
    fs = 1024
    bin_size = 128
    out_path = Path("Plotting/Fig2/predictions.pdf")

    # --- Bin only neural data ---
    n_bins = min(hfa.shape[0], lfa.shape[0]) // bin_size
    hfa_bin = hfa[:n_bins * bin_size].reshape(n_bins, bin_size, -1).mean(axis=1)
    lfa_bin = lfa[:n_bins * bin_size].reshape(n_bins, bin_size, -1).mean(axis=1)
    time_axis = np.arange(n_bins) * (bin_size / fs)

    # --- Combine (HFA top, LFA bottom) and normalize ---
    data = np.vstack([lfa_bin.T, hfa_bin.T])  # flip order → HFA on top
    data = np.clip(data, -3, 5)

    # --- Time window ---
    x1, x2 = start_time, start_time + duration
    mask = (time_axis >= x1) & (time_axis <= x2)
    data = data[:, mask]
    time_axis = time_axis[mask]
    time_rel = time_axis - start_time  # relative 0–20s for x-axis

    # Align full-resolution moveprob with window
    move_t = np.arange(len(moveprob)) / fs
    mask_full = (move_t >= x1) & (move_t <= x2)
    moveprob = moveprob[mask_full]
    move_t_rel = move_t[mask_full] - start_time

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.imshow(data, aspect='auto', cmap='turbo',
               extent=[time_rel[0], time_rel[-1], 0, data.shape[0]],
               vmin=-3, vmax=5, origin='lower')

    ax1.set_title("Real-time decoder predictions", fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.set_ylabel("Features", fontsize=14)
    ax1.set_xlim(0, duration)
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_xticklabels([0, 5, 10, 15, 20], fontsize=12)
    ax1.set_yticks([hfa_bin.shape[1] // 2 + lfa_bin.shape[1], lfa_bin.shape[1] // 2])
    ax1.set_yticklabels(['HFA', 'LFA'], rotation=90, va='center', fontsize=12)
    ax1.tick_params(axis='x', length=0)
    ax1.tick_params(axis='y', length=0)
    for s in ['top', 'right', 'left', 'bottom']:
        ax1.spines[s].set_visible(False)

    # --- Overlay movement probability (unbinned) ---
    ax2 = ax1.twinx()
    moveprob_plot = 0.98 * moveprob + 0.01
    ax2.plot(move_t_rel, moveprob_plot, 'w-', lw=2.5, alpha=0.9)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Movement probability (%)", fontsize=14, rotation=270, labelpad=20)
    ax2.set_yticks([0.03, 0.5, 1])
    ax2.set_yticklabels(['0', '50', '100'], fontsize=12)
    ax2.tick_params(axis='y', length=0)
    for s in ['top', 'right', 'left', 'bottom']:
        ax2.spines[s].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
