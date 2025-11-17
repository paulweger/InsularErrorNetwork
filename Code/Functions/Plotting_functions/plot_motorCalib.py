import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path


def plot_motor_power(arr: np.ndarray, fs: int, ytick_vals: list, title: str, save_path: str = None):
    """
    Plot movement-onset power for motor-relevant channels.
    """
    mean_line = (arr.mean(1) - arr.mean(1).min()) / arr.mean(1).ptp() * (arr.shape[1]-1)

    plt.figure(figsize=(6,4))
    plt.imshow(arr.T, aspect='auto', cmap='turbo', vmin=-1, vmax=1)
    plt.xticks([0, fs, 2*fs, 3*fs, 4*fs-1], ['-2','-1','0','1','2'], size=12)
    plt.yticks(ytick_vals, ytick_vals, size=12)
    plt.axvline(2*fs, color='grey', lw=1.5, ls='--')
    plt.gca().invert_yaxis()
    plt.plot(mean_line, color='white', lw=2)
    for s in ['top','right','left','bottom']: plt.gca().spines[s].set_visible(False)
    plt.tick_params(axis='both', which='both', length=0)

    cbar = plt.colorbar(shrink=0.5)
    cbar.outline.set_visible(False)
    cbar.set_label("Power (n.u.)", size=13)
    cbar.set_ticks([-1,0,1])
    cbar.ax.tick_params(length=0)
    cbar.ax.set_yticklabels([-1,0,1], size=12)

    plt.ylabel("Motor relevant channels", fontsize=14)
    plt.xlabel("Time (s)", fontsize=14)
    plt.title(title, fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else: plt.show()


def plot_regional_activity(regional_hfa, fs=1024, save_path = None):
    """
    Plot mean HFA time-course per brain region across trials,
    Input: regional_hfa[region] = array of shape [trials, time]
    """
    
    # Regions to plot and their colors / sig marker height
    areas = ['Precentral gyrus', 'Postcentral gyrus', 'Posterior Insula']
    colors = {'Precentral gyrus':'forestgreen',
              'Postcentral gyrus':'tab:blue',
              'Posterior Insula':'#00B3B3'}
    sig_y  = {'Precentral gyrus':-0.19,
              'Postcentral gyrus':-0.21,
              'Posterior Insula':-0.23}

    # Determine time axis from first available region
    for a in areas:
        if a in regional_hfa and regional_hfa[a].size > 0:
            tlen = regional_hfa[a].shape[1]
            break
    time = np.linspace(-2, 2, tlen)

    plt.figure(figsize=(6,4))
    print("\nRegional HFA peak latencies:")
    for area in areas:
        data = regional_hfa.get(area, None)
        if data is None or data.size == 0: continue

        # Mean and SEM across trials
        mean = data.mean(axis=0)
        sem  = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        plt.plot(time, mean, color=colors[area], label=area, linewidth=2)
        plt.fill_between(time, mean - sem, mean + sem, color=colors[area], alpha=0.2)

        # Print latency of peak 
        peak_time = (np.argmax(mean[:int(3*fs)]) - 2*fs) / fs * 1000
        print(area, ':', np.round(peak_time,1), 'ms')

        # One-sample t-test against 0 at each time point
        _, p = ttest_1samp(data, 0, axis=0)
        sig = p < 0.05
        plt.scatter(time[sig], [sig_y[area]] * sig.sum(),
                    color=colors[area], s=0.5, zorder=5)

    # Make plot nice
    plt.axvline(0, linestyle='--', color='grey', linewidth=1.5)
    plt.xlim([-1, 2])
    plt.ylim([-0.25, 1])
    plt.xticks([-1, 0, 1, 2], size=12)
    plt.yticks([0, 0.5, 1], size=12)
    plt.yticklabels(['0', '0.5', '1'], size=12)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("High frequency activity (n.u.)", fontsize=14)
    plt.title("Timing of regional activity", fontsize=16)
    plt.tick_params(axis='both', which='both', length=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, loc='upper right', fontsize=12, handlelength=1, handletextpad=0.4)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else: plt.show()
    


def plot_incremental_training(decoding_results, save_path = None):
    """
    Plot incremental BCI training results.
    """

    # === Sort participants by final accuracy ===
    accuracies = {k: v['accuracy'][-1] for k, v in decoding_results.items()}
    sorted_keys = sorted(accuracies, key=accuracies.get)
    n = len(sorted_keys)

    # === Generate consistent color gradient ===
    cmap = mcolors.LinearSegmentedColormap.from_list("green_to_blue", ['#006400', '#98FB98', '#66CCFF', '#003366'])
    colors = [cmap(i / (n - 1)) for i in range(n)]
    participant_colors = {k: colors[i] for i, k in enumerate(sorted_keys)}

    # === Extract sorted data ===
    top_acc_values = np.array([accuracies[k] for k in sorted_keys])
    top_acc_std = np.array([decoding_results[k]['accuracy_std'][-1] for k in sorted_keys])

    # === Setup figure ===
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)

    # === Left panel: Accuracy over time ===
    ax0 = fig.add_subplot(gs[0])
    max_x_val, xt, xtl = 0, None, None

    for participant, data in decoding_results.items():
        x_vals, avgAcc, stdAcc = data['x_vals'], data['accuracy'], data['accuracy_std']
        if x_vals[-1] >= max_x_val:
            max_x_val, xt, xtl = x_vals[-1], data['x_ticks'], data['x_tick_labels']
        color = participant_colors[participant]
        ax0.plot(x_vals, avgAcc, color=color, linewidth=2, label=participant, zorder=3)
        # Uncomment to show uncertainty shading:
        # ax0.fill_between(x_vals, avgAcc - stdAcc, avgAcc + stdAcc, color=color, alpha=0.2, zorder=2)

    # === Styling ===
    ax0.set_xlabel("Training data (min)", fontsize=14)
    ax0.set_ylabel("Accuracy (%)", fontsize=14)
    ax0.set_title("Incremental training", fontsize=16)
    ax0.set_xticks(xt); ax0.set_xticklabels(xtl, fontsize=12)
    ax0.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax0.set_yticklabels(['50', '60', '70', '80', '90'], fontsize=12)
    ax0.set_xlim([0, 3 * 60]); ax0.set_ylim([0.488, 0.9])
    ax0.axhline(0.5, color='grey', linestyle='--', linewidth=1.5)
    ax0.tick_params(axis='both', which='both', length=0)
    ax0.spines['top'].set_visible(False); ax0.spines['right'].set_visible(False)
    # ax0.legend(frameon=False, fontsize=10, loc='lower right')

    # === Right panel: Top accuracies per participant ===
    ax1 = fig.add_subplot(gs[1])
    jitter = np.random.uniform(-0.25, 0.25, size=n)
    for i, key in enumerate(sorted_keys):
        ax1.errorbar(jitter[i], top_acc_values[i],   # yerr=top_acc_std[i],
                     fmt='o', color=participant_colors[key], markersize=6, capsize=3)

    # Cosmetic adjustments
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([0.48, 0.9])
    yticks = ax1.get_yticks()
    ax1.set_yticks([yt for yt in yticks if yt >= 0.5])
    ax1.spines['left'].set_visible(False)
    ax1.plot([-0.5, -0.5], [0.5, 0.9], transform=ax1.transData, color='black', lw=1)
    ax1.set_xticks([])
    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.spines['left'].set_linewidth(1)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.set_yticklabels([])  # hide labels
    ax1.tick_params(axis='y', direction='in', length=3.5, width=0.8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else: plt.show()
    
    
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
    out = Path(f'Plotting/Fig4/insula_motor_distribution.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
