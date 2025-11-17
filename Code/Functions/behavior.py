import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple



def compute_behavior(labels, use_rolling=True, use_smoothing=True, window_size=20, sigma=15):
    """
    Compute rolling/smoothed performance from game labels.

    Args:
        labels: dict with 'Points' and 'timestamp'
        use_rolling: compute rolling performance over trials
        use_smoothing: apply Gaussian smoothing
        window_size: rolling window size
        sigma: smoothing sigma

    Returns:
        dict with:
            't': timestamps for each rolling/smoothed performance value
            'trialSuccess': 1=success, 0=failure
            'trialTimes': indices of trial events
            'avgPerformance': rolling/smoothed performance per trial
    """
    points = np.asarray(labels["Points"])
    timestamps = np.asarray(labels["timestamp"]) - labels["timestamp"][0]

    # Find trial events
    delta = np.diff(points)
    success = np.where(delta == 1)[0]
    failure = np.where(delta <= -1)[0]
    trial_idx = np.sort(np.concatenate([success, failure]))
    trialSuccess = np.where(np.isin(trial_idx, success), 1, 0)

    # Rolling performance
    avgPerformance = np.array([
        np.mean(trialSuccess[max(0, i - window_size + 1):i + 1])
        for i in range(len(trialSuccess))
    ]) if use_rolling else trialSuccess

    # Smoothing
    if use_smoothing:
        avgPerformance = gaussian_filter1d(avgPerformance, sigma=sigma)

    # Corresponding timestamps for each performance value
    t_perf = timestamps[trial_idx]

    return {
        "t": t_perf,               
        "trialSuccess": trialSuccess,
        "trialTimes": trial_idx,
        "avgPerformance": avgPerformance
    }
    

def plot_behavior(time_behavior, rollingBehavior, avgSuccess, save_path=None):
    """
    Plot overall game performance per participant.
    
    Args:
        time_behavior (dict): {participant: timestamps vector}
        rollingBehavior (dict): {participant: rolling performance vector}
        avgSuccess (dict): {participant: average success (0–1 or 0–100)}
        save_path (str, optional): path to save the figure (e.g., 'output/performance.pdf')
    """
    participants = list(rollingBehavior.keys())

    # --- Participant colors --- (Color ordered according to training accuracy, low to high)
    train_colors = {
        "P07": (0.0, 0.392, 0.0),
        "P08": (0.196, 0.587, 0.196),
        "P02": (0.393, 0.782, 0.393),
        "P09": (0.596, 0.984, 0.596),
        "P06": (0.531, 0.924, 0.729),
        "P05": (0.465, 0.861, 0.867),
        "P03": (0.4, 0.8, 1.0),
        "P04": (0.264, 0.595, 0.795),
        "P01": (0.132, 0.398, 0.598),
        "P10": (0.0, 0.2, 0.4),
    }
    shades = [train_colors[ppt] for ppt in participants if ppt in train_colors]

    # --- Prepare data ---
    avgPerformance_list = [rollingBehavior[ppt] for ppt in participants]
    t_marker_list = [time_behavior[ppt] for ppt in participants]
    grandAvgPerf = [avgSuccess[ppt] for ppt in participants]
    grandStdPerf = [np.std(rollingBehavior[ppt]) for ppt in participants]

    # --- Figure setup ---
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1], wspace=0.15)

    # --- Left: rolling performance over time ---
    ax0 = plt.subplot(gs[0])
    for i, ppt in enumerate(participants):
        x = t_marker_list[i] / 60  # Convert seconds to minutes
        x[0] = 0.05                 # Force start briefly after 0 so it doesnt overlap with axis
        y = avgPerformance_list[i]
        mask = x <= 10
        ax0.plot(x[mask], y[mask], color=shades[i], linewidth=2.5, clip_on=False)

    # Cosmetics
    ax0.axhline(0.5, color='grey', linestyle='--', linewidth=1.5)
    ax0.set(
        xlabel="Time (min)",
        ylabel="Performance (%)",
        title="Game performance",
        xlim=(0, 10),
        ylim=(0.485, 1),
        yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        yticklabels=['50', '60', '70', '80', '90', '100']
    )
    ax0.set_title("Game performance", fontsize=14)
    ax0.set_xlabel("Time (min)", fontsize=13)
    ax0.set_ylabel("Performance (%)", fontsize=13)
    ax0.tick_params(axis='both', labelsize=12, length=0)
    for spine in ['top', 'right']:
        ax0.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax0.spines[spine].set_linewidth(1)

    # --- Right: summary performance scatter ---
    ax1 = plt.subplot(gs[1])
    jitter = np.random.uniform(-0.4, 0.4, len(grandAvgPerf))
    for i in range(len(grandAvgPerf)):
        ax1.errorbar(
            jitter[i], grandAvgPerf[i],
            yerr=grandStdPerf[i], fmt='o',
            color=shades[i], markersize=6, capsize=3
        )

    # Cosmetics
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([0.485, 1])
    ax1.plot([-0.5, -0.5], [0.5, 1.0], transform=ax1.transData, color='black', lw=1, clip_on=False)
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', labelsize=12, direction='in')
    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_linewidth(1)

    # --- Save or show ---
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()
        
        
        
        
def plot_error_vs_standard(time_behavior, rollingBehavior, avgSuccess,
                           time_behavior_error, rollingBehaviorError, avgSuccessError,
                           save_path=None):
    """
    Plot standard vs error-corrected performance for participants that exist in both datasets.
    """
    # Keep only participants present in both
    participants = sorted(set(avgSuccess.keys()) & set(avgSuccessError.keys()))
    if not participants:
        print("No overlapping participants to plot!")
        return
    else:
        print(f"Participants with error correction: {len(participants)} / {len(avgSuccess)}")

    # Colors for up to 4 participants (light/dark for each)
    shades = ['#AACDAA', 'forestgreen', 'lightsteelblue', 'tab:blue']
    participant_palettes = [(shades[i*2], shades[i*2+1]) for i in range(len(participants))]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.5, 4.5),
                                   gridspec_kw={'height_ratios':[6.5,3.5], 'hspace':0.5})

    # --- TOP PLOT ---
    legends = ['Standard game', 'Error correction']
    for i, p in enumerate(participants):
        t_std = time_behavior[p].copy()
        t_err = time_behavior_error[p].copy()
        t_std[0] = 0.05         # This is done to let behavior begin at start of x-axis without overlapping with y axis
        t_err[0] = 0.05         # Because the first obstacle is only passed after a few seconds

        ax0.plot(t_std/60, rollingBehavior[p], color=participant_palettes[i][0], lw=2.5,
                 label=legends[0] if i==0 else "")
        ax0.plot(t_err/60, rollingBehaviorError[p], color=participant_palettes[i][1], lw=2.5,
                 label=legends[1] if i==0 else "")
    # Format plot
    ax0.set_xlim(0,10)
    ax0.set_ylim(0.48,1)
    ax0.set_xlabel("Time (min)", fontsize=13)
    ax0.set_ylabel("Performance (%)", fontsize=13)
    ax0.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax0.set_yticklabels(['50', '60', '70', '80', '90', '100'], fontsize=12)
    ax0.set_xticks([0, 2, 4, 6, 8, 10])
    ax0.set_xticklabels([str(x) for x in [0, 2, 4, 6, 8, 10]], fontsize=12)
    ax0.set_title("Performance with error correction", fontsize=14)
    ax0.axhline(0.5, color='grey', linestyle='--', lw=2)
    ax0.spines['top'].set_color('none')
    ax0.spines['right'].set_color('none')
    ax0.spines['left'].set_linewidth(1)
    ax0.spines['bottom'].set_linewidth(1)
    ax0.tick_params(axis='both', labelsize=12, length=0)

    # Split legend
    handles = [
        tuple(Patch(facecolor=participant_palettes[i][0]) for i in range(len(participants))),  # all participants, cond 1
        tuple(Patch(facecolor=participant_palettes[i][1]) for i in range(len(participants))),  # all participants, cond 2
    ]
    ax0.legend(handles, ['Standard game', 'Error correction'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc='lower right', bbox_to_anchor=(1.03,0),
               bbox_transform=ax0.transAxes, fontsize=11, frameon=False,
               ncol=2, handlelength=2, handletextpad=0.4)

    # --- BOTTOM PLOT ---
    bar_width = 0.6
    group_spacing = 0.1
    y_pos = []
    cur_y = 0
    for _ in participants:
        y_pos.extend([cur_y, cur_y+bar_width])
        cur_y += 2+group_spacing

    for i, p in enumerate(participants):
        std_idx, err_idx = 2*i, 2*i+1
        ax1.barh(y_pos[std_idx], avgSuccess[p], height=bar_width, color=participant_palettes[i][0])
        ax1.barh(y_pos[err_idx], avgSuccessError[p], height=bar_width, color=participant_palettes[i][1])

    # Format plot
    ax1.set_yticks([(y_pos[i]+y_pos[i+1])/2 for i in range(0, len(y_pos),2)])
    ax1.set_yticklabels(participants, fontsize=13)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average performance (%)', fontsize=13)
    ax1.set_xlim(0.49,0.9)
    ax1.set_xticks([0.5,0.6,0.7,0.8,0.9])
    ax1.set_xticklabels(['50','60','70','80','90'], fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', length=0)
    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()
