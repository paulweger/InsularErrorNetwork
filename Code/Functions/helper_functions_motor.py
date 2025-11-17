import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.io import loadmat
from pathlib import Path
from scipy.stats import gaussian_kde

# Local imports
from Functions.RNN import RNN


def train_incremental(x, y, prior, algorithm, runs=50, test_fraction=0.2):
    """
    Incrementally trains a model using shifted test sets.
    Each run uses `test_fraction` of data for testing and the rest for training,
    shifting the test set window in each iteration.
    """

    y = LabelEncoder().fit_transform(y.ravel())                                                
    n_samples = len(y)
    test_size = int(n_samples * test_fraction)
    all_fold_accuracies, all_fold_increments = [], []

    for run in range(runs):
        print(".", end=" ")

        start_idx = int((run * n_samples) / runs) % (n_samples - test_size)
        test_idx = np.arange(start_idx, start_idx + test_size)
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        x_train_full, y_train_full = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        n_train = x_train_full.shape[0]
        x_test_selected = x_test

        # Define increments 
        inc_steps = [4, 10, 20, 40, 80, 120, 160, 200, 240]
        current = 240 + 80
        while current < n_train:
            inc_steps.append(current)
            current += 80
        if inc_steps[-1] != n_train: inc_steps.append(n_train)   
    
        # Loop over increments
        fold_accuracies, fold_increments, scores = [], [], defaultdict(list)
        for j, increment_size in enumerate(inc_steps):
            balanced_candidates = []

            # Make balanced training set
            for start_index in range(0, len(y_train_full) - increment_size + 1):
                x_candidate = x_train_full[start_index:start_index + increment_size]
                y_candidate = y_train_full[start_index:start_index + increment_size]
                unique_classes, class_counts = np.unique(y_candidate, return_counts=True)
                if len(unique_classes) > 1:
                    class_ratios = class_counts / sum(class_counts)
                    if np.all(np.abs(class_ratios - 0.5) <= 0.10):  # up to 10% imbalance
                        balanced_candidates.append(start_index)
            if balanced_candidates:
                start_index = np.random.choice(balanced_candidates)
                x_train = x_train_full[start_index:start_index + increment_size]
                y_train = y_train_full[start_index:start_index + increment_size]
            else:
                print(f"⚠️ Warning: No balanced window found for increment size {increment_size}")
                x_train = x_train_full[:increment_size]
                y_train = y_train_full[:increment_size]
    

            # Perform feature selection 
            selector = SelectKBest(score_func=f_classif, k=50)
            x_train = selector.fit_transform(x_train, y_train)
            x_test_selected = selector.transform(x_test)
            

            # Train ML model
            algorithm = str(algorithm)
            class_weights = {0: prior[0], 1: prior[1]}
            if algorithm == 'LDA': model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=prior)
            elif algorithm == 'RNN': model = RNN(prior=prior, input_dim=x_train.shape[1])
            else: 
                print(f"Algorithm {algorithm} not found")
                continue
            model.fit(x_train, y_train)

            # Evaluate
            if algorithm == 'RNN': 
                y_prob = model.evaluate(x_test_selected)
                y_pred = np.array(y_prob >= 0.5, dtype=int)
            else: 
                y_prob = model.predict_proba(x_test_selected)[:, 1]
                y_pred = model.predict(x_test_selected)
            accuracy = np.mean(y_pred == y_test)


            # Store results
            fold_accuracies.append(accuracy)
            fold_increments.append(increment_size)
        all_fold_accuracies.append(fold_accuracies)
        all_fold_increments.append(fold_increments)

    # Convert to numpy arrays for easier handling
    all_fold_accuracies = np.array(all_fold_accuracies)
    all_fold_increments = np.array(all_fold_increments)
    
    return all_fold_accuracies, all_fold_increments



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
