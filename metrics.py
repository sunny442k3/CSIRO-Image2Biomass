import numpy as np

R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = R2_WEIGHTS
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]; yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores

def weighted_r2_score_global(y_true: np.ndarray, y_pred: np.ndarray):
    weights = R2_WEIGHTS
    flat_true = y_true.copy().reshape(-1)
    flat_pred = y_pred.copy().reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2