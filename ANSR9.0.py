# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:57:58 2026
This is the code of the ANSR algorithm, without five-fold cross-validation or multiple rounds of averaging.
@author: robot
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# =========================== PARAMETER CONFIGURATION ===========================
# ==============================================================================
DATASET_FILE = 'ecoli3.csv'          # Dataset path
OPTIMAL_K_FACTORS = [0.3]                    # Core scaling factor γ
OPTIMAL_DENSITY_K = 3                        # k0 for local density
OPTIMAL_BOUNDARY_PERCENTILE = 60              # Boundary threshold τ
RANDOM_SEED = 42                             # Random seed for reproducibility
TEST_SIZE = 0.2                              # Train:Test = 8:2

# ==============================================================================
# ============================== CORE FUNCTIONS ================================
# ==============================================================================
def batch_calculate_neighbor_stats(knn_model, X, k):
    """
    Compute 7 statistical features from neighbor distances in batch.
    Returns: [mean, var, max, median, iqr, q1, q3] for each sample
    """
    if knn_model is None or len(X) == 0:
        return np.zeros((len(X), 7))
    distances, _ = knn_model.kneighbors(X)
    mean_dist = np.mean(distances, axis=1)
    var_dist = np.var(distances, axis=1)
    max_dist = np.max(distances, axis=1)
    median_dist = np.median(distances, axis=1)
    q1_dist = np.percentile(distances, 25, axis=1)
    q3_dist = np.percentile(distances, 75, axis=1)
    iqr_dist = q3_dist - q1_dist
    return np.column_stack([mean_dist, var_dist, max_dist, median_dist, iqr_dist, q1_dist, q3_dist])

def g_mean_score(y_true, y_pred):
    """Compute G-mean for imbalanced binary classification"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    fp = cm[0, 1]
    recall = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return np.sqrt(recall * spec)

def specificity_score(y_true, y_pred):
    """Compute Specificity (True Negative Rate)"""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp + 1e-8)

def npv_score(y_true, y_pred):
    """Compute Negative Predictive Value (NPV)"""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fn = cm[1, 0]
    return tn / (tn + fn + 1e-8)

def calculate_metrics(y_true, y_pred, y_prob, minority_class):
    """Compute all evaluation metrics"""
    metrics = {}
    metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
    metrics['Gmean'] = g_mean_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred, pos_label=minority_class, zero_division=0)
    metrics['Precision'] = precision_score(y_true, y_pred, pos_label=minority_class, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, pos_label=minority_class, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['Specificity'] = specificity_score(y_true, y_pred)
    metrics['NPV'] = npv_score(y_true, y_pred)
    return metrics

def build_knn_for_class(X_subset, k):
    """Build BallTree KNN for a given class subset"""
    if len(X_subset) <= 1:
        return None
    k = min(k, len(X_subset) - 1)
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    knn.fit(X_subset)
    return knn

def calculate_local_density(X, k0=OPTIMAL_DENSITY_K):
    """Compute local density as average kNN distance"""
    if len(X) < k0:
        return np.zeros(len(X))
    knn = NearestNeighbors(n_neighbors=k0, algorithm='ball_tree', n_jobs=-1)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    return np.mean(distances, axis=1)

def identify_boundary_samples(X, y, minority_class):
    """Identify boundary samples using distance ratio to class centers"""
    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]
    if len(X_min) < 2 or len(X_maj) < 2:
        return np.zeros(len(X), dtype=bool)
    center_min = np.mean(X_min, axis=0)
    center_maj = np.mean(X_maj, axis=0)
    d_min = np.linalg.norm(X - center_min, axis=1)
    d_maj = np.linalg.norm(X - center_maj, axis=1)
    ratio = d_min / (d_maj + 1e-8)
    threshold = np.percentile(ratio, OPTIMAL_BOUNDARY_PERCENTILE)
    return ratio >= threshold

def evaluate_model(model, X_train, y_train, X_test, y_test, minority_class):
    """Train model and return full evaluation metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return calculate_metrics(y_test, y_pred, y_prob, minority_class)

# ==============================================================================
# ============================== MAIN PIPELINE ================================
# ==============================================================================
if __name__ == '__main__':
    # Load dataset
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset {DATASET_FILE} not found.")
        exit()

    # Split features and labels
    y = df['class']
    X = df.drop(columns=['class'])
    minority_class = y.value_counts().idxmin()

    print(f"Dataset: {DATASET_FILE}")
    print(f"Minority class: {minority_class}")
    print("=" * 70)

    # Train-test split (stratified, 8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Separate minority and majority class in training set
    train_min_mask = (y_train == minority_class)
    train_maj_mask = ~train_min_mask
    X_train_min = X_train[train_min_mask]
    X_train_maj = X_train[train_maj_mask]
    n_min = len(X_train_min)
    n_maj = len(X_train_maj)

    # Compute local density and boundary flags
    density_min = calculate_local_density(X_train_min) if n_min >= OPTIMAL_DENSITY_K else np.zeros(n_min)
    boundary_min = identify_boundary_samples(X_train_min, np.full(n_min, minority_class), minority_class) if n_min > 0 else np.zeros(n_min, dtype=bool)
    density_maj = calculate_local_density(X_train_maj) if n_maj >= OPTIMAL_DENSITY_K else np.zeros(n_maj)
    boundary_maj = identify_boundary_samples(X_train_maj, np.zeros(n_maj), 0) if n_maj > 0 else np.zeros(n_maj, dtype=bool)

    # Adaptive k calculation
    knn_dict = {}
    for gamma in OPTIMAL_K_FACTORS:
        # Adaptive k for minority class
        if n_min > 0:
            k_base_min = max(1, int(n_min * gamma))
            dens_thr_min = np.percentile(density_min, 70) if n_min > 1 else 0
            dens_ratio_min = np.mean(density_min > dens_thr_min)
            bnd_ratio_min = np.mean(boundary_min)
            k_min = int(k_base_min * (1 + 0.5 * (dens_ratio_min + bnd_ratio_min)))
            k_min = min(k_min, n_min - 1)
        else:
            k_min = 1

        # Adaptive k for majority class
        if n_maj > 0:
            k_base_maj = max(1, int(n_maj * gamma))
            dens_thr_maj = np.percentile(density_maj, 70) if n_maj > 1 else 0
            dens_ratio_maj = np.mean(density_maj > dens_thr_maj)
            bnd_ratio_maj = np.mean(boundary_maj)
            k_maj = int(k_base_maj * (1 + 0.3 * (dens_ratio_maj + bnd_ratio_maj)))
            k_maj = min(k_maj, n_maj - 1)
        else:
            k_maj = 1

        # Build KNN models
        knn_min = build_knn_for_class(X_train_min, k_min)
        knn_maj = build_knn_for_class(X_train_maj, k_maj)
        knn_dict[gamma] = (knn_min, knn_maj, k_min, k_maj)

    # Construct statistical ratio features
    stats = ['mean', 'var', 'max', 'median', 'iqr', 'q1', 'q3']
    X_train_feat = pd.DataFrame(index=range(len(X_train)))
    X_test_feat = pd.DataFrame(index=range(len(X_test)))

    for gamma in OPTIMAL_K_FACTORS:
        knn_min, knn_maj, k_min, k_maj = knn_dict[gamma]

        # Compute neighbor stats for training set
        st_min_train = batch_calculate_neighbor_stats(knn_min, X_train, k_min)
        st_maj_train = batch_calculate_neighbor_stats(knn_maj, X_train, k_maj)
        for i, s in enumerate(stats):
            X_train_feat[f'A_{s}_{gamma}'] = st_min_train[:, i]
            X_train_feat[f'C_{s}_{gamma}'] = st_maj_train[:, i]

        # Compute neighbor stats for test set
        st_min_test = batch_calculate_neighbor_stats(knn_min, X_test, k_min)
        st_maj_test = batch_calculate_neighbor_stats(knn_maj, X_test, k_maj)
        for i, s in enumerate(stats):
            X_test_feat[f'A_{s}_{gamma}'] = st_min_test[:, i]
            X_test_feat[f'C_{s}_{gamma}'] = st_maj_test[:, i]

    # Compute final ratio features
    feat_list = []
    for gamma in OPTIMAL_K_FACTORS:
        for s in stats:
            col = f'new_{s}_{gamma}'
            X_train_feat[col] = X_train_feat[f'A_{s}_{gamma}'] / (X_train_feat[f'C_{s}_{gamma}'] + 1e-8)
            X_test_feat[col] = X_test_feat[f'A_{s}_{gamma}'] / (X_test_feat[f'C_{s}_{gamma}'] + 1e-8)
            feat_list.append(col)

    X_train_final = X_train_feat[feat_list]
    X_test_final = X_test_feat[feat_list]

    # Evaluate GaussianNB classifier
    clf = GaussianNB()
    results = evaluate_model(clf, X_train_final, y_train, X_test_final, y_test, minority_class)

    # Print final results
    print("Final Evaluation Metrics:")
    print("-" * 70)
    for key, val in results.items():
        print(f"{key:12s} : {val:.4f}")