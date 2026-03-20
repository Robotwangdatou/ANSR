# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 08:36:40 2026
The code of ANSR
@author: WW
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# =========================== PARAMETER CONFIGURATION ===========================
# ==============================================================================
# Core parameters - Modify these according to your dataset and requirements
DATASET_FILE = 'Ecoli.csv'          # Dataset file path/name
OPTIMAL_K_FACTORS = [0.3]            # Optimal K factors for adaptive KNN
OPTIMAL_DENSITY_K = 3                # K value for local density calculation
OPTIMAL_BOUNDARY_PERCENTILE = 60     # Percentile for boundary sample detection
N_REPEATS = 10                       # Number of experiment repeats
N_FOLDS = 5                          # Number of folds for stratified K-fold
RANDOM_SEED_BASE = 42                # Base random seed for reproducibility

# ==============================================================================
# ============================== CORE FUNCTIONS ================================
# ==============================================================================
def batch_calculate_neighbor_stats(knn_model, X, k):
    """
    Batch calculate neighbor statistical features for all samples
    Returns: array of shape (n_samples, 7) with columns: mean/var/max/median/iqr/q1/q3
    """
    if knn_model is None or len(X) == 0:
        return np.zeros((len(X), 7))
    
    # Get neighbor distances in batch (vectorized operation)
    distances, _ = knn_model.kneighbors(X)
    
    # Calculate statistical features in batch
    mean_dist = np.mean(distances, axis=1)
    var_dist = np.var(distances, axis=1)
    max_dist = np.max(distances, axis=1)
    median_dist = np.median(distances, axis=1)
    q1_dist = np.percentile(distances, 25, axis=1)
    q3_dist = np.percentile(distances, 75, axis=1)
    iqr_dist = q3_dist - q1_dist
    
    # Combine features into matrix
    stats_matrix = np.column_stack([
        mean_dist, var_dist, max_dist, median_dist,
        iqr_dist, q1_dist, q3_dist
    ])
    return stats_matrix

def g_mean_score(y_true, y_pred):   
    """Calculate G-mean score from confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    sensitivities = np.nan_to_num(sensitivities, nan=0.0)
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean

def calculate_metrics(y_true, y_pred, y_prob, minority_class):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # AUC calculation (binary/multi-class)
    if len(np.unique(y_true)) == 2:
        metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] >= 2 else 0.0
    else:
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        metrics['AUC'] = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    
    # G-mean score
    metrics['Gmean'] = g_mean_score(y_true, y_pred)
    
    # Minority class metrics
    metrics['Recall_minor'] = recall_score(y_true, y_pred, labels=[minority_class], average='binary', zero_division=0)
    metrics['Precision_minor'] = precision_score(y_true, y_pred, labels=[minority_class], average='binary', zero_division=0)
    metrics['F1_minor'] = f1_score(y_true, y_pred, labels=[minority_class], average='binary', zero_division=0)
    
    return metrics

def build_knn_for_test(X_train_subset, k):
    """Build KNN model with multi-threading optimization"""
    if len(X_train_subset) <= 1:
        return None
    k = min(k, len(X_train_subset) - 1)
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    knn.fit(X_train_subset)
    return knn

def calculate_local_density(X, k0=OPTIMAL_DENSITY_K):
    """Calculate local density using KNN mean distance"""
    if len(X) < k0:
        return np.array([0.0] * len(X))
    knn = NearestNeighbors(n_neighbors=k0, algorithm='ball_tree', n_jobs=-1)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    return np.mean(distances, axis=1)

def is_boundary_sample(X, y, minority_class):
    """Identify boundary samples using distance ratio method"""
    X_minor = X[y == minority_class]
    X_major = X[y != minority_class]
    
    if len(X_minor) < 3 or len(X_major) < 3:
        return np.array([False] * len(X))
    
    # Calculate class centers and distance ratios
    center_minor = np.mean(X_minor, axis=0)
    center_major = np.mean(X_major, axis=0)
    dist_to_minor = np.linalg.norm(X - center_minor, axis=1)
    dist_to_major = np.linalg.norm(X - center_major, axis=1)
    distance_ratio = dist_to_minor / (dist_to_major + 1e-8)
    
    # Determine boundary samples using percentile threshold
    boundary_threshold = np.percentile(distance_ratio, OPTIMAL_BOUNDARY_PERCENTILE)
    return distance_ratio >= boundary_threshold

def evaluate_classifier(classifier, x_train, y_train, x_test, y_test, minority_class):
    """Train and evaluate classifier with comprehensive metrics"""
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    # Get prediction probabilities
    if hasattr(classifier, 'predict_proba'):
        y_prob = classifier.predict_proba(x_test)
    else:
        y_prob = classifier.decision_function(x_test)
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T
    
    return calculate_metrics(y_test, y_pred, y_prob, minority_class)

# ==============================================================================
# ============================== MAIN EXECUTION ================================
# ==============================================================================
if __name__ == '__main__':
    # 1. Load dataset
    try:
        datasetY = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found, please check the path")
        exit()
    
    # 2. Data preprocessing
    y = datasetY['class']
    X = datasetY.drop(columns='class')
    class_distribution = y.value_counts()
    minority_class = class_distribution.idxmin()
    
    print(f"Minority class label: {minority_class}")
    print("="*60)
    
    # 3. Initialize variables
    all_results = []
    classifiers = {'GaussianNB': GaussianNB()}
    
    # 4. Multi-repeat stratified K-fold experiments
    for repeat in range(N_REPEATS):
        print(f"Running repeat {repeat+1}/{N_REPEATS}")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat+RANDOM_SEED_BASE)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Split train/test data
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
            
            # Split minority/majority classes
            minority_mask_train = (y_train == minority_class)
            majority_mask_train = ~minority_mask_train
            X_train_minor = X_train[minority_mask_train].values
            X_train_major = X_train[majority_mask_train].values
            n_minor = len(X_train_minor)
            n_major = len(X_train_major)
            
            # Calculate density and boundary attributes
            minor_density = calculate_local_density(X_train_minor) if n_minor>OPTIMAL_DENSITY_K else np.array([0.0]*n_minor)
            minor_boundary = is_boundary_sample(X_train_minor, np.full(n_minor, minority_class), minority_class) if n_minor>0 else np.array([False]*n_minor)
            major_density = calculate_local_density(X_train_major) if n_major>OPTIMAL_DENSITY_K else np.array([0.0]*n_major)
            major_boundary = is_boundary_sample(X_train_major, np.where(y_train[majority_mask_train]==minority_class,1,0), 0) if n_major>0 else np.array([False]*n_major)
            
            # Adaptive K value calculation
            knn_models = {}
            for factor in OPTIMAL_K_FACTORS:
                # Minority class K value
                k_base_minor = max(1, int(n_minor * factor)) if n_minor>0 else 1
                if n_minor>0:
                    minor_density_thresh = np.percentile(minor_density, 70) if n_minor>1 else 0
                    minor_low_density_ratio = np.mean(minor_density > minor_density_thresh)
                    minor_boundary_ratio = np.mean(minor_boundary)
                    k_minor = int(k_base_minor * (1 + 0.5 * (minor_low_density_ratio + minor_boundary_ratio)))
                    k_minor = min(k_minor, n_minor - 1)
                else:
                    k_minor = 1
                
                # Majority class K value
                k_base_major = max(1, int(n_major * factor)) if n_major>0 else 1
                if n_major>0:
                    major_density_thresh = np.percentile(major_density, 70) if n_major>1 else 0
                    major_low_density_ratio = np.mean(major_density > major_density_thresh)
                    major_boundary_ratio = np.mean(major_boundary)
                    k_major = int(k_base_major * (1 + 0.3 * (major_low_density_ratio + major_boundary_ratio)))
                    k_major = min(k_major, n_major - 1)
                else:
                    k_major = 1
                
                # Build KNN models
                knn_minor = build_knn_for_test(X_train_minor, k_minor)
                knn_major = build_knn_for_test(X_train_major, k_major)
                knn_models[factor] = (knn_minor, knn_major, k_minor, k_major)
            
            # Batch feature calculation
            x_train1 = pd.DataFrame(index=X_train.index)
            x_test1 = pd.DataFrame(index=X_test.index)
            
            for factor in OPTIMAL_K_FACTORS:
                knn_minor, knn_major, k_minor, k_major = knn_models[factor]
                
                # Calculate train set features
                train_minor_stats = batch_calculate_neighbor_stats(knn_minor, X_train.values, k_minor)
                train_major_stats = batch_calculate_neighbor_stats(knn_major, X_train.values, k_major)
                
                # Assign train features
                stats_cols = ['mean', 'var', 'max', 'median', 'iqr', 'q1', 'q3']
                for i, stat in enumerate(stats_cols):
                    x_train1[f'A_{stat}_{factor}'] = train_minor_stats[:, i]
                    x_train1[f'C_{stat}_{factor}'] = train_major_stats[:, i]
                
                # Calculate test set features
                test_minor_stats = batch_calculate_neighbor_stats(knn_minor, X_test.values, k_minor)
                test_major_stats = batch_calculate_neighbor_stats(knn_major, X_test.values, k_major)
                
                # Assign test features
                for i, stat in enumerate(stats_cols):
                    x_test1[f'A_{stat}_{factor}'] = test_minor_stats[:, i]
                    x_test1[f'C_{stat}_{factor}'] = test_major_stats[:, i]
            
            # Calculate feature ratios (vectorized operation with zero division protection)
            for factor in OPTIMAL_K_FACTORS:
                for stat in stats_cols:
                    a_col = f'A_{stat}_{factor}'
                    c_col = f'C_{stat}_{factor}'
                    new_col = f'new_{stat}_{factor}'
                    
                    # Zero division handling
                    train_c_vals = x_train1[c_col].values
                    eps = 1e-8 if np.min(train_c_vals) == 0 else np.min(train_c_vals) * 1e-3
                    x_train1[new_col] = x_train1[a_col].values / (train_c_vals + eps)
                    
                    test_c_vals = x_test1[c_col].values
                    x_test1[new_col] = x_test1[a_col].values / (test_c_vals + eps)
            
            # Filter features
            new_features = [f'new_{stat}_{factor}' for factor in OPTIMAL_K_FACTORS 
                           for stat in stats_cols]
            x_train1 = x_train1[new_features]
            x_test1 = x_test1[new_features]
            
            # Evaluate classifiers
            for clf_name, clf in classifiers.items():
                fold_metrics = evaluate_classifier(clf, x_train1, y_train, x_test1, y_test, minority_class)
                fold_metrics['repeat'] = repeat+1
                fold_metrics['fold'] = fold_idx+1
                fold_metrics['classifier'] = clf_name
                all_results.append(fold_metrics)
    
    # Save and display results
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print(f"Final Results ({N_FOLDS}-fold × {N_REPEATS} repeats)")
    print("="*60)
    
    for clf_name in classifiers.keys():
        clf_results = results_df[results_df['classifier'] == clf_name]
        metrics = ['AUC', 'Gmean', 'Recall_minor', 'Precision_minor', 'F1_minor']
        mean_vals = clf_results[metrics].mean()
        std_vals = clf_results[metrics].std()
        
        print(f"\nClassifier: {clf_name}")
        for metric in metrics:
            print(f"{metric:15s} | Mean: {mean_vals[metric]:.4f} | Std: {std_vals[metric]:.4f}")