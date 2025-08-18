# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:51:07 2025
Implementation of ANSR algorithm
@author: robot
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.naive_bayes import GaussianNB

# Calculate G-mean
def g_mean_score(y_true, y_pred):   
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean

# Calculate evaluation metrics
def calculate_metrics(y, y_predict, y_prob, Jresult):
    # Check if it is a binary classification problem
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, y_prob[:, 1])
    else:
        # Multi-class processing
        y_bin = label_binarize(y, classes=np.unique(y))
        auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    
    g_mean = g_mean_score(y, y_predict)
    f1 = f1_score(y, y_predict, average='macro')  # F-measure
    new_r = [auc, g_mean, f1]
    Jresult.extend(new_r)

# Calculate neighbor statistical features 
def calculate_neighbor_stats(knn_model, point, k):
    distances, _ = knn_model.kneighbors(point.reshape(1, -1))
    distances = distances[0]
    
    # Basic statistics
    mean_dist = np.mean(distances)
    var_dist = np.var(distances)
    max_dist = np.max(distances)
    median_dist = np.median(distances)  # 50th percentile
    
    # Added 25th percentile (Q1) and 75th percentile (Q3)
    q1_dist = np.percentile(distances, 25)  # 25th percentile
    q3_dist = np.percentile(distances, 75)  # 75th percentile
    iqr_dist = q3_dist - q1_dist  # Interquartile range (based on Q1 and Q3)
    
    return mean_dist, var_dist, max_dist, median_dist, iqr_dist, q1_dist, q3_dist

# Classifier evaluation function
def evaluate_classifier(classifier, x_train, y_train, x_test, y_test):
    Jresult = []
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    # Get probabilities
    if hasattr(classifier, 'predict_proba'):
        y_prob = classifier.predict_proba(x_test)
    else:
        y_prob = classifier.decision_function(x_test)
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T
    
    calculate_metrics(y_test, y_pred, y_prob, Jresult)
    return Jresult

# Added: Calculate local density of samples (for k-value adjustment)
def calculate_local_density(X, k0=5):
    """Calculate local density of each sample (measured by average distance to nearest k0 samples)"""
    knn = NearestNeighbors(n_neighbors=k0, algorithm='ball_tree')
    knn.fit(X)
    distances, _ = knn.kneighbors(X)  # Distances to nearest k0 samples for each sample
    mean_dist = np.mean(distances, axis=1)  # Smaller average distance indicates higher density
    return mean_dist

# Added: Determine if a sample is a boundary sample
def is_boundary_sample(X, y, minority_class):
    """Judge if a sample is a boundary sample by distance ratio to two class centers"""
    # Calculate centers of minority and majority classes
    X_minor = X[y == minority_class]
    X_major = X[y != minority_class]
    center_minor = np.mean(X_minor, axis=0) if len(X_minor) > 0 else np.zeros(X.shape[1])
    center_major = np.mean(X_major, axis=0) if len(X_major) > 0 else np.zeros(X.shape[1])
    
    # Distance from each sample to the two class centers
    dist_to_minor = np.linalg.norm(X - center_minor, axis=1)
    dist_to_major = np.linalg.norm(X - center_major, axis=1)
    
    # Distance ratio (closer to 1, more likely to be a boundary)
    distance_ratio = dist_to_minor / (dist_to_major + 1e-8)
    boundary_threshold = np.percentile(distance_ratio, 70)  # Take 70th percentile as threshold
    return distance_ratio >= boundary_threshold  # True indicates boundary sample

# Main function
if __name__ == '__main__':
    try:
        datasetY = pd.read_csv('iris.csv')
    except FileNotFoundError:
        print("Error: csv file not found, please check the path")
        exit()
        
    k_factors = [0.5]  # k-value base ratio factor
    y = datasetY['class']
    X = datasetY.drop(columns='class')
    
    # Determine minority class
    class_distribution = y.value_counts()
    minority_class = class_distribution.idxmin()
    print(f"Minority class label: {minority_class}")
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Classifier definition (only保留GNB)
    classifiers = {
        'GaussianNB': lambda: GaussianNB()
    }
    
    # Result storage
    experiment_results = {name: pd.DataFrame(columns=['AUC', 'Gmean', 'Fmeasure']) 
                         for name in classifiers.keys()}
    
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Data standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert standardized data back to DataFrame, retain index
        X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        
        x_train1 = pd.DataFrame(index=X_train.index)
        x_test1 = pd.DataFrame(index=X_test.index)
        
        # Minority and majority class indices
        minority_indices = np.where(y_train == minority_class)[0]
        majority_indices = np.where(y_train != minority_class)[0]
        
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)
        
        X_train_minority = X_train.iloc[minority_indices]
        X_train_majority = X_train.iloc[majority_indices]
        
        # Added: Calculate local density and boundary attributes of minority and majority samples
        # Density and boundary judgment for minority samples
        minor_density = calculate_local_density(X_train_minority.values, k0=5) if n_minority > 5 else np.array([0.0]*n_minority)
        minor_is_boundary = is_boundary_sample(
            X_train_minority.values, 
            np.full(n_minority, minority_class),  # Construct minority class label array
            minority_class=minority_class
        ) if n_minority > 0 else np.array([False]*n_minority)
        
        # Density and boundary judgment for majority samples
        major_density = calculate_local_density(X_train_majority.values, k0=5) if n_majority > 5 else np.array([0.0]*n_majority)
        major_is_boundary = is_boundary_sample(
            X_train_majority.values, 
            np.where(y_train.iloc[majority_indices] == minority_class, 1, 0),  # Set majority class label to 0
            minority_class=0  # Temporary label, only used for center calculation
        ) if n_majority > 0 else np.array([False]*n_majority)
        
        # Build KNN models (adaptive k-value)
        knn_models = {}
        for factor in k_factors:
            # Base k-value (factor ratio of minority sample count)
            k_base_minor = max(1, int(n_minority * factor)) if n_minority > 0 else 1
            k_base_major = max(1, int(n_majority * factor)) if n_majority > 0 else 1
            
            # Adaptively adjust minority k-value: increase k for low-density and boundary samples
            if n_minority > 0:
                minor_density_threshold = np.percentile(minor_density, 70) if n_minority > 1 else 0
                minor_low_density_ratio = np.mean(minor_density > minor_density_threshold)
                minor_boundary_ratio = np.mean(minor_is_boundary)
                # Adjust k based on comprehensive ratio (maximum 1.5x magnification)
                k_minor = int(k_base_minor * (1 + 0.5 * (minor_low_density_ratio + minor_boundary_ratio)))
                k_minor = min(k_minor, n_minority - 1)  # Not exceeding sample count - 1
            else:
                k_minor = 1
            
            # Adaptively adjust majority k-value (smaller magnification to avoid excessive noise)
            if n_majority > 0:
                major_density_threshold = np.percentile(major_density, 70) if n_majority > 1 else 0
                major_low_density_ratio = np.mean(major_density > major_density_threshold)
                major_boundary_ratio = np.mean(major_is_boundary)
                k_major = int(k_base_major * (1 + 0.3 * (major_low_density_ratio + major_boundary_ratio)))
                k_major = min(k_major, n_majority - 1)
            else:
                k_major = 1
            
            # Build KNN model with adaptive k-value
            knn_minority = NearestNeighbors(n_neighbors=k_minor, algorithm='ball_tree')
            knn_minority.fit(X_train_minority.values) if n_minority > 0 else None
            
            knn_majority = NearestNeighbors(n_neighbors=k_major, algorithm='ball_tree')
            knn_majority.fit(X_train_majority.values) if n_majority > 0 else None
            
            knn_models[factor] = (knn_minority, knn_majority, k_minor, k_major)
        
        # Training set feature construction 
        for idx in X_train.index:
            point = X_train.loc[idx].values
            for factor in k_factors:
                knn_minority, knn_majority, k_minor, k_major = knn_models[factor]
                
                # Get minority neighbor statistical features
                if n_minority > 0:
                    stats = calculate_neighbor_stats(knn_minority, point, k_minor)
                    A_mean, A_var, A_max, A_median, A_iqr, A_q1, A_q3 = stats
                else:
                    A_mean = A_var = A_max = A_median = A_iqr = A_q1 = A_q3 = 0.0
                
                # Get majority neighbor statistical features
                if n_majority > 0:
                    stats_c = calculate_neighbor_stats(knn_majority, point, k_major)
                    C_mean, C_var, C_max, C_median, C_iqr, C_q1, C_q3 = stats_c
                else:
                    C_mean = C_var = C_max = C_median = C_iqr = C_q1 = C_q3 = 0.0
                
                # Store features
                x_train1.loc[idx, f'A_mean_{factor}'] = A_mean
                x_train1.loc[idx, f'A_var_{factor}'] = A_var
                x_train1.loc[idx, f'A_max_{factor}'] = A_max
                x_train1.loc[idx, f'A_median_{factor}'] = A_median
                x_train1.loc[idx, f'A_iqr_{factor}'] = A_iqr
                x_train1.loc[idx, f'A_q1_{factor}'] = A_q1
                x_train1.loc[idx, f'A_q3_{factor}'] = A_q3
                
                x_train1.loc[idx, f'C_mean_{factor}'] = C_mean
                x_train1.loc[idx, f'C_var_{factor}'] = C_var
                x_train1.loc[idx, f'C_max_{factor}'] = C_max
                x_train1.loc[idx, f'C_median_{factor}'] = C_median
                x_train1.loc[idx, f'C_iqr_{factor}'] = C_iqr
                x_train1.loc[idx, f'C_q1_{factor}'] = C_q1
                x_train1.loc[idx, f'C_q3_{factor}'] = C_q3
        
        # Test set feature construction 
        for idx in X_test.index:
            point = X_test.loc[idx].values
            for factor in k_factors:
                knn_minority, knn_majority, k_minor, k_major = knn_models[factor]
                
                # Get minority neighbor statistical features
                if n_minority > 0:
                    stats = calculate_neighbor_stats(knn_minority, point, k_minor)
                    A_mean, A_var, A_max, A_median, A_iqr, A_q1, A_q3 = stats
                else:
                    A_mean = A_var = A_max = A_median = A_iqr = A_q1 = A_q3 = 0.0
                
                # Get majority neighbor statistical features
                if n_majority > 0:
                    stats_c = calculate_neighbor_stats(knn_majority, point, k_major)
                    C_mean, C_var, C_max, C_median, C_iqr, C_q1, C_q3 = stats_c
                else:
                    C_mean = C_var = C_max = C_median = C_iqr = C_q1 = C_q3 = 0.0
                
                # Store features 
                x_test1.loc[idx, f'A_mean_{factor}'] = A_mean
                x_test1.loc[idx, f'A_var_{factor}'] = A_var
                x_test1.loc[idx, f'A_max_{factor}'] = A_max
                x_test1.loc[idx, f'A_median_{factor}'] = A_median
                x_test1.loc[idx, f'A_iqr_{factor}'] = A_iqr
                x_test1.loc[idx, f'A_q1_{factor}'] = A_q1
                x_test1.loc[idx, f'A_q3_{factor}'] = A_q3
                
                x_test1.loc[idx, f'C_mean_{factor}'] = C_mean
                x_test1.loc[idx, f'C_var_{factor}'] = C_var
                x_test1.loc[idx, f'C_max_{factor}'] = C_max
                x_test1.loc[idx, f'C_median_{factor}'] = C_median
                x_test1.loc[idx, f'C_iqr_{factor}'] = C_iqr
                x_test1.loc[idx, f'C_q1_{factor}'] = C_q1
                x_test1.loc[idx, f'C_q3_{factor}'] = C_q3
        
        # Calculate feature ratios 
        for factor in k_factors:
            # Statistics 
            for stat in ['mean', 'var', 'max', 'median', 'iqr', 'q1', 'q3']:
                a_col = f'A_{stat}_{factor}'
                c_col = f'C_{stat}_{factor}'
                new_col = f'new_{stat}_{factor}'
                
                x_train1[new_col] = x_train1[a_col] / x_train1[c_col].replace(0, np.nan)
                x_test1[new_col] = x_test1[a_col] / x_test1[c_col].replace(0, np.nan)
                
                x_train1[new_col] = x_train1[new_col].fillna(0)
                x_test1[new_col] = x_test1[new_col].fillna(0)
        
        # Filter new features 
        new_features = [f'new_{stat}_{factor}' for factor in k_factors 
                       for stat in ['mean', 'var', 'max', 'median', 'iqr', 'q1', 'q3']]
        x_train1 = x_train1[new_features]
        x_test1 = x_test1[new_features]
        
        # Evaluate classifier (only GNB)
        for name, classifier_factory in classifiers.items():
            classifier = classifier_factory()
            answer = evaluate_classifier(classifier, x_train1, y_train, x_test1, y_test)
            df = pd.DataFrame([answer], columns=experiment_results[name].columns)
            experiment_results[name] = pd.concat([experiment_results[name], df], ignore_index=True)
    
    # Output results in standard format
    print("\nGaussianNB classification metrics:")
    name = 'GaussianNB'
    avg_metrics = experiment_results[name].mean(axis=0)
    print(f"AUC: {avg_metrics['AUC']:.4f}")
    print(f"Gmean: {avg_metrics['Gmean']:.4f}")
    print(f"Fmeasure: {avg_metrics['Fmeasure']:.4f}")

