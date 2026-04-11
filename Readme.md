
# ANSR: Adaptive Neighbor Statistical Ratio Feature Construction Algorithm

ANSR (Adaptive Neighbor Statistical Ratio) is a feature-construction algorithm designed for **imbalanced data classification** with Gaussian Naive Bayes (GNB). It constructs adaptive neighbor statistical ratio features to strengthen class discriminability while preserving Gaussian compatibility and model interpretability.

---

## Features
- Adaptive neighborhood size based on **local density** and **boundary detection**
- Constructs **statistical ratio features** (mean, var, max, median, IQR, Q1, Q3)
- Compatible with Gaussian Naive Bayes and lightweight classifiers
- Supports direct evaluation with common imbalanced learning metrics


---

## Requirements
```
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.2.0
imbalanced-learn >= 0.10.0
matplotlib >= 3.4.0
scipy >= 1.7.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Format
- Place your CSV dataset in the root directory
- The label column must be named **class**
- Example: `shuttle-2_vs_5.csv`

---

## Usage
### Run ANSR with Default Parameters
```bash
python ansr.py
```

### Customization
Modify parameters in `ansr.py`:
```python
DATASET_FILE = 'your_dataset.csv'
TEST_SIZE = 0.2          # Train:Test = 8:2
OPTIMAL_K_FACTORS = [0.3]
OPTIMAL_DENSITY_K = 8
OPTIMAL_BOUNDARY_PERCENTILE = 70
```

---

## Evaluation Metrics
The code reports 8 metrics for imbalanced classification:
- AUC: Overall discriminative ability
- G-mean: Balance between majority and minority performance
- Recall: Minority-class detection rate
- Precision: Minority-class prediction accuracy
- F1: F1-score for minority class
- MCC: Matthews Correlation Coefficient (global consistency)
- Specificity: True Negative Rate (majority identification)
- NPV: Negative Predictive Value

---

## Project Structure
```
ANSR/
├── ansr.py                # Main implementation of ANSR algorithm
├── README.md             # Project description and user guide
├── requirements.txt      # Dependencies
├── *.csv                 # Imbalanced datasets (user-provided)
└── results/              # Output figures and logs (optional)
```

---

## Method Overview
1. Compute **local density** for each sample
2. Identify **boundary samples** using distance ratio
3. Dynamically adjust neighborhood size for minority/majority classes
4. Compute 7 neighbor statistics for both classes
5. Construct **statistical ratio features**
6. Train and evaluate Gaussian Naive Bayes classifier
