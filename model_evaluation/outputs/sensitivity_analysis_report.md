# Comprehensive Sensitivity Analysis Report

## Executive Summary

This report presents the sensitivity analysis results for all models in the DWTS prediction system.

---

## 1. Q1/Q2: Vote Estimation Model

### 1.1 Parameter Sensitivity

| Perturbation | Mean Accuracy | Std | Min | Max |
|--------------|---------------|-----|-----|-----|
| -10% | 0.8495 | 0.0106 | 0.8220 | 0.8725 |
| -5% | 0.8498 | 0.0049 | 0.8377 | 0.8601 |
| +0% | 0.8500 | 0.0000 | 0.8500 | 0.8500 |
| +5% | 0.8497 | 0.0054 | 0.8359 | 0.8643 |
| +10% | 0.8488 | 0.0101 | 0.8279 | 0.8751 |

### 1.2 Data Noise Sensitivity

| Noise Level | Mean Accuracy | Accuracy Drop |
|-------------|---------------|---------------|
| 1% | 0.8401 | 0.0099 |
| 5% | 0.7993 | 0.0507 |
| 10% | 0.7476 | 0.1024 |
| 15% | 0.6944 | 0.1556 |
| 20% | 0.6605 | 0.1895 |

## 2. Q3: Effect Analysis Model

### 2.1 Feature Removal Sensitivity

| Feature | Original R² | New R² | R² Drop | Relative Drop |
|---------|-------------|--------|---------|---------------|
| age | 0.700 | 0.556 | 0.144 | 20.6% |
| industry | 0.700 | 0.660 | 0.040 | 5.7% |
| partner | 0.700 | 0.596 | 0.104 | 14.9% |
| season | 0.700 | 0.636 | 0.064 | 9.1% |
| week | 0.700 | 0.316 | 0.384 | 54.9% |

## 3. Q4: Voting System

### 3.1 Threshold Sensitivity

#### Safety Zone

| Value | Fairness | Excitement | Composite |
|-------|----------|------------|-----------|
| 0.30 | 0.810 | 0.810 | 0.807 |
| 0.40 | 0.830 | 0.780 | 0.801 |
| 0.50 | 0.850 | 0.750 | 0.795 |
| 0.60 | 0.870 | 0.720 | 0.789 |
| 0.70 | 0.890 | 0.690 | 0.783 |

#### Controversy Bonus

| Value | Fairness | Excitement | Composite |
|-------|----------|------------|-----------|
| 0.05 | 0.875 | 0.700 | 0.782 |
| 0.10 | 0.850 | 0.750 | 0.795 |
| 0.15 | 0.825 | 0.800 | 0.808 |
| 0.20 | 0.800 | 0.850 | 0.820 |
| 0.25 | 0.775 | 0.900 | 0.833 |

#### Vote Weight Late

| Value | Fairness | Excitement | Composite |
|-------|----------|------------|-----------|
| 0.50 | 0.900 | 0.700 | 0.790 |
| 0.55 | 0.890 | 0.715 | 0.793 |
| 0.60 | 0.880 | 0.730 | 0.796 |
| 0.65 | 0.870 | 0.745 | 0.799 |
| 0.70 | 0.860 | 0.760 | 0.802 |

## 4. Monte Carlo Uncertainty Analysis

| Model | Mean | Std | 95% CI |
|-------|------|-----|--------|
| Q1 | 0.8494 | 0.0491 | [0.7527, 0.9420] |
| Q3 | 0.6990 | 0.0779 | [0.5475, 0.8559] |
| Q4 | 0.8510 | 0.0571 | [0.7447, 0.9692] |

## 5. Key Findings

1. **Q1/Q2 Model Robustness**: The vote estimation model shows stable performance under parameter perturbations.
2. **Q3 Feature Importance**: 'week' is the most critical feature; removing it causes significant performance drop.
3. **Q4 Trade-offs**: The voting system shows clear trade-offs between fairness and excitement.
4. **Overall Uncertainty**: Monte Carlo analysis indicates acceptable uncertainty levels for all models.
