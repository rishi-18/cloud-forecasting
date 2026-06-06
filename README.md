# Cloud Cost & Execution Time Forecasting for Serverless Workloads

## Overview

This project explores the prediction of serverless function execution duration using machine learning and deep learning techniques. The objective is to forecast execution time based on infrastructure configuration, workload characteristics, and historical execution metrics, enabling better cloud cost optimization and resource allocation decisions.

The project includes:

* Data cleaning and preprocessing
* Feature engineering with lag and rolling-window statistics
* Sequence generation for LSTM models
* Walk-forward validation
* Model comparison using LSTM, Linear Regression, and XGBoost
* Performance analysis and interpretation

---

## Problem Statement

Serverless platforms automatically manage infrastructure resources, but execution duration directly impacts operational cost and performance.

The goal of this project is to predict:

* **Execution Duration (ms)**

using features such as:

* Memory allocation
* Cold start status
* Invocation count
* Error rate
* Runtime environment
* Function architecture
* Historical execution behavior

---

## Dataset

The dataset simulates serverless computing environments and contains metrics related to cloud function executions.

### Key Features

| Feature               | Description                                           |
| --------------------- | ----------------------------------------------------- |
| memory_mb             | Allocated memory for function                         |
| cold_start            | Indicates whether invocation experienced a cold start |
| error_rate            | Function error rate                                   |
| invocations_count     | Number of invocations                                 |
| architecture          | CPU architecture                                      |
| runtime_environment   | Runtime used by the function                          |
| optimization_strategy | Applied optimization strategy                         |
| function_name         | Function identifier                                   |
| total_cost_usd        | Execution cost                                        |
| execution_duration_ms | Target variable                                       |

---

## Project Structure

```text
project/
│
├── serverless_architecture_metrics.csv
├── serverless_cleaned.csv
│
├── 01_data_cleaning.py
├── 02_feature_engineering.py
├── 03_model.py
│
├── X_seq.npy
├── y_seq.npy
│
├── scaler_X.pkl
├── scaler_y.pkl
│
├── lstm_final.keras
│
└── README.md
```

---

## Data Preprocessing

### Cleaning

* Removed invalid records
* Handled missing values
* Removed extreme anomalies and unrealistic execution times
* Generated cleaned dataset

### Feature Engineering

Categorical variables were encoded using Label Encoding.

Historical features were created:

```python
duration_lag1
duration_lag2
duration_lag3

duration_roll_mean_10
duration_roll_std_10
```

### Log Transformation

To reduce the impact of skewed execution durations:

```python
execution_duration_ms = log1p(execution_duration_ms)
```

The same transformation was applied to lag and rolling-duration features before scaling.

### Scaling

Features and targets were normalized using:

```python
MinMaxScaler
```

---

## LSTM Model

### Architecture

```text
Input Sequence (20 timesteps × 10 features)
        ↓
LSTM (64 units)
        ↓
Dropout (0.2)
        ↓
Dense (16, ReLU)
        ↓
Dense (1)
```

### Training Strategy

Walk-forward validation was used to simulate real-world forecasting.

```text
Training Window → Validation → Future Test Window
```

This prevents information leakage from future observations.

---

## Model Evaluation

### LSTM Results

```text
Fold RMSEs:
1728.2
1676.4
1707.5
1521.6
1683.4

Mean RMSE: 1663.4 ms
Std RMSE : 73.2 ms
```

### Linear Regression Baseline

```text
RMSE: 1114.87 ms
```

### XGBoost Model

```text
RMSE: 100.49 ms
```

---

## Key Findings

### Correlation Analysis

Strongest relationships with execution duration:

```text
memory_mb             -0.66
duration_roll_mean     0.33
duration_roll_std      0.27
cold_start             0.18
```

Lag features showed very weak correlation:

```text
duration_lag1          0.004
duration_lag2          0.015
duration_lag3          0.013
```

### Interpretation

The analysis revealed that execution duration is primarily influenced by:

* Memory allocation
* Cold starts
* Resource configuration

rather than temporal dependencies.

As a result:

* LSTM had limited sequential patterns to learn.
* Traditional regression methods performed better.
* XGBoost significantly outperformed both Linear Regression and LSTM.

---

## Technologies Used

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Model Persistence

* Joblib

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow xgboost joblib
```

### Step 1: Data Cleaning

```bash
python 01_data_cleaning.py
```

### Step 2: Feature Engineering

```bash
python 02_feature_engineering.py
```

Generates:

```text
X_seq.npy
y_seq.npy
scaler_X.pkl
scaler_y.pkl
```

### Step 3: Train LSTM

```bash
python 03_model.py
```

Generates:

```text
lstm_final.keras
```

---

## Future Improvements

* Hyperparameter optimization
* Attention-based sequence models
* Transformer architectures
* Cost forecasting alongside duration forecasting
* Multi-target prediction
* Real-world cloud monitoring integration
* Automated retraining pipeline

---

## Conclusion

This project demonstrates the complete machine learning workflow for cloud workload forecasting, from data preprocessing and feature engineering to model evaluation and comparison.

Experimental results showed that execution duration in the dataset is primarily driven by infrastructure characteristics rather than temporal dependencies. Consequently, tree-based models such as XGBoost achieved superior performance compared to sequence-based deep learning models like LSTM.
