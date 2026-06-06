import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataframe (expects a CSV named 'data.csv' in the working directory)
DATA_PATH = 'serverless_cleaned.csv'
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please provide the dataset as '{DATA_PATH}'")

cat_cols = ['architecture', 'runtime_environment',
            'optimization_strategy', 'function_name']

le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

# Result: architecture_enc (0=arm64, 1=x86_64), etc.

df['duration_lag1'] = df['execution_duration_ms'].shift(1)
df['duration_lag2'] = df['execution_duration_ms'].shift(2)
df['duration_lag3'] = df['execution_duration_ms'].shift(3)
df['cost_lag1']     = df['total_cost_usd'].shift(1)

df['duration_roll_mean_10'] = df['execution_duration_ms'].rolling(10).mean()
df['duration_roll_std_10']  = df['execution_duration_ms'].rolling(10).std()
df['cost_roll_mean_10']     = df['total_cost_usd'].rolling(10).mean()

df.dropna(inplace=True)  # removes first 10 rows with NaN from rolling

print("\n=== TARGET DISTRIBUTION ===")
print(df['execution_duration_ms'].describe())

print("\n=== PERCENTILES ===")
print(df['execution_duration_ms'].quantile([0.5, 0.9, 0.95, 0.99, 0.999]))

print("\n=== CORRELATIONS ===")
corr = df.corr(numeric_only=True)
print(corr['execution_duration_ms'].sort_values(ascending=False))

# ---------------------------------------------------------
# LOG TRANSFORM BEFORE SCALING
# ---------------------------------------------------------
import numpy as np

df['execution_duration_ms'] = np.log1p(df['execution_duration_ms'])

# Also log-transform lag features so they stay consistent
df['duration_lag1'] = np.log1p(df['duration_lag1'])
df['duration_lag2'] = np.log1p(df['duration_lag2'])
df['duration_lag3'] = np.log1p(df['duration_lag3'])
df['duration_roll_mean_10'] = np.log1p(df['duration_roll_mean_10'])
df['duration_roll_std_10']  = np.log1p(df['duration_roll_std_10'])
# ---------------------------------------------------------

feature_cols = [
    'memory_mb', 'cold_start', 'error_rate', 'invocations_count',
    'architecture_enc', 'duration_lag1', 'duration_lag2',
    'duration_lag3', 'duration_roll_mean_10', 'duration_roll_std_10'
]
target_col = 'execution_duration_ms'

from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# Save scalers — you'll need them at inference time
import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

def create_sequences(X, y, window=20):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=50)

# print(f"X shape: {X_seq.shape}")  # (19961, 20, 10)
# print(f"y shape: {y_seq.shape}")  # (19961, 1)

# print("=== SHAPE CHECK ===")
# print(f"X_seq shape: {X_seq.shape}")   # expect (N, window, 10)
# print(f"y_seq shape: {y_seq.shape}")   # expect (N, 1)

# print("\n=== SCALE CHECK ===")
# print(f"X_seq min: {X_seq.min():.4f}")  # expect ~0.0
# print(f"X_seq max: {X_seq.max():.4f}")  # expect ~1.0
# print(f"y_seq min: {y_seq.min():.4f}")  # expect ~0.0
# print(f"y_seq max: {y_seq.max():.4f}")  # expect ~1.0

# print("\n=== NaN CHECK ===")
# print(f"NaNs in X: {np.isnan(X_seq).sum()}")  # expect 0
# print(f"NaNs in y: {np.isnan(y_seq).sum()}")  # expect 0

# print("\n=== SAMPLE CHECK ===")
# print(f"First window (row 0), first feature col:\n{X_seq[0, :, 0]}")

# print(f"Rows after dropna: {len(df)}")
# print(f"Sequences = {len(df)} - window(20) = {len(df) - 20}")

np.save('X_seq.npy', X_seq)
np.save('y_seq.npy', y_seq)
print("Saved.")

# Verify files load back correctly
sX = joblib.load('scaler_X.pkl')
sY = joblib.load('scaler_y.pkl')
X  = np.load('X_seq.npy')
y  = np.load('y_seq.npy')

print(sX)               # MinMaxScaler object
print(X.shape, y.shape) # (7601, 20, 10) and (7601, 1)