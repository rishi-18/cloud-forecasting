import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv("serverless_cleaned.csv")

# Feature engineering
df['duration_lag1'] = df['execution_duration_ms'].shift(1)
df['duration_lag2'] = df['execution_duration_ms'].shift(2)
df['duration_lag3'] = df['execution_duration_ms'].shift(3)

df['duration_roll_mean_10'] = df['execution_duration_ms'].rolling(10).mean()
df['duration_roll_std_10'] = df['execution_duration_ms'].rolling(10).std()

df.dropna(inplace=True)

# Encode categoricals if not already encoded
cat_cols = [
    'architecture',
    'runtime_environment',
    'optimization_strategy',
    'function_name'
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

features = [
    'memory_mb',
    'cold_start',
    'error_rate',
    'invocations_count',
    'duration_lag1',
    'duration_lag2',
    'duration_lag3',
    'duration_roll_mean_10',
    'duration_roll_std_10'
]

X = df[features]
y = df['execution_duration_ms']

# Time-aware split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, pred))
print(f"XGBoost RMSE: {rmse:.2f} ms")