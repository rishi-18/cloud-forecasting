import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('serverless_architecture_metrics.csv')

# Reconstruct timestamps evenly across 90 days
start = datetime(2024, 1, 1)
end   = datetime(2024, 3, 31)
df['timestamp'] = pd.date_range(start=start, end=end, periods=len(df))
df = df.sort_values('timestamp').reset_index(drop=True)

#The max value is 900,000ms (15 minutes) — these are Lambda timeout events, not real execution times. Mean is 1,443ms but median is only 431ms. These ~200 rows would completely distort what the LSTM learns. Cap them at the 99th percentile
cap = df['execution_duration_ms'].quantile(0.99)
df['execution_duration_ms'] = df['execution_duration_ms'].clip(upper=cap)

# Verify
print(f"Before: max={900000}, mean=1443")
print(f"After:  max={cap:.0f}, mean={df['execution_duration_ms'].mean():.0f}")


#Four representations exist: 'TRUE', 'FALSE', '1', '0'. All as strings. Must normalize to integer 0/1 before feeding into the model.
df['cold_start'] = (
    df['cold_start']
      .str.upper()
      .map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0})
      .astype(int)
)

print(df['cold_start'].value_counts())  # should show only 0 and 1


#missing values
df['optimization_strategy'] = df['optimization_strategy'].fillna('None')
df['runtime_environment']   = df['runtime_environment'].fillna(
    df['runtime_environment'].mode()[0]
)

print(df.isnull().sum())  # should show all zeros


df.to_csv('serverless_cleaned.csv', index=False)
print("Saved.")