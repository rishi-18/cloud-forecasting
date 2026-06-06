import pandas as pd

# df = pd.read_csv("serverless_cleaned.csv")

# print(df['execution_duration_ms'].describe())

# print(df['execution_duration_ms'].quantile([0.5,0.9,0.95,0.99,0.999]))

# corr = df.corr(numeric_only=True)
# print(corr['execution_duration_ms'].sort_values(ascending=False))

import pandas as pd

df = pd.read_csv("serverless_cleaned.csv")

print(df[['memory_mb',
          'cold_start',
          'execution_duration_ms']]
      .corr())
# print(df['execution_duration_ms'].value_counts().head(20))

# df['duration_lag1'] = df['execution_duration_ms'].shift(1)
# df['duration_lag2'] = df['execution_duration_ms'].shift(2)
# df['duration_lag3'] = df['execution_duration_ms'].shift(3)

# df['duration_roll_mean_10'] = df['execution_duration_ms'].rolling(10).mean()
# df['duration_roll_std_10']  = df['execution_duration_ms'].rolling(10).std()

# df.dropna(inplace=True)

# corr = df.corr(numeric_only=True)

# print(
#     corr['execution_duration_ms']
#     .sort_values(ascending=False)
# )