import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math

X = np.load('X_seq.npy')
y = np.load('y_seq.npy')
scaler_y = joblib.load('scaler_y.pkl')

def build_model():
    model = Sequential([
        LSTM(64, input_shape=(20, 10), return_sequences=False),
        Dropout(0.2),
    Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
def walk_forward_split(X, y, n_splits=5, test_size=0.1):
    n = len(X)
    fold_size = int(n * test_size)
    splits = []
    for i in range(n_splits):
        test_end   = n - i * fold_size
        test_start = test_end - fold_size
        if test_start <= fold_size:
            break
        splits.append((X[:test_start], y[:test_start],
                        X[test_start:test_end], y[test_start:test_end]))
    return splits[::-1]

splits = walk_forward_split(X, y)
fold_rmses = []

es = EarlyStopping(monitor='val_loss', patience=5,
                   restore_best_weights=True)


for fold, (X_tr, y_tr, X_te, y_te) in enumerate(splits):
    print(f"\nFold {fold+1}/5 — train={len(X_tr):,}  test={len(X_te):,}")
    val_cut = int(len(X_tr) * 0.15)
    X_tv, y_tv = X_tr[-val_cut:], y_tr[-val_cut:]
    X_tr, y_tr = X_tr[:-val_cut], y_tr[:-val_cut]

    model = build_model()
    model.fit(X_tr, y_tr,
              validation_data=(X_tv, y_tv),
              epochs=50,
              batch_size=32,
              callbacks=[es],
              verbose=1)

    pred_scaled = model.predict(X_te, verbose=0)
    pred_log = scaler_y.inverse_transform(pred_scaled)
    pred = np.expm1(pred_log)  # undo log1p

    actual_log = scaler_y.inverse_transform(y_te)
    actual = np.expm1(actual_log)  # undo log1p

    rmse = math.sqrt(mean_squared_error(actual, pred))
    
    fold_rmses.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.1f} ms")

model.save('lstm_final.keras')
print(f"\n=== Walk-Forward Results ===")
print(f"Fold RMSEs : {[round(r,1) for r in fold_rmses]}")
print(f"Mean RMSE  : {np.mean(fold_rmses):.1f} ms")
print(f"Std  RMSE  : {np.std(fold_rmses):.1f} ms")
 