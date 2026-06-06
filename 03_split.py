import numpy as np
import joblib
X = np.load('X_seq.npy')
y = np.load('y_seq.npy')

def walk_forward_split(X, y, n_splits=5, test_size=0.1):
    n = len(X)
    fold_size = int(n * test_size)
    splits = []
    for i in range(n_splits):
        test_end   = n - i * fold_size
        test_start = test_end - fold_size
        if test_start <= fold_size:
            break
        X_tr, y_tr = X[:test_start], y[:test_start]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        splits.append((X_tr, y_tr, X_te, y_te))
    return splits[::-1]  # chronological order

splits = walk_forward_split(X, y, n_splits=5, test_size=0.1)
for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
    print(f"Fold {i+1}: train={Xtr.shape[0]:,}  test={Xte.shape[0]:,}")