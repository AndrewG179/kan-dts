from sklearn.model_selection import train_test_split
import numpy as np

def random_split(X, y, test_ratio=0.2, seed=42):
    """Order-agnostic split – keep for non-time-series baselines."""
    return train_test_split(X, y, test_size=test_ratio, random_state=seed)

def chronological_split(X, y, eval_ratio=0.1, test_ratio=0.1):
    """Deterministic 80/10/10 split that respects temporal order."""
    n      = len(X)
    n_test = int(n * test_ratio)
    n_eval = int(n * eval_ratio)
    n_train = n - n_eval - n_test
    return (
        (X[:n_train],  y[:n_train]),
        (X[n_train:n_train+n_eval], y[n_train:n_train+n_eval]),
        (X[-n_test:],  y[-n_test:])
    )