import numpy as np

def create_windows(df, input_len=30, pred_len=1):
    X, y = [], []
    for i in range(len(df) - input_len - pred_len + 1):
        X.append(df["y"].values[i:i+input_len])
        y.append(df["y"].values[i+input_len:i+input_len+pred_len])
    return np.array(X), np.array(y)