# ---------- constant values used in final experiment ----------
INPUT_LEN  = 10
PRED_LEN   = 2
EPOCHS     = 200
LR         = 1e-3

KAN_OPT="Adam"

# ---------- search spaces ----------
from sklearn.model_selection import ParameterGrid

MLP_SEARCH = ParameterGrid({
    "hidden": [
        [64,64],
        [128,128,128]
    ],
    "lr":     [1e-3, 1e-4],
    "epochs": [100, 200],
})

KAN_SEARCH = ParameterGrid({
    "grid":  [5, 10],
    "hidden": [
        [2],
        [10],
        [2, 2],
        [5, 5]
    ],
    "epochs": [100],
    "k":     [3],
    "opt":   ["Adam"],
    "lr":    [1e-2, 1e-3],
    "lamb_l1": [0.0, 1e-4, 1e-2],
    "lamb_entropy": [0.0, 1e-4, 1e-2],
})

LSTM_SEARCH = ParameterGrid({
    "hidden_size": [32, 64],
    "num_layers": [1],
    "lr": [1e-3, 1e-4],
    "epochs": [100],
    "dropout": [0.0]
})