# ---------- constant values used in final experiment ----------
INPUT_LEN  = 100
PRED_LEN   = 14
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
        [10],
        [5,  5],
        [10, 10]
    ],
    "epochs": [100],
    "k":     [3],
    "opt":   ["Adam"],
})