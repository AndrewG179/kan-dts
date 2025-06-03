# ---------- constant values used in final experiment ----------
INPUT_LEN  = 100
PRED_LEN   = 14
EPOCHS     = 200
LR         = 1e-3

# final production architectures
MLP_WIDTH  = [INPUT_LEN, 128, 128, 128, PRED_LEN]
KAN_WIDTH  = [INPUT_LEN,   5,   5, PRED_LEN]
KAN_GRID   = 5
KAN_K      = 3           # spline order
KAN_OPT    = "LBFGS"

# ---------- search spaces ----------
from sklearn.model_selection import ParameterGrid

MLP_SEARCH = ParameterGrid({
    "hidden": [[64,64], [128,128,128]],
    "lr":     [1e-3, 1e-4],
    "epochs": [100, 200],
})

KAN_SEARCH = ParameterGrid({
    "grid":  [5, 10],
    "width": [
        [INPUT_LEN, 5,  5,  PRED_LEN],
        [INPUT_LEN, 10, 10, PRED_LEN]
    ],
    "k":     [3],
    "opt":   ["Adam"],
})