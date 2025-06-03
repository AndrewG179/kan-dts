from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from config.hyperparameters import (
    INPUT_LEN, PRED_LEN, MLP_SEARCH, KAN_SEARCH
)
from config.datasets import REAL_DATASETS, SYN_SPECS
from data_pipeline.real_data import load_small_dataset
from data_pipeline.data_creation import load_synthetic
from data_pipeline.preprocessing import create_windows
from data_pipeline.splits import chronological_split
from models.mlp_model import MLP
from models.kan_model import KANWrapper
from models.train_test import train_mlp, train_kan
import torch 


# ---------- helpers ---------------------------------------------------------
def _load_dataset(name: str):
    import pandas as pd

    if name in REAL_DATASETS:
        series = load_small_dataset(name).squeeze()
        return pd.DataFrame({"y": series})

    _, kind, noise = name.split("-", 2)
    return load_synthetic(kind=kind, noise_std=float(noise))


def _split(X, y):
    return chronological_split(X, y)


def _evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(X_te, dtype=torch.float32)).cpu().numpy()
    return mean_squared_error(y_te.squeeze(), preds.squeeze())


# ---------- main loop -------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        REAL_DATASETS +
        [f"syn-{kind}-{noise}" for (kind, noise) in SYN_SPECS]
    )

    rows = []

    for ds_name in datasets:
        df = _load_dataset(ds_name)
        X, y = create_windows(df, input_len=INPUT_LEN, pred_len=PRED_LEN)

        # ---------- MLP sweep ----------
        for hp in MLP_SEARCH:
            (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _split(X, y)

            mlp = MLP([INPUT_LEN, *hp["hidden"], PRED_LEN])
            _, _, trained = train_mlp(
                mlp, X_tr, y_tr, X_val, y_val,
                epochs=hp["epochs"], lr=hp["lr"]
            )
            mse = _evaluate(trained, X_te, y_te)

            rows.append({
                "dataset": ds_name,
                "model":   "mlp",
                "mse":     mse,
                "params":  hp
            })

        # ---------- KAN sweep ----------
        for hp in KAN_SEARCH:
            (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _split(X, y)

            kan = KANWrapper(width=hp["width"],
                             grid=hp["grid"],
                             k=hp["k"])
            _, _, trained = train_kan(
                kan, X_tr, y_tr, X_val, y_val,
                steps=20
            )
            mse = _evaluate(trained, X_te, y_te)

            rows.append({
                "dataset": ds_name,
                "model":   "kan",
                "mse":     mse,
                "grid":    hp["grid"],
                "width":   hp["width"],
                "k":       hp["k"]
            })

    # ---------- save structured results -------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    # one *_BEST.json per (dataset, model)
    for (ds, mdl), grp in df.groupby(["dataset", "model"]):
        best = grp.sort_values("mse").iloc[0].to_dict()
        with open(out_dir / f"{ds}_{mdl}_BEST.json", "w") as f:
            json.dump(best, f, indent=2)
    import matplotlib.pyplot as plt

    # 1) BEST result per (dataset, model)  ────────────────────────────────
    best_df = (
        df.groupby(["dataset", "model"])["mse"]
          .min()                       # best (lowest) MSE
          .unstack()                  # cols = model, rows = dataset
          .sort_index()               # keep datasets in alpha order
    )

    ax = best_df.plot(kind="bar", rot=45)
    ax.set_ylabel("Test MSE  ↓ better")
    ax.set_title("KAN vs MLP — best result on every dataset")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_results_best.png", dpi=300)
    plt.close()

    # 2) *All* runs at once  ──────────────────────────────────────────────
    fig, ax = plt.subplots()
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        ax.scatter(subset["dataset"], subset["mse"],
                   label=model, alpha=0.6, s=40)

    ax.set_ylabel("Test MSE")
    ax.set_title("All hyper-parameter sweeps")
    ax.legend(title="Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_results_all.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()