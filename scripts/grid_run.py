from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config.hyperparameters import INPUT_LEN, PRED_LEN, MLP_SEARCH, KAN_SEARCH
from config.datasets import REAL_DATASETS, SYN_SPECS
from data_pipeline.real_data import load_small_dataset
from data_pipeline.data_creation import load_synthetic
from data_pipeline.preprocessing import create_windows
from data_pipeline.splits import chronological_split
from models.mlp_model import MLP
from kan import KAN
from models.train_test import train_mlp, train_kan


# ---------- Helpers ---------------------------------------------------------

def _load_dataset(name: str):
    if name in REAL_DATASETS:
        series = load_small_dataset(name).squeeze()
        return pd.DataFrame({"y": series})
    _, kind, noise = name.split("-", 2)
    return load_synthetic(kind=kind, noise_std=float(noise))

def _normalize(X, y):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    y_mean = y.mean()
    y_std = y.std() + 1e-6
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def _evaluate(model, X_te, y_te, y_mean, y_std):
    if hasattr(model, "eval") and callable(model.eval):
        model.eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(X_te, dtype=torch.float32)).cpu().numpy()
    preds = preds * y_std + y_mean
    return mean_squared_error(y_te.squeeze(), preds.squeeze())

def save_kan_activations(kan_model, save_path: Path):
    kan_model.plot(
        folder=str(save_path),
        beta=3.0,
        scale=0.5,
        varscale=1.0
    )

def run_mlp_search(X, y, y_mean, y_std, ds_name):
    results = []
    for hp in MLP_SEARCH:
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(X, y)
        mlp = MLP([INPUT_LEN, *hp["hidden"], PRED_LEN])
        print(f"Running {ds_name} | mlp | {hp}")
        _, _, trained = train_mlp(mlp, X_tr, y_tr, X_val, y_val, epochs=hp["epochs"], lr=hp["lr"])
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({"dataset": ds_name, "model": "mlp", "mse": mse, "params": hp})
    return results


def run_kan_search(X, y, y_mean, y_std, ds_name):
    results = []
    for hp in KAN_SEARCH:
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(X, y)
        kan = KAN(
            width=[INPUT_LEN, *hp["hidden"], PRED_LEN],
            grid=hp["grid"],
            k=hp["k"],
            symbolic_enabled=False
        )
        print(f"Running {ds_name} | kan | {hp}")
        _, _, trained = train_kan(
            kan, X_tr, y_tr, X_val, y_val,
            steps=hp["epochs"],
            lr=hp["lr"],
            lamb_l1=hp["lamb_l1"],
            lamb_entropy=hp["lamb_entropy"]
        )
        trained.prune()
        # ------------Saving splines (takes a lot of time)--------------
        # hp_str = (
        #     f"epochs_{hp['epochs']}_grid_{hp['grid']}_k_{hp['k']}_hidden_" +
        #     "_".join(map(str, hp['hidden']))
        # )
        # save_path = out_dir / "activations" / ds_name / hp_str
        # save_kan_activations(trained, save_path)
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({"dataset": ds_name, "model": "kan", "mse": mse, "params": hp})
    return results


def save_best_results(df, out_dir):
    for (ds, mdl), grp in df.groupby(["dataset", "model"]):
        best = grp.sort_values("mse").iloc[0].to_dict()
        with open(out_dir / f"{ds}_{mdl}_BEST.json", "w") as f:
            json.dump(best, f, indent=2)


def plot_results(df, out_dir):
    best_df = df.groupby(["dataset", "model"])["mse"].min().unstack().sort_index()

    ax = best_df.plot(kind="bar", rot=45, log=True)
    ax.set_ylabel("Val MSE  ↓ better")
    ax.set_title("KAN vs MLP — best result on every dataset")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_results_best.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        ax.scatter(subset["dataset"], subset["mse"], label=model, alpha=0.6, s=40)
    ax.set_ylabel("Val MSE")
    ax.set_title("All hyper-parameter sweeps")
    ax.set_yscale("log")
    ax.legend(title="Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_results_all.png", dpi=300)
    plt.close()

    ratio = (best_df["kan"] / best_df["mlp"]).rename("KAN ÷ MLP")
    plt.figure(figsize=(6, len(ratio) * 0.4))
    sns.heatmap(ratio.to_frame().T, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar=False)
    plt.yticks(rotation=0)
    plt.title("Relative val MSE (KAN ÷ MLP)")
    plt.tight_layout()
    plt.savefig(out_dir / "grid_results_ratio_heatmap.png", dpi=300)
    plt.close()


# ---------- Main Script -----------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = REAL_DATASETS + [f"syn-{kind}-{noise}" for (kind, noise) in SYN_SPECS]
    all_results = []

    for ds_name in datasets:
        df = _load_dataset(ds_name)
        X, y = create_windows(df, input_len=INPUT_LEN, pred_len=PRED_LEN)
        X, y, X_mean, X_std, y_mean, y_std = _normalize(X, y)

        all_results += run_mlp_search(X, y, y_mean, y_std, ds_name)
        all_results += run_kan_search(X, y, y_mean, y_std, ds_name)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    save_best_results(df, out_dir)
    plot_results(df, out_dir)


if __name__ == "__main__":
    main()