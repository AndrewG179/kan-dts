from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config.hyperparameters import INPUT_LEN, PRED_LEN, MLP_SEARCH, KAN_SEARCH, LSTM_SEARCH
from config.datasets import REAL_DATASETS, SYN_SPECS
from data_pipeline.real_data import load_small_dataset
from data_pipeline.data_creation import load_synthetic
from data_pipeline.preprocessing import create_windows
from data_pipeline.splits import chronological_split
from models.mlp_model import MLP
from kan import KAN
from models.train_test import train_mlp, train_kan, train_lstm
from models.lstm_model import LSTMModel

import shutil

#delete splines
# splines_dir = Path(__file__).resolve().parents[1] / "results" / "batch" / "splines"
# if splines_dir.exists():
#     shutil.rmtree(splines_dir)  # 🔥 Delete everything inside
# splines_dir.mkdir(parents=True)
# ---------- Helpers ---------------------------------------------------------


"""#Give me a random baseline model so I can see your model increasing 
#Do some kind of comprehensive test looking at noise increases 
Plot the training curve over your epochs 
training loss and validation loss.
Use MAE as well for a second metric 
Plot residuals 
#Plot splines 
Needs to be in a geniune report. With figures 
Have your methods before (turn this in)
Use the neurips latex."""

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

def run_mlp_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name):
    results = []
    for hp in MLP_SEARCH:
        #optimize
        mlp = MLP([INPUT_LEN, *hp["hidden"], PRED_LEN])
        print(f"Running {ds_name} | mlp | {hp}")
        tr_losses, val_losses, trained = train_mlp(mlp, X_tr, y_tr, X_val, y_val, epochs=hp["epochs"], lr=hp["lr"])
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({
            "dataset": ds_name, "model": "mlp", "mse": mse, "params": hp,
            "tr_losses": tr_losses, "val_losses": val_losses
        })
    return results


def run_kan_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name, out_dir=None):
    results = []
    for hp in KAN_SEARCH:
        kan = KAN(
            width=[INPUT_LEN, *hp["hidden"], PRED_LEN],
            grid=hp["grid"],
            k=hp["k"],
            symbolic_enabled=False
        )
        print(f"Running {ds_name} | kan | {hp}")
        tr_losses, val_losses, trained = train_kan(
            kan, X_tr, y_tr, X_val, y_val,
            steps=hp["epochs"],
            lr=hp["lr"],
            lamb_l1=hp["lamb_l1"],
            lamb_entropy=hp["lamb_entropy"]
        )
        trained.prune()
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({
            "dataset": ds_name, "model": "kan", "mse": mse, "params": hp,
            "tr_losses": tr_losses, "val_losses": val_losses
        })
        #plot splines
        # splines_dir = Path(out_dir) / "splines"
        # splines_dir.mkdir(parents=True, exist_ok=True)
        # hp_str = f"epochs_{hp['epochs']}_grid_{hp['grid']}_k_{hp['k']}_hidden_" + "_".join(map(str, hp['hidden']))
        # plot_path = splines_dir / f"{ds_name}_{hp_str}"
        # trained(torch.as_tensor(X_te, dtype=torch.float32))
        # trained.plot(folder=str(plot_path), beta=1)
        # plt.savefig(str(plot_path / "combined_splines.png"))
        # plt.close()
    return results

def run_lstm_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name):
    results = []
    # LSTM expects input shape: (batch, seq_len, input_size)
    # Our X is (N, input_len), reshape to (N, input_len, 1)
    X_tr_lstm = X_tr[..., None]
    X_val_lstm = X_val[..., None]
    X_te_lstm = X_te[..., None]
    for hp in LSTM_SEARCH:
        lstm = LSTMModel(
            input_size=1,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            output_size=PRED_LEN,  # use PRED_LEN from config
            dropout=hp["dropout"]
        )
        print(f"Running {ds_name} | lstm | {hp}")
        tr_losses, val_losses, trained = train_lstm(
            lstm, X_tr_lstm, y_tr, X_val_lstm, y_val,
            epochs=hp["epochs"], lr=hp["lr"]
        )
        # Evaluate
        mse = _evaluate(trained, X_te_lstm, y_te, y_mean, y_std)
        results.append({
            "dataset": ds_name, "model": "lstm", "mse": mse, "params": hp,
            "tr_losses": tr_losses, "val_losses": val_losses
        })
    return results

def save_best_results(df, out_dir):
    # Create separate folder for raw results
    raw_results_dir = out_dir / "raw_results"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    
    for (ds, mdl), grp in df.groupby(["dataset", "model"]):
        best = grp.sort_values("mse").iloc[0].to_dict()
        with open(raw_results_dir / f"{ds}_{mdl}_BEST.json", "w") as f:
            json.dump(best, f, indent=2)


def plot_results(df, out_dir):
    best_df = df.groupby(["dataset", "model"])["mse"].min().unstack().sort_index()

    ax = best_df.plot(kind="bar", rot=45, log=True)
    ax.set_ylabel("Val MSE  ↓ better")
    ax.set_title("KAN vs MLP vs LSTM — best result on every dataset")
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

    # Only plot ratio heatmap if both KAN and MLP are present
    if "kan" in best_df.columns and "mlp" in best_df.columns:
        ratio = (best_df["kan"] / best_df["mlp"]).rename("KAN ÷ MLP")
        plt.figure(figsize=(6, len(ratio) * 0.4))
        sns.heatmap(ratio.to_frame().T, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar=False)
        plt.yticks(rotation=0)
        plt.title("Relative val MSE (KAN ÷ MLP)")
        plt.tight_layout()
        plt.savefig(out_dir / "grid_results_ratio_heatmap.png", dpi=300)
        plt.close()

def plot_training_curves(df, out_dir):
    """Plot training curves using already collected loss data."""
    
    # Create separate folder for training graphs
    training_graphs_dir = out_dir / "training_graphs"
    training_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Select datasets to plot: all real + one of each synthetic type
    real_datasets = [ds for ds in df['dataset'].unique() if ds in REAL_DATASETS]
    synthetic_datasets = [ds for ds in df['dataset'].unique() if ds.startswith('syn-')]
    
    # Get one representative from each synthetic type
    trend_only = [ds for ds in synthetic_datasets if 'trend_only' in ds][0]
    season_only = [ds for ds in synthetic_datasets if 'season_only' in ds][0]
    trend_season = [ds for ds in synthetic_datasets if 'trend_season' in ds][0]
    
    selected_datasets = real_datasets + [trend_only, season_only, trend_season]
    
    print(f"Plotting training curves for: {selected_datasets}")
    
    for ds_name in selected_datasets:
        ds_data = df[df['dataset'] == ds_name]
        
        # Create subplot for each model
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Training Curves - {ds_name}', fontsize=16)
        
        for idx, model in enumerate(['mlp', 'kan', 'lstm']):
            model_data = ds_data[ds_data['model'] == model]
            if len(model_data) == 0:
                continue
                
            # Get best configuration for this model
            best_config = model_data.loc[model_data['mse'].idxmin()]
            
            # Get the loss curves that were already collected
            tr_losses = best_config['tr_losses']
            val_losses = best_config['val_losses']
            epochs = range(1, len(tr_losses) + 1)
            
            ax = axes[idx]
            
            # Plot training and validation curves
            ax.plot(epochs, tr_losses, label='Training Loss', linewidth=2)
            ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(f'{model.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Check for overfitting (validation loss increases while training loss decreases)
            if len(val_losses) > 10:
                recent_val_trend = val_losses[-10:]
                recent_tr_trend = tr_losses[-10:]
                if (recent_val_trend[-1] > recent_val_trend[0] and 
                    recent_tr_trend[-1] < recent_tr_trend[0]):
                    ax.text(0.5, 0.9, 'OVERFITTING', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(training_graphs_dir / f'training_curves_{ds_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves for {ds_name} in {training_graphs_dir}")

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
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(X, y)

        all_results += run_mlp_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name)
        all_results += run_kan_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name, out_dir=out_dir)
        all_results += run_lstm_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    save_best_results(df, out_dir)
    plot_results(df, out_dir)
    
    # Add training curve analysis using already collected data
    print("\n=== Plotting Training Curves ===")
    plot_training_curves(df, out_dir)
    print("Training curve analysis complete!")


if __name__ == "__main__":
    main()