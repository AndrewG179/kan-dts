from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config.hyperparameters import INPUT_LEN, PRED_LEN, MLP_SEARCH, KAN_SEARCH, LSTM_SEARCH
from config.datasets import NOISE_ANALYSIS_SPECS
from data_pipeline.data_creation import load_synthetic
from data_pipeline.preprocessing import create_windows
from data_pipeline.splits import chronological_split
from models.mlp_model import MLP
from kan import KAN
from models.train_test import train_mlp, train_kan, train_lstm
from models.lstm_model import LSTMModel

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
        mlp = MLP([INPUT_LEN, *hp["hidden"], PRED_LEN])
        print(f"Running {ds_name} | mlp | {hp}")
        _, _, trained = train_mlp(mlp, X_tr, y_tr, X_val, y_val, epochs=hp["epochs"], lr=hp["lr"])
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({"dataset": ds_name, "model": "mlp", "mse": mse, "params": hp})
    return results

def run_kan_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name):
    results = []
    for hp in KAN_SEARCH:
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
        mse = _evaluate(trained, X_te, y_te, y_mean, y_std)
        results.append({"dataset": ds_name, "model": "kan", "mse": mse, "params": hp})
    return results

def run_lstm_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name):
    results = []
    X_tr_lstm = X_tr[..., None]
    X_val_lstm = X_val[..., None]
    X_te_lstm = X_te[..., None]
    for hp in LSTM_SEARCH:
        lstm = LSTMModel(
            input_size=1,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            output_size=PRED_LEN,
            dropout=hp["dropout"]
        )
        print(f"Running {ds_name} | lstm | {hp}")
        _, _, trained = train_lstm(
            lstm, X_tr_lstm, y_tr, X_val_lstm, y_val,
            epochs=hp["epochs"], lr=hp["lr"]
        )
        mse = _evaluate(trained, X_te_lstm, y_te, y_mean, y_std)
        results.append({"dataset": ds_name, "model": "lstm", "mse": mse, "params": hp})
    return results

def create_noise_comparison_plots(df, out_dir):
    """Create comprehensive noise comparison plots."""
    
    # Extract noise level and dataset type
    df[['type', 'noise']] = df['dataset'].str.replace('syn-', '').str.split('-', expand=True)
    df['noise'] = df['noise'].astype(float)
    
    # Create separate plots for each dataset type
    for dataset_type in ['trend_only', 'season_only', 'trend_season']:
        type_data = df[df['type'] == dataset_type]
        
        plt.figure(figsize=(10, 6))
        
        for model in type_data['model'].unique():
            model_data = type_data[type_data['model'] == model]
            best_by_noise = model_data.groupby('noise')['mse'].min().sort_index()
            
            plt.plot(best_by_noise.index, best_by_noise.values, 
                   marker='o', linewidth=2, markersize=8, label=model.upper())
        
        plt.xlabel('Noise Level (σ)')
        plt.ylabel('Best MSE (lower is better)')
        plt.title(f'Model Performance vs Noise Level - {dataset_type.replace("_", " ").title()}')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig(out_dir / f'noise_analysis_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual noise analysis plot for {dataset_type}")
    
    # Create combined plot with all dataset types
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, dataset_type in enumerate(['trend_only', 'season_only', 'trend_season']):
        type_data = df[df['type'] == dataset_type]
        ax = axes[idx]
        
        for model in type_data['model'].unique():
            model_data = type_data[type_data['model'] == model]
            best_by_noise = model_data.groupby('noise')['mse'].min().sort_index()
            
            ax.plot(best_by_noise.index, best_by_noise.values, 
                   marker='o', linewidth=2, markersize=8, label=model.upper())
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Best MSE (lower is better)')
        ax.set_title(f'{dataset_type.replace("_", " ").title()}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comprehensive_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined noise analysis plot")
    
    # Create summary table
    summary_data = []
    for dataset_type in df['type'].unique():
        for model in df['model'].unique():
            type_model_data = df[(df['type'] == dataset_type) & (df['model'] == model)]
            if len(type_model_data) > 0:
                no_noise_mse = type_model_data[type_model_data['noise'] == 0.0]['mse'].min()
                high_noise_mse = type_model_data[type_model_data['noise'] == 1.0]['mse'].min()
                degradation_factor = high_noise_mse / no_noise_mse if no_noise_mse > 0 else float('inf')
                
                summary_data.append({
                    'dataset_type': dataset_type,
                    'model': model,
                    'no_noise_mse': no_noise_mse,
                    'high_noise_mse': high_noise_mse,
                    'degradation_factor': degradation_factor
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir / 'noise_robustness_summary.csv', index=False)
    
    print(f"Saved comprehensive noise analysis to {out_dir}")
    return summary_df

def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "noise_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Running Comprehensive Noise Analysis ===")
    
    datasets = [f"syn-{kind}-{noise}" for (kind, noise) in NOISE_ANALYSIS_SPECS]
    all_results = []
    
    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")
        df = load_synthetic(kind=ds_name.split('-')[1], noise_std=float(ds_name.split('-')[2]))
        X, y = create_windows(df, input_len=INPUT_LEN, pred_len=PRED_LEN)
        X, y, X_mean, X_std, y_mean, y_std = _normalize(X, y)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(X, y)
        
        all_results += run_mlp_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name)
        all_results += run_kan_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name)
        all_results += run_lstm_search(X_tr, y_tr, X_val, y_val, X_te, y_te, y_mean, y_std, ds_name)
    
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "noise_analysis_results.csv", index=False)
    
    # Create comprehensive noise comparison plots
    summary = create_noise_comparison_plots(df, out_dir)
    print("\n=== Noise Analysis Complete ===")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main() 