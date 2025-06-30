import torch
import torch.nn as nn
import torch.optim as optim
from config.hyperparameters import KAN_OPT
import copy

def train_mlp(model, Xtr, ytr, Xval, yval, *, epochs, lr, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device).squeeze()
    Xval= torch.tensor(Xval,dtype=torch.float32, device=device)
    yval= torch.tensor(yval,dtype=torch.float32, device=device).squeeze()

    tr_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = copy.deepcopy(model.cpu())
    model.to(device)
    epochs_no_improve = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr).squeeze(), ytr)
        loss.backward(); opt.step()
        tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            v = crit(model(Xval).squeeze(), yval).item()
        val_losses.append(v)

        if v < best_val_loss:
            best_val_loss = v
            best_model = copy.deepcopy(model.cpu())
            model.to(device)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    return tr_losses, val_losses, best_model

def train_kan(model, Xtr, ytr, Xval, yval, *, steps, lr, lamb_l1, lamb_entropy, patience=10):
    import numpy as np
    dataset = {
        "train_input": torch.tensor(Xtr, dtype=torch.float32),
        "train_label": torch.tensor(ytr, dtype=torch.float32),
        "test_input":  torch.tensor(Xval, dtype=torch.float32),
        "test_label":  torch.tensor(yval, dtype=torch.float32)
    }
    best_val_loss = float('inf')
    best_model = None
    val_losses = []
    tr_losses = []
    steps_no_improve = 0
    
    # Enhanced early stopping parameters
    min_delta = 1e-6  # Minimum improvement threshold
    
    for step in range(steps):
        # Fit for 1 step at a time
        model.fit(
            dataset=dataset,
            opt=KAN_OPT,
            steps=1,
            lr=lr,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy
        )
        
        # Evaluate after each step
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(Xval, dtype=torch.float32)).cpu().numpy().squeeze()
            val_true = np.array(yval).squeeze()
            val_loss = np.mean((val_pred - val_true) ** 2)
            val_losses.append(val_loss)
            tr_pred = model(torch.tensor(Xtr, dtype=torch.float32)).cpu().numpy().squeeze()
            tr_true = np.array(ytr).squeeze()
            tr_loss = np.mean((tr_pred - tr_true) ** 2)
            tr_losses.append(tr_loss)
            
        # Enhanced early stopping logic with minimum improvement threshold
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            steps_no_improve = 0
        else:
            steps_no_improve += 1
            
        # Additional overfitting detection: if validation loss increases while training loss decreases
        if len(val_losses) >= 5:
            recent_val_trend = np.mean(val_losses[-3:]) - np.mean(val_losses[-5:-2])
            recent_tr_trend = np.mean(tr_losses[-3:]) - np.mean(tr_losses[-5:-2])
            
            # Strong overfitting signal: val loss increasing, train loss decreasing
            if recent_val_trend > 0 and recent_tr_trend < -min_delta:
                print(f"Overfitting detected at step {step+1} - stopping early")
                break
            
        if steps_no_improve >= patience:
            print(f"Early stopping at step {step+1}")
            break
            
    if best_model is None:
        best_model = model
    return tr_losses, val_losses, best_model

def train_lstm(model, Xtr, ytr, Xval, yval, *, epochs, lr, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=lr)

    # LSTM expects input shape: (batch, seq_len, input_size)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device).squeeze()
    Xval= torch.tensor(Xval,dtype=torch.float32, device=device)
    yval= torch.tensor(yval,dtype=torch.float32, device=device).squeeze()

    tr_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = copy.deepcopy(model.cpu())
    model.to(device)
    epochs_no_improve = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = crit(out.squeeze(), ytr)
        loss.backward(); opt.step()
        tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out = model(Xval)
            v = crit(val_out.squeeze(), yval).item()
        val_losses.append(v)

        if v < best_val_loss:
            best_val_loss = v
            best_model = copy.deepcopy(model.cpu())
            model.to(device)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    return tr_losses, val_losses, best_model