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

def train_kan(model, Xtr, ytr, Xval, yval, *, steps, lr, lamb_l1, lamb_entropy, patience=10, eval_every=10):
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
    total_steps = 0
    for step in range(0, steps, eval_every):
        # Fit for eval_every steps
        model.fit(
            dataset=dataset,
            opt=KAN_OPT,
            steps=eval_every,
            lr=lr,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy
        )
        total_steps += eval_every
        # Evaluate
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            steps_no_improve = 0
        else:
            steps_no_improve += 1
        if steps_no_improve >= patience:
            print(f"Early stopping at step {total_steps}")
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