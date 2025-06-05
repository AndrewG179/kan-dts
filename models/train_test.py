import torch
import torch.nn as nn
import torch.optim as optim
from config.hyperparameters import KAN_OPT

def train_mlp(model, Xtr, ytr, Xval, yval, *, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device).squeeze()
    Xval= torch.tensor(Xval,dtype=torch.float32, device=device)
    yval= torch.tensor(yval,dtype=torch.float32, device=device).squeeze()

    tr_losses, val_losses = [], []
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
    return tr_losses, val_losses, model.cpu()

def train_kan(model, Xtr, ytr, Xval, yval, *, steps, lr, lamb_l1, lamb_entropy):
    dataset = {
        "train_input": torch.tensor(Xtr, dtype=torch.float32),
        "train_label": torch.tensor(ytr, dtype=torch.float32),
        "test_input":  torch.tensor(Xval, dtype=torch.float32),
        "test_label":  torch.tensor(yval, dtype=torch.float32)
    }
    model.fit(
        dataset=dataset,
        opt=KAN_OPT,
        steps=steps,
        lr=lr,
        lamb_l1=lamb_l1,
        lamb_entropy=lamb_entropy
    )
    return [], [], model