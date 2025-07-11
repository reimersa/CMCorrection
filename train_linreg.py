import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.utils.class_weight import compute_sample_weight
import time
import os
from tqdm import tqdm

# --- Select device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define paths ---
modulename = "ML_F3W_WXIH0190"
inputfolder = f"/eos/user/a/areimers/hgcal/dnn_inputs/{modulename}"
modelfolder = f"/eos/user/a/areimers/hgcal/dnn_models/{modulename}"
os.makedirs(name=modelfolder, exist_ok=True)

# --- Load pre-split Data ---
X_train = np.load(f"{inputfolder}/inputs_train.npy")
X_val   = np.load(f"{inputfolder}/inputs_val.npy")
y_train = np.load(f"{inputfolder}/targets_train.npy")
y_val   = np.load(f"{inputfolder}/targets_val.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32)

print(f"Loaded train: {X_train.shape}, val: {X_val.shape}")
print("Target mean (train):", y_train.mean().item(), " -- std:", y_train.std().item())

# --- Create DataLoaders ---
batch_size = 4096
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size)

# --- Define Model ---
class RegressionLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

model = RegressionLinear(input_dim=X_train.shape[1]).to(device)

# --- Training Setup ---
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
)

# --- Training Loop ---
# n_epochs = 1
n_epochs = 3
train_losses = []
val_losses = []
best_val_loss = float("inf")
patience_counter = 0
early_stop_patience = 8

for epoch in range(n_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    nevts_processed_train = 0
    print(f"\nEpoch {epoch+1}/{n_epochs}")

    pbar_train = tqdm(train_loader, desc="Training", leave=False)
    # for batch_X, batch_y, batch_w in pbar_train:
    for batch_X, batch_y in pbar_train:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
        nevts_processed_train += batch_X.size(0)

        current_lr = optimizer.param_groups[0]['lr']
        pbar_train.set_postfix(
            current_loss=train_loss / nevts_processed_train,
            lr=f"{current_lr:.2e}"
        )

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    nevts_processed_val = 0
    pbar_val = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch_X, batch_y in pbar_val:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            nevts_processed_val += batch_X.size(0)
            pbar_val.set_postfix(current_loss=val_loss / nevts_processed_val)

    val_loss /= len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{modelfolder}/linreg_best.pth")
        print(f"New best model saved with val loss: {val_loss:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{early_stop_patience})")

    if patience_counter >= early_stop_patience:
        print("Early stopping triggered.")
        break

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    torch.save(model.state_dict(), f"{modelfolder}/linreg.pth")
    np.save(f"{modelfolder}/linred_train_losses.npy", np.array(train_losses))
    np.save(f"{modelfolder}/linred_val_losses.npy", np.array(val_losses))

print(f"Last model state saved to {modelfolder}/linreg.pth")
print(f"Best model state saved to: {modelfolder}/linreg_best.pth")