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
# modulename = "ML_F3W_WXIH0190"
# modulenames = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"]
modulenames = ["ML_F3W_WXIH0190"]
# modelfolder = f"/eos/user/a/areimers/hgcal/dnn_models/{'_'.join(modulenames)}_newtraining"
modelfolder = f"/eos/user/a/areimers/hgcal/dnn_models/{'_'.join(modulenames)}_newtraining_4L_512N"
os.makedirs(name=modelfolder, exist_ok=True)

X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []

for modulename in modulenames:
    inputfolder = f"/eos/user/a/areimers/hgcal/dnn_inputs/{modulename}"

    # --- Load pre-split Data ---
    X_train_thismodule = np.load(f"{inputfolder}/inputs_train.npy")
    X_val_thismodule   = np.load(f"{inputfolder}/inputs_val.npy")
    y_train_thismodule = np.load(f"{inputfolder}/targets_train.npy")
    y_val_thismodule   = np.load(f"{inputfolder}/targets_val.npy")

    X_train_thismodule = torch.tensor(X_train_thismodule, dtype=torch.float32)
    X_val_thismodule   = torch.tensor(X_val_thismodule,   dtype=torch.float32)
    y_train_thismodule = torch.tensor(y_train_thismodule, dtype=torch.float32)
    y_val_thismodule   = torch.tensor(y_val_thismodule,   dtype=torch.float32)

    # Append to lists
    X_train_list.append(X_train_thismodule)
    y_train_list.append(y_train_thismodule)
    X_val_list.append(X_val_thismodule)
    y_val_list.append(y_val_thismodule)

    print(f"Loaded data from module {modulename}")
    print(f"Train: {X_train_thismodule.shape}, val: {X_val_thismodule.shape}")
    print(f"Target mean (train): {y_train_thismodule.mean().item()}, -- std: {y_train_thismodule.std().item()}\n")

# Concatenate all data
X_train = torch.cat(X_train_list, dim=0)
y_train = torch.cat(y_train_list, dim=0)
X_val   = torch.cat(X_val_list,   dim=0)
y_val   = torch.cat(y_val_list,   dim=0)

# --- Verify concatenation ---
print(f"Final shapes: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"              X_val   {X_val.shape},   y_val   {y_val.shape}")

# Check matching lengths
assert X_train.shape[0] == y_train.shape[0], "Mismatch in training data size!"
assert X_val.shape[0]   == y_val.shape[0],   "Mismatch in validation data size!"

# Check combined statistics
print(f"Combined target mean (train): {y_train.mean().item():.4f}, std: {y_train.std().item():.4f}")
print(f"Combined target mean (val):   {y_val.mean().item():.4f}, std: {y_val.std().item():.4f}")

# --- Create DataLoaders ---
batch_size = 4096 * len(modulenames)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size)

# --- Define Model ---
class RegressionDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# --- Define Model ---
class RegressionDNN4L(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# model = RegressionDNN(input_dim=X_train.shape[1]).to(device)
model = RegressionDNN4L(input_dim=X_train.shape[1]).to(device)

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
n_epochs = 1000
train_losses = []
val_losses = []
best_val_loss = float("inf")
patience_counter = 0
early_stop_patience = 15  # stop after 30 epochs without improvement

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
        torch.save(model.state_dict(), f"{modelfolder}/regression_dnn_best.pth")
        print(f"New best model saved with val loss: {val_loss:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{early_stop_patience})")

    if patience_counter >= early_stop_patience:
        print("Early stopping triggered.")
        break

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    torch.save(model.state_dict(), f"{modelfolder}/regression_dnn.pth")
    np.save(f"{modelfolder}/train_losses.npy", np.array(train_losses))
    np.save(f"{modelfolder}/val_losses.npy", np.array(val_losses))

print(f"Last model state saved to {modelfolder}/regression_dnn.pth")
print(f"Best model state saved to: {modelfolder}/regression_dnn_best.pth")