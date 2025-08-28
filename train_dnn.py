#! /eos/user/a/areimers/torch-env/bin/python

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import argparse

import models




defaults = {
    # "modules_for_training": ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"],
    "modules_for_training": ["ML_F3W_WXIH0190"],

    # "nodes_per_layer": [128, 128, 64],
    "nodes_per_layer": [512, 512, 512, 512, 64],
    # "nodes_per_layer": [16, 16, 16],

    "dropout_rate": 0.0,
    # "dropout_rate": 0.05,
    # "dropout_rate": 0.1,
    # "dropout_rate": 0.2,
    # "dropout_rate": 0.3,

    "max_epochs": 1000,
    # "max_epochs": 30,

    "modeltag": "",
    # "modeltag": "test",
    # "modeltag": "submittest",

    # "inputfoldertag": "",
    # "inputfoldertag": "_nochadc",
    "inputfoldertag": "_1cm2ch",

    "override_full_model_name": False,
    "new_model_name": "TESTTEST",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a flexible DNN model on HGCal inputs"
    )
    # Data modules
    parser.add_argument(
        '-m', '--modules',
        nargs='+',
        default=defaults["modules_for_training"],
        help="List of module names to train on"
    )
    # Architecture
    parser.add_argument(
        '-n', '--nodes',
        nargs='+',
        type=int,
        default=defaults["nodes_per_layer"],
        help="Nodes per hidden layer"
    )
    parser.add_argument(
        '-d', '--dropout',
        type=float,
        default=defaults["dropout_rate"],
        help="Dropout rate between layers"
    )
    # Training settings
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=defaults["max_epochs"],
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        '-t', '--tag',
        type=str,
        default=defaults["modeltag"],
        help="Custom tag for model naming"
    )
    parser.add_argument(
        '--inputfoldertag',
        type=str,
        default=defaults["inputfoldertag"],
        help="Tag for folder to take inputs from, different flavors of inputs exist"
    )
    parser.add_argument(
        '--override-name',
        type=bool,
        default=defaults["override_full_model_name"],
        help="Override full model name with provided --new-name"
    )
    parser.add_argument(
        '--new-name',
        type=str,
        default=defaults["new_model_name"],
        help="New model name if overriding"
    )
    return parser.parse_args()










def main():
    args = parse_args()

    modulenames_for_training = args.modules
    nodes_per_layer = args.nodes
    dropout_rate = args.dropout
    max_epochs = args.epochs
    modeltag = args.tag
    override_full_model_name = args.override_name
    new_model_name = args.new_name

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    for modulename in modulenames_for_training:
        inputfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs{args.inputfoldertag}/{modulename}"

        X_train_list.append(torch.tensor(np.load(f"{inputfolder}/inputs_train.npy"), dtype=torch.float32))
        X_val_list.append(torch.tensor(np.load(f"{inputfolder}/inputs_val.npy"), dtype=torch.float32))
        y_train_list.append(torch.tensor(np.load(f"{inputfolder}/targets_train.npy"), dtype=torch.float32))
        y_val_list.append(torch.tensor(np.load(f"{inputfolder}/targets_val.npy"), dtype=torch.float32))

        # Print module-wise loading info
        n_train = X_train_list[-1].size(0)
        n_val = X_val_list[-1].size(0)
        print(f"Loaded module {modulename}: train events = {n_train}, val events = {n_val}")

    # Concatenate all modules
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_val   = torch.cat(X_val_list,   dim=0)
    y_val   = torch.cat(y_val_list,   dim=0)

    # Verify concatenation
    print(f"Final shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"              X_val   {X_val.shape},   y_val   {y_val.shape}")
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in training data size!"
    assert X_val.shape[0]   == y_val.shape[0],   "Mismatch in validation data size!"

    # Combined statistics
    print(f"Combined target mean (train): {y_train.mean().item():.4f}, std: {y_train.std().item():.4f}")
    print(f"Combined target mean (val):   {y_val.mean().item():.4f}, std: {y_val.std().item():.4f}")

    # DataLoaders
    n_modules = len(modulenames_for_training)
    base_bs = 4096
    batch_size = base_bs * n_modules
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size)

    # Model
    model = models.DNNFlex(
        input_dim=X_train.shape[1],
        nodes_per_layer=nodes_per_layer,
        dropout_rate=dropout_rate,
        tag=modeltag
    ).to(device)
    if override_full_model_name:
        model.override_model_string(new_model_name)

    modelfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_models{args.inputfoldertag}/{'_'.join(modulenames_for_training)}/{model.get_model_string()}"
    os.makedirs(modelfolder, exist_ok=False)

    # Training setup
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 * n_modules)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6 * n_modules)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    window = 5
    avg_val_history = []

    for epoch in range(max_epochs):
        print(f"\n--- Starting epoch {epoch+1}/{max_epochs} ---")
        start = time.time()
        model.train()
        total_train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward(); optimizer.step()
            total_train_loss += loss.item() * Xb.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                total_val_loss += loss_fn(model(Xb), yb).item() * Xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Scheduler step with rolling average
        if len(val_losses) >= window:
            avg_recent = float(np.mean(val_losses[-window:]))
            avg_val_history.append(avg_recent)
            scheduler.step(avg_recent)
            print(f"[Epoch {epoch+1}] 5-epoch avg val loss: {avg_recent:.4f}")
        else:
            print(f"[Epoch {epoch+1}] warming up scheduler")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{modelfolder}/regression_dnn_best.pth")
            print(f"New best model (loss: {val_loss:.6f}) saved.")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print("Early stopping.")
                break

        # Save latest and histories
        torch.save(model.state_dict(), f"{modelfolder}/regression_dnn.pth")
        np.save(f"{modelfolder}/train_losses.npy", np.array(train_losses))
        np.save(f"{modelfolder}/val_losses.npy",   np.array(val_losses))

        print(f"Epoch {epoch+1}/{max_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Time {time.time()-start:.1f}s")

    print(f"Training completed. Models and logs in {modelfolder}")


if __name__ == '__main__':
    main()
