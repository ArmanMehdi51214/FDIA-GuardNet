from __future__ import annotations

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.hybrid_cnn_transformer_bilstm import HybridCNNTransformerBiLSTM
from src.preprocessing.dataloader import create_dataloader


# -------------------------------
# Config
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SEQ_DIR = "data/processed/sequences/train"
VAL_SEQ_DIR   = "data/processed/sequences/val"
TEST_SEQ_DIR  = "data/processed/sequences/test"

BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10

LOG_DIR = "results/logs"
CKPT_DIR = "results/checkpoints"
FIG_DIR = "results/figures"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# -------------------------------
# Utilities
# -------------------------------

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, p, r, f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        y_true.extend(y.cpu().numpy())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

    acc, p, r, f1 = compute_metrics(y_true, y_pred)
    return np.mean(losses), acc, p, r, f1


# -------------------------------
# Main Training Loop
# -------------------------------

def main():
    writer = SummaryWriter(LOG_DIR)

    train_loader = create_dataloader(TRAIN_SEQ_DIR, BATCH_SIZE, shuffle=True)
    val_loader   = create_dataloader(VAL_SEQ_DIR,   BATCH_SIZE, shuffle=False)
    test_loader  = create_dataloader(TEST_SEQ_DIR,  BATCH_SIZE, shuffle=False)

    model = HybridCNNTransformerBiLSTM().to(DEVICE)

    class_weights = torch.tensor([1.0, 1.85], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    writer.close()

    # -------------------------------
    # Final Evaluation on TEST
    # -------------------------------
    model.load_state_dict(torch.load(f"{CKPT_DIR}/best_model.pt"))
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(
        model, test_loader, criterion
    )

    print("\n=== TEST RESULTS ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_p:.4f}")
    print(f"Recall: {test_r:.4f}")
    print(f"F1-score: {test_f1:.4f}")

    # -------------------------------
    # Plot Learning Curves
    # -------------------------------
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Learning Curve")
    plt.savefig(f"{FIG_DIR}/loss_curve.png")

    plt.figure()
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.title("Validation F1 Curve")
    plt.savefig(f"{FIG_DIR}/f1_curve.png")


if __name__ == "__main__":
    main()
