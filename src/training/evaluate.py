import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.models.hybrid_cnn_transformer_bilstm import HybridCNNTransformerBiLSTM
from src.preprocessing.dataloader import create_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_SEQ_DIR = "data/processed/sequences/test"
CKPT_PATH = "results/checkpoints/best_model.pt"
FIG_DIR = "results/figures"

def main():
    model = HybridCNNTransformerBiLSTM().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    test_loader = create_dataloader(TEST_SEQ_DIR, batch_size=64, shuffle=False)

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(y.numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png")
    plt.close()

    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curve.png")
    plt.close()

    print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
