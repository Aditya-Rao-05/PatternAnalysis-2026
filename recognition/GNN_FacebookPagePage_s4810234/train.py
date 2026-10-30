"""
train.py - Train, validate, test and save the MLP baseline and the GCN.

Both models are trained on the IDENTICAL stratified split (same seed) so the
comparison is fair. Training is transductive: the full graph is passed through
the model every step, but the loss only uses training-node labels.

Usage:
    python train.py                # trains both models
    python train.py --model gcn    # trains only the GCN

Outputs:
    checkpoints/<model>_best.pt    best weights by validation loss (gitignored)
    figures/<model>_curves.png     loss + accuracy curves for the README
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import get_data
from modules import GCN, MLPBaseline, count_parameters

MODELS = {"mlp": MLPBaseline, "gcn": GCN}


def set_seed(seed):
    """Fix random seeds so weight initialisation and dropout are reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data, mask):
    """Return (loss, accuracy) on the nodes selected by `mask`.

    no_grad + model.eval(): no gradients are tracked and dropout is turned off,
    so this is a clean measurement that cannot change the weights.
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask]).item()
    acc = (logits[mask].argmax(dim=1) == data.y[mask]).float().mean().item()
    return loss, acc


def train_model(name, data, args, device):
    """Train one model with early stopping on validation loss."""
    set_seed(args.seed)
    model = MODELS[name](in_dim=data.num_node_features, hidden_dim=args.hidden,
                         num_classes=int(data.y.max()) + 1,
                         dropout=args.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_epoch, bad_epochs = float("inf"), 0, 0
    ckpt_path = os.path.join("checkpoints", f"{name}_best.pt")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- one training step ----
        model.train()                                   # dropout ON
        optimiser.zero_grad()                           # clear old gradients
        logits = model(data.x, data.edge_index)         # forward pass on the WHOLE graph
        loss = F.cross_entropy(logits[data.train_mask],  # loss on TRAIN labels only
                               data.y[data.train_mask])
        loss.backward()                                 # compute gradients
        optimiser.step()                                # update weights

        # ---- measure progress ----
        tr_loss, tr_acc = evaluate(model, data, data.train_mask)
        va_loss, va_acc = evaluate(model, data, data.val_mask)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)

        # ---- early stopping: keep the weights with the lowest val loss ----
        if va_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = va_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

        if epoch % 25 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f}"
                  f" | val loss {va_loss:.4f} acc {va_acc:.3f}")

    train_time = time.time() - start

    # reload the best weights and report held-out test accuracy
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc = evaluate(model, data, data.test_mask)
    _, val_acc = evaluate(model, data, data.val_mask)
    print(f"[{name}] best epoch {best_epoch} | val acc {val_acc:.3f} | test acc {test_acc:.3f}"
          f" | params {count_parameters(model)} | train time {train_time:.1f}s\n")

    plot_curves(name, history, best_epoch)
    return model


def plot_curves(name, history, best_epoch):
    """Save train/val loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((ax1, "loss", "Cross-entropy loss"), (ax2, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(best_epoch, color="grey", linestyle="--", label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{name.upper()} - {title}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("figures", f"{name}_curves.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MLP baseline and GCN on Facebook Page-Page")
    parser.add_argument("--model", choices=["mlp", "gcn", "both"], default="both")
    parser.add_argument("--data", default="data/facebook.npz")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50, help="early-stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # CPU is plenty for this graph (~22k nodes); CUDA is used automatically if present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = get_data(args.data, seed=args.seed)
    data = data.to(device)

    names = ["mlp", "gcn"] if args.model == "both" else [args.model]
    for name in names:
        train_model(name, data, args, device)


if __name__ == "__main__":
    main()