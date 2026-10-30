"""
evaluate.py - Quantitative benchmarking of the trained MLP baseline vs GCN
on the held-out TEST split (identical for both models).

Reports (spec section 2.1, guideline 3):
    - per-class precision, recall, F1; macro-F1; accuracy
    - one-vs-rest AUROC (per class and macro)
    - confusion matrices
    - 2D t-SNE of the hidden embeddings, coloured by ground-truth class
    - resource profile: parameter count, inference latency, peak memory

Run after train.py:
    python evaluate.py

Outputs:
    figures/confusion_matrices.png
    figures/tsne_embeddings.png
    results/metrics.json
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from dataset import CLASS_NAMES, get_data
from modules import GCN, MLPBaseline, count_parameters

MODELS = {"mlp": MLPBaseline, "gcn": GCN}


def load_model(name, data, device, hidden=64, dropout=0.5):
    """Rebuild a model with the training hyperparameters and load its best checkpoint."""
    model = MODELS[name](in_dim=data.num_node_features, hidden_dim=hidden,
                         num_classes=len(CLASS_NAMES), dropout=dropout).to(device)
    path = os.path.join("checkpoints", f"{name}_best.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, data):
    """Full-graph forward pass. Returns softmax probabilities and hidden embeddings."""
    logits, emb = model(data.x, data.edge_index, return_embedding=True)
    return F.softmax(logits, dim=1), emb


@torch.no_grad()
def profile(model, data, device, repeats=20):
    """Average full-graph inference latency (ms) and peak memory."""
    model(data.x, data.edge_index)  # warm-up run (first call is slower)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        model(data.x, data.edge_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = 1000 * (time.perf_counter() - start) / repeats
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None
    return latency_ms, peak_mb


def compute_metrics(probs, y_true):
    """All classification metrics on one set of predictions (CPU tensors)."""
    y_pred = probs.argmax(dim=1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_auroc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "per_class": {
            c: {"precision": report[c]["precision"], "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
                "auroc": roc_auc_score((y_true == i).int(), probs[:, i])}
            for i, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion(results):
    """Side-by-side row-normalised confusion matrices (rows = true class)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, results):
        cm = torch.tensor(results[name]["confusion_matrix"], dtype=torch.float)
        cm_norm = cm / cm.sum(dim=1, keepdim=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center",
                        va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASS_NAMES, rotation=30)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name.upper()} (test acc {results[name]['accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_tsne(embeddings, y_true, seed=42):
    """t-SNE of the 64-dim hidden layer for test nodes, coloured by true class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (name, emb) in zip(axes, embeddings.items()):
        xy = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=seed).fit_transform(emb.numpy())
        for i, c in enumerate(CLASS_NAMES):
            pts = xy[(y_true == i).numpy()]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, label=c)
        ax.set_title(f"t-SNE of {name.upper()} hidden embeddings (test nodes)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig("figures/tsne_embeddings.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLP vs GCN on the test split")
    parser.add_argument("--data", default="data/facebook.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = get_data(args.data, seed=args.seed)
    data = data.to(device)
    test = data.test_mask
    y_test = data.y[test].cpu()

    results, embeddings = {}, {}
    for name in ("mlp", "gcn"):
        model = load_model(name, data, device)
        probs, emb = predict_all(model, data)
        results[name] = compute_metrics(probs[test].cpu(), y_test)
        latency, peak_mb = profile(model, data, device)
        results[name]["params"] = count_parameters(model)
        results[name]["inference_ms"] = latency
        results[name]["peak_gpu_mb"] = peak_mb
        embeddings[name] = emb[test].cpu()

    # ---- print summary tables ----
    print(f"\n{'':<12}{'MLP':>10}{'GCN':>10}")
    for key in ("accuracy", "macro_f1", "macro_auroc"):
        print(f"{key:<12}{results['mlp'][key]:>10.3f}{results['gcn'][key]:>10.3f}")
    print(f"{'params':<12}{results['mlp']['params']:>10}{results['gcn']['params']:>10}")
    print(f"{'infer (ms)':<12}{results['mlp']['inference_ms']:>10.2f}"
          f"{results['gcn']['inference_ms']:>10.2f}")
    print(f"device: {device.type}"
          + ("" if device.type == "cuda" else " (peak GPU memory not applicable on CPU)"))

    print("\nPer-class test metrics (precision / recall / F1 / AUROC):")
    for name in ("mlp", "gcn"):
        print(f"  {name.upper()}")
        for c, m in results[name]["per_class"].items():
            print(f"    {c:<11} P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  AUROC {m['auroc']:.3f}")

    plot_confusion(results)
    print("\nComputing t-SNE (takes ~30-60 s)...")
    plot_tsne(embeddings, y_test, seed=args.seed)

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved figures/confusion_matrices.png, figures/tsne_embeddings.png, results/metrics.json")


if __name__ == "__main__":
    main()