"""
predict.py - Example usage of the trained GCN with the triage policy.

Loads the saved GCN weights, the temperature (calibration) and the confidence
threshold chosen in dilemma.py, then:
    1. reports test-set accuracy,
    2. prints example predictions for test pages with an AUTO / HUMAN REVIEW decision,
    3. optionally looks up specific pages by name (--name),
    4. saves a figure of class probabilities for the examples.

Usage:
    python predict.py                      # 8 random test pages
    python predict.py --name "Senate"      # pages whose name contains "Senate"
    python predict.py --n 12 --seed 7

Output:
    figures/predict_examples.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # allowed in predict.py (visualisation only)
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, get_data
from evaluate import load_model


def main():
    parser = argparse.ArgumentParser(description="Run the trained GCN with the triage policy")
    parser.add_argument("--data", default="data/facebook.npz")
    parser.add_argument("--split_seed", type=int, default=42, help="must match training")
    parser.add_argument("--seed", type=int, default=0, help="which random test pages to show")
    parser.add_argument("--n", type=int, default=8, help="number of example pages")
    parser.add_argument("--name", default=None, help="show pages whose name contains this text")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, names = get_data(args.data, seed=args.split_seed)
    data = data.to(device)

    # calibration + operating threshold from dilemma.py (both chosen on validation)
    with open("results/dilemma.json") as f:
        cfg = json.load(f)["gcn"]
    T, threshold = cfg["temperature"], cfg["threshold_policy"]["threshold"]

    model = load_model("gcn", data, device)
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index) / T, dim=1).cpu()
    conf, pred = probs.max(dim=1)
    y = data.y.cpu()
    test_idx = torch.where(data.test_mask.cpu())[0]

    test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()
    automated = conf[test_idx] >= threshold
    print(f"\nGCN test accuracy: {test_acc:.3f}  ({len(test_idx)} test pages)")
    print(f"Triage policy: temperature T={T:.2f}, auto-handle if confidence >= {threshold:.3f}")
    print(f"  -> {100 * automated.float().mean():.1f}% automated, "
          f"{100 * (~automated).float().mean():.1f}% sent to human review")

    # choose example pages
    if args.name:
        chosen = [i for i in test_idx.tolist() if args.name.lower() in names[i].lower()]
        if not chosen:
            print(f"\nNo test pages match '{args.name}'.")
            return
        chosen = chosen[:args.n]
    else:
        g = torch.Generator().manual_seed(args.seed)
        chosen = test_idx[torch.randperm(len(test_idx), generator=g)[:args.n]].tolist()

    print(f"\n{'page name':<40}{'true':<11}{'predicted':<11}{'conf':>6}  decision")
    print("-" * 82)
    for i in chosen:
        decision = "AUTO" if conf[i] >= threshold else "HUMAN REVIEW"
        mark = "" if pred[i] == y[i] else "  <- wrong"
        print(f"{names[i][:39]:<40}{CLASS_NAMES[y[i]]:<11}{CLASS_NAMES[pred[i]]:<11}"
              f"{conf[i].item():>6.2f}  {decision}{mark}")

    # visualisation: per-page class probabilities with the threshold marked
    rows = len(chosen)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 1.1 * rows + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, chosen):
        colours = ["tab:green" if k == y[i] else "tab:grey" for k in range(len(CLASS_NAMES))]
        ax.barh(CLASS_NAMES, probs[i].numpy(), color=colours)
        ax.axvline(threshold, color="tab:red", linestyle="--", lw=1)
        decision = "AUTO" if conf[i] >= threshold else "REVIEW"
        ax.set_title(f"{names[i][:45]}  ->  {CLASS_NAMES[pred[i]]} ({decision})", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
    axes[-1].set_xlabel("Calibrated probability (green = true class, red line = threshold)")
    fig.tight_layout()
    fig.savefig("figures/predict_examples.png", dpi=150)
    plt.close(fig)
    print("\nSaved figures/predict_examples.png")


if __name__ == "__main__":
    main()