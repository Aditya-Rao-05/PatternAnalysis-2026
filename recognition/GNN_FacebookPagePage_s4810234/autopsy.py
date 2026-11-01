"""
autopsy.py - Failure case autopsy for the GCN (spec section 2.1, guideline 4).

For misclassified TEST pages, this script reports:
    - page name, true vs predicted class, calibrated confidence
    - degree (number of neighbours)
    - neighbour label mix and node homophily (fraction of neighbours that share
      the page's TRUE class)
    - whether the features-only MLP got the same page right

It then checks whether failures are SYSTEMATIC by comparing GCN and MLP error
rates across node-homophily and degree bins.

Note: neighbour labels are read from the ground truth purely for this post-hoc
diagnosis; they are never used for training or prediction.

Run after train.py and dilemma.py:
    python autopsy.py

Outputs:
    figures/error_by_homophily_degree.png
    results/gcn_errors.csv        (all misclassified test pages, most confident first)
    results/autopsy.json
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from dataset import CLASS_NAMES, get_data
from evaluate import load_model


def neighbour_stats(edge_index, y, num_nodes):
    """Per node: degree, count of neighbours in each class, and node homophily."""
    src, dst = edge_index
    deg = degree(src, num_nodes=num_nodes)
    counts = torch.zeros(num_nodes, len(CLASS_NAMES))
    # for every edge src->dst, add one to src's count for dst's class
    counts.index_put_((src, y[dst]), torch.ones(src.numel()), accumulate=True)
    homophily = counts[torch.arange(num_nodes), y] / deg.clamp(min=1)
    return deg, counts, homophily


def error_rate_by_bins(values, wrong, edges):
    """Error rate of the nodes whose value falls in each [lo, hi) bin."""
    rates, sizes = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (values >= lo) & (values < hi)
        sizes.append(int(in_bin.sum()))
        rates.append(wrong[in_bin].float().mean().item() if in_bin.any() else float("nan"))
    return rates, sizes


def main():
    parser = argparse.ArgumentParser(description="Failure case autopsy for the GCN")
    parser.add_argument("--data", default="data/facebook.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=8, help="how many confident errors to print")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, names = get_data(args.data, seed=args.seed)
    data = data.to(device)

    # calibrated GCN probabilities (temperature from dilemma.py), raw MLP predictions
    with open("results/dilemma.json") as f:
        T = json.load(f)["gcn"]["temperature"]
    with torch.no_grad():
        gcn_probs = F.softmax(load_model("gcn", data, device)(data.x, data.edge_index) / T, 1).cpu()
        mlp_pred = load_model("mlp", data, device)(data.x, data.edge_index).argmax(1).cpu()

    y = data.y.cpu()
    test = data.test_mask.cpu()
    deg, counts, homophily = neighbour_stats(data.edge_index.cpu(), y, data.num_nodes)
    conf, gcn_pred = gcn_probs.max(1)
    gcn_wrong = gcn_pred != y
    mlp_wrong = mlp_pred != y

    # ---- individual failure cases: misclassified test pages, most confident first ----
    err_idx = torch.where(test & gcn_wrong)[0]
    err_idx = err_idx[conf[err_idx].argsort(descending=True)]

    rows = []
    for i in err_idx.tolist():
        mix = {c: int(counts[i, k]) for k, c in enumerate(CLASS_NAMES)}
        rows.append({
            "node": i, "page_name": names[i],
            "true": CLASS_NAMES[y[i]], "gcn_pred": CLASS_NAMES[gcn_pred[i]],
            "gcn_conf": round(conf[i].item(), 3),
            "mlp_pred": CLASS_NAMES[mlp_pred[i]],
            "degree": int(deg[i]), "homophily": round(homophily[i].item(), 3),
            "neighbours_in_pred_class": round(counts[i, gcn_pred[i]].item() / max(deg[i].item(), 1), 3),
            **{f"n_{c}": v for c, v in mix.items()},
        })

    os.makedirs("results", exist_ok=True)
    with open("results/gcn_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGCN test errors: {len(rows)} (full list in results/gcn_errors.csv)")
    print(f"\nTop {args.top} most confident GCN errors:")
    header = f"{'page name':<38}{'true':<11}{'pred':<11}{'conf':>5}{'deg':>5}{'homo':>6}{'%pred':>6}  MLP"
    print(header)
    print("-" * len(header))
    for r in rows[:args.top]:
        print(f"{r['page_name'][:37]:<38}{r['true']:<11}{r['gcn_pred']:<11}{r['gcn_conf']:>5.2f}"
              f"{r['degree']:>5}{r['homophily']:>6.2f}{r['neighbours_in_pred_class']:>6.2f}  {r['mlp_pred']}")
    print("  homo  = fraction of neighbours sharing the page's TRUE class")
    print("  %pred = fraction of neighbours in the class the GCN PREDICTED")

    # ---- the most common confusion pairs: are they driven by neighbours? ----
    pairs = {}
    for r in rows:
        pairs.setdefault((r["true"], r["gcn_pred"]), []).append(r)
    print("\nMost common GCN confusions (true -> predicted):")
    for (t, p), rs in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:4]:
        share = sum(r["neighbours_in_pred_class"] for r in rs) / len(rs)
        mlp_same = sum(r["mlp_pred"] == p for r in rs) / len(rs)
        print(f"  {t:>10} -> {p:<10} n={len(rs):<4} mean share of neighbours in predicted class "
              f"{share:.2f} | MLP made the same mistake {100 * mlp_same:.0f}%")

    # ---- graph helped vs hurt ----
    helped = int((test & mlp_wrong & ~gcn_wrong).sum())
    hurt = int((test & ~mlp_wrong & gcn_wrong).sum())
    both = int((test & mlp_wrong & gcn_wrong).sum())
    print(f"\nTest pages: GCN fixed {helped} MLP errors | GCN broke {hurt} MLP-correct pages"
          f" | both wrong {both}")

    # ---- systematic? error rate by homophily and degree (test pages) ----
    h_edges = [0.0, 0.25, 0.5, 0.75, 1.0001]
    d_edges = [1, 2, 4, 8, 16, 10_000]
    h_t, d_t = homophily[test], deg[test]
    g_h, n_h = error_rate_by_bins(h_t, gcn_wrong[test], h_edges)
    m_h, _ = error_rate_by_bins(h_t, mlp_wrong[test], h_edges)
    g_d, n_d = error_rate_by_bins(d_t, gcn_wrong[test], d_edges)
    m_d, _ = error_rate_by_bins(d_t, mlp_wrong[test], d_edges)
    h_labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
    d_labels = ["1", "2-3", "4-7", "8-15", "16+"]

    print("\nError rate by node homophily (test):")
    for lab, n, g, m in zip(h_labels, n_h, g_h, m_h):
        print(f"  {lab:<9} n={n:<5} GCN {g:.3f}  MLP {m:.3f}")
    print("Error rate by degree (test):")
    for lab, n, g, m in zip(d_labels, n_d, g_d, m_d):
        print(f"  {lab:<9} n={n:<5} GCN {g:.3f}  MLP {m:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, labels, g, m, sizes, title in (
            (axes[0], h_labels, g_h, m_h, n_h, "Error rate by node homophily"),
            (axes[1], d_labels, g_d, m_d, n_d, "Error rate by degree")):
        x = torch.arange(len(labels)).float()
        ax.bar(x - 0.2, m, width=0.4, label="MLP")
        ax.bar(x + 0.2, g, width=0.4, label="GCN")
        ax.set_xticks(x.tolist(), [f"{l}\n(n={n})" for l, n in zip(labels, sizes)])
        ax.set_ylabel("Test error rate")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/error_by_homophily_degree.png", dpi=150)
    plt.close(fig)

    with open("results/autopsy.json", "w") as f:
        json.dump({"top_confident_errors": rows[:args.top],
                   "gcn_fixed_mlp_errors": helped, "gcn_broke_mlp_correct": hurt,
                   "both_wrong": both,
                   "error_by_homophily": dict(zip(h_labels, zip(n_h, g_h, m_h))),
                   "error_by_degree": dict(zip(d_labels, zip(n_d, g_d, m_d)))}, f, indent=2)
    print("\nSaved figures/error_by_homophily_degree.png, results/gcn_errors.csv, results/autopsy.json")


if __name__ == "__main__":
    main()