"""
dilemma.py - The Confidence and Ambiguity Dilemma (spec section 2.1).

Question: can we trust each model's confidence enough to auto-handle
high-confidence pages and route uncertain ones to human reviewers?

Steps:
    1. Confidence of correct vs incorrect test predictions (is the model
       less sure when it is wrong, or stubbornly overconfident?)
    2. Reliability diagrams + Expected Calibration Error (ECE)
    3. Temperature scaling, fitted on the VALIDATION split only [Guo et al., 2017]
    4. Coverage vs accuracy curve (reject option)
    5. Pick a confidence threshold on VALIDATION that reaches a target accuracy
       on auto-handled pages, then report what happens on TEST

Run after train.py:
    python dilemma.py

Outputs:
    figures/confidence_histograms.png
    figures/reliability_diagrams.png
    figures/coverage_accuracy.png
    results/dilemma.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import get_data
from evaluate import load_model


# ---------------------------------------------------------------------------
# Calibration tools
# ---------------------------------------------------------------------------

def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: average gap between confidence and accuracy across confidence bins.

    Predictions are grouped into bins by confidence (e.g. 0.90-0.97). In each
    bin we compare the mean confidence with the actual accuracy; ECE is the
    weighted average of those gaps (weights = fraction of samples in the bin).
    0 = perfectly calibrated.

    Returns:
        ece (float), per-bin (mean confidence, accuracy, count) for plotting.
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            bins.append((None, None, 0))
            continue
        bin_conf = conf[in_bin].mean().item()
        bin_acc = correct[in_bin].mean().item()
        ece += (n / len(labels)) * abs(bin_acc - bin_conf)
        bins.append((bin_conf, bin_acc, n))
    return ece, bins


def fit_temperature(val_logits, val_labels):
    """Find the temperature T > 0 that minimises validation negative log-likelihood.

    Calibrated probabilities are softmax(logits / T). T > 1 softens
    (less confident), T < 1 sharpens. Dividing every logit by the same T never
    changes which class is largest, so accuracy is unchanged - only confidence.
    A simple grid search is used so the procedure is transparent.
    """
    temps = torch.linspace(0.25, 5.0, 476)  # step 0.01
    nll = torch.stack([F.cross_entropy(val_logits / t, val_labels) for t in temps])
    return temps[nll.argmin()].item()


def coverage_accuracy(probs, labels, thresholds):
    """For each threshold t: coverage = fraction of pages with confidence >= t
    (auto-handled), accuracy = accuracy on those pages only."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    cov, acc = [], []
    for t in thresholds:
        keep = conf >= t
        cov.append(keep.float().mean().item())
        acc.append(correct[keep].mean().item() if keep.any() else float("nan"))
    return cov, acc


def choose_threshold(probs, labels, target_acc):
    """Lowest confidence threshold whose auto-handled accuracy >= target_acc,
    i.e. the threshold that automates the MOST pages while meeting the target.
    Chosen on validation data only."""
    thresholds = torch.linspace(0.25, 0.999, 750)
    cov, acc = coverage_accuracy(probs, labels, thresholds)
    for t, c, a in zip(thresholds.tolist(), cov, acc):
        if a == a and a >= target_acc:  # a == a filters NaN
            return t
    return None


def threshold_report(probs, labels, t):
    """Coverage, auto-handled accuracy and routed fraction at threshold t."""
    conf, pred = probs.max(dim=1)
    keep = conf >= t
    return {
        "threshold": t,
        "coverage": keep.float().mean().item(),
        "routed_to_human": 1 - keep.float().mean().item(),
        "auto_accuracy": (pred[keep] == labels[keep]).float().mean().item(),
        "routed_accuracy": (pred[~keep] == labels[~keep]).float().mean().item()
        if (~keep).any() else None,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confidence_histograms(test_probs, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, probs) in zip(axes, test_probs.items()):
        conf, pred = probs.max(dim=1)
        right = (pred == y_test)
        bins = torch.linspace(0.25, 1.0, 31).tolist()
        ax.hist(conf[right].numpy(), bins=bins, alpha=0.6, label=f"correct (n={int(right.sum())})")
        ax.hist(conf[~right].numpy(), bins=bins, alpha=0.8, label=f"incorrect (n={int((~right).sum())})")
        ax.set_yscale("log")
        ax.set_xlabel("Confidence (max softmax probability)")
        ax.set_title(f"{name.upper()} - confidence on test predictions")
        ax.legend()
    axes[0].set_ylabel("Number of pages (log scale)")
    fig.tight_layout()
    fig.savefig("figures/confidence_histograms.png", dpi=150)
    plt.close(fig)


def plot_reliability(curves):
    """curves: {label: (bins, ece)} -> one reliability diagram per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ("mlp", "gcn")):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for label, (bins, ece) in curves.items():
            if not label.startswith(model):
                continue
            xs = [b[0] for b in bins if b[2] > 0]
            ys = [b[1] for b in bins if b[2] > 0]
            ax.plot(xs, ys, "o-", label=f"{label.split('_', 1)[1]} (ECE {ece:.3f})")
        ax.set_xlim(0.25, 1)
        ax.set_ylim(0.25, 1)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Accuracy in bin")
        ax.set_title(f"{model.upper()} reliability diagram (test)")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/reliability_diagrams.png", dpi=150)
    plt.close(fig)


def plot_coverage(test_probs, y_test, chosen, target):
    thresholds = torch.linspace(0.25, 0.999, 300)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, probs in test_probs.items():
        cov, acc = coverage_accuracy(probs, y_test, thresholds)
        ax.plot(cov, acc, label=name.upper())
        if chosen.get(name):
            r = chosen[name]
            ax.scatter([r["coverage"]], [r["auto_accuracy"]], zorder=5)
            ax.annotate(f"t={r['threshold']:.2f}\n{100 * r['coverage']:.0f}% automated",
                        (r["coverage"], r["auto_accuracy"]),
                        textcoords="offset points", xytext=(-80, -30))
    ax.axhline(target, color="grey", linestyle="--", lw=1, label=f"target accuracy {target}")
    ax.set_xlabel("Coverage (fraction of pages auto-handled)")
    ax.set_ylabel("Accuracy on auto-handled pages")
    ax.set_title("Reject option: coverage vs accuracy (test, calibrated)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/coverage_accuracy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibration and reject-option analysis")
    parser.add_argument("--data", default="data/facebook.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_acc", type=float, default=0.97,
                        help="required accuracy on auto-handled pages (feasibility criterion)")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = get_data(args.data, seed=args.seed)
    data = data.to(device)
    y_val, y_test = data.y[data.val_mask].cpu(), data.y[data.test_mask].cpu()

    results, curves, test_probs_cal, chosen = {}, {}, {}, {}
    raw_test_probs = {}
    for name in ("mlp", "gcn"):
        model = load_model(name, data, device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index).cpu()
        val_logits, test_logits = logits[data.val_mask.cpu()], logits[data.test_mask.cpu()]

        # 1-2. uncalibrated confidence + ECE
        p_raw = F.softmax(test_logits, dim=1)
        raw_test_probs[name] = p_raw
        ece_raw, bins_raw = expected_calibration_error(p_raw, y_test)

        # 3. temperature scaling (fit on validation, apply to test)
        T = fit_temperature(val_logits, y_val)
        p_cal = F.softmax(test_logits / T, dim=1)
        p_val_cal = F.softmax(val_logits / T, dim=1)
        ece_cal, bins_cal = expected_calibration_error(p_cal, y_test)
        test_probs_cal[name] = p_cal
        curves[f"{name}_uncalibrated"] = (bins_raw, ece_raw)
        curves[f"{name}_temperature-scaled (T={T:.2f})"] = (bins_cal, ece_cal)

        # confidence of right vs wrong predictions (calibration-independent ranking info)
        conf, pred = p_raw.max(dim=1)
        right = pred == y_test
        wrong_conf = conf[~right]

        # 5. threshold chosen on VALIDATION, evaluated on TEST
        t = choose_threshold(p_val_cal, y_val, args.target_acc)
        chosen[name] = threshold_report(p_cal, y_test, t) if t is not None else None

        results[name] = {
            "temperature": T,
            "ece_uncalibrated": ece_raw,
            "ece_temperature_scaled": ece_cal,
            "mean_conf_correct": conf[right].mean().item(),
            "mean_conf_incorrect": wrong_conf.mean().item(),
            "errors_with_conf_over_0.9": int((wrong_conf > 0.9).sum()),
            "total_errors": int((~right).sum()),
            "threshold_policy": chosen[name],
        }

    # ---- print summary ----
    print(f"\n{'':<34}{'MLP':>10}{'GCN':>10}")
    rows = [("ECE (uncalibrated)", "ece_uncalibrated", ".3f"),
            ("Temperature T (fit on val)", "temperature", ".2f"),
            ("ECE (temperature-scaled)", "ece_temperature_scaled", ".3f"),
            ("Mean confidence when correct", "mean_conf_correct", ".3f"),
            ("Mean confidence when wrong", "mean_conf_incorrect", ".3f"),
            ("Errors with confidence > 0.9", "errors_with_conf_over_0.9", "d"),
            ("Total test errors", "total_errors", "d")]
    for label, key, fmt in rows:
        print(f"{label:<34}{results['mlp'][key]:>10{fmt}}{results['gcn'][key]:>10{fmt}}")

    print(f"\nReject option (target {args.target_acc} accuracy on auto-handled pages;"
          f" threshold chosen on VALIDATION, reported on TEST):")
    for name in ("mlp", "gcn"):
        r = chosen[name]
        if r is None:
            print(f"  {name.upper()}: target not reachable on validation")
            continue
        routed_acc = "n/a" if r["routed_accuracy"] is None else f"{r['routed_accuracy']:.3f}"
        print(f"  {name.upper()}: threshold {r['threshold']:.3f} | automated {100 * r['coverage']:.1f}%"
              f" | routed to humans {100 * r['routed_to_human']:.1f}%"
              f" | auto accuracy {r['auto_accuracy']:.3f}"
              f" | accuracy on routed pages {routed_acc}")

    plot_confidence_histograms(raw_test_probs, y_test)
    plot_reliability(curves)
    plot_coverage(test_probs_cal, y_test, chosen, args.target_acc)
    with open("results/dilemma.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/confidence_histograms.png, figures/reliability_diagrams.png,"
          " figures/coverage_accuracy.png, results/dilemma.json")


if __name__ == "__main__":
    main()