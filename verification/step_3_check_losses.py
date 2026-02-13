#!/usr/bin/env python
"""
Automated loss analysis for BSVAE verification.

Parses train_losses.csv and validates:
- kl_loss at final epoch > threshold
- effective_beta follows expected annealing curve
- All kl_dim_* > threshold (no collapsed dimensions)
- recon_loss decreased from epoch 0
"""
import argparse
import csv
import sys
from collections import defaultdict


def parse_losses_csv(filepath):
    """Parse train_losses.csv into {loss_name: {epoch: value}} dict."""
    losses = defaultdict(dict)
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            try:
                epoch = int(row[0])
                name = row[1]
                value = float(row[2])
                losses[name][epoch] = value
            except (ValueError, IndexError):
                continue
    return losses


def check_regression(losses):
    """Validate regression (old-equivalent) run."""
    results = []
    passed = True

    # Check that effective_beta is constant (should be 2.0 with warmup=0)
    if "effective_beta" in losses:
        betas = list(losses["effective_beta"].values())
        beta_constant = all(abs(b - betas[0]) < 1e-6 for b in betas)
        results.append(("effective_beta constant", "PASS" if beta_constant else "FAIL",
                        f"values range: [{min(betas):.4f}, {max(betas):.4f}]"))
        if not beta_constant:
            passed = False

    # Check recon_loss decreased
    if "recon_loss" in losses:
        epochs = sorted(losses["recon_loss"].keys())
        if len(epochs) >= 2:
            first = losses["recon_loss"][epochs[0]]
            last = losses["recon_loss"][epochs[-1]]
            decreased = last < first
            results.append(("recon_loss decreased", "PASS" if decreased else "FAIL",
                            f"{first:.4f} -> {last:.4f}"))
            if not decreased:
                passed = False

    return passed, results


def check_anticollapse(losses):
    """Validate anti-collapse run with new settings."""
    results = []
    passed = True

    # Check kl_loss at final epoch > 0.1
    if "kl_loss" in losses:
        epochs = sorted(losses["kl_loss"].keys())
        if epochs:
            final_kl = losses["kl_loss"][epochs[-1]]
            kl_ok = final_kl > 0.1
            results.append(("kl_loss > 0.1 at final epoch", "PASS" if kl_ok else "FAIL",
                            f"kl_loss={final_kl:.6f}"))
            if not kl_ok:
                passed = False

    # Check effective_beta follows annealing (should start at 0 and ramp up)
    if "effective_beta" in losses:
        epochs = sorted(losses["effective_beta"].keys())
        if len(epochs) >= 2:
            first_beta = losses["effective_beta"][epochs[0]]
            last_beta = losses["effective_beta"][epochs[-1]]
            annealed = last_beta > first_beta or (first_beta == 0 and last_beta > 0)
            results.append(("effective_beta annealed", "PASS" if annealed else "FAIL",
                            f"{first_beta:.4f} -> {last_beta:.4f}"))
            if not annealed:
                passed = False

    # Check per-dimension KL (all kl_dim_* > 0.01)
    dim_keys = [k for k in losses if k.startswith("kl_dim_")]
    if dim_keys:
        collapsed_dims = []
        for dk in sorted(dim_keys):
            epochs = sorted(losses[dk].keys())
            if epochs:
                final_val = losses[dk][epochs[-1]]
                if final_val < 0.01:
                    collapsed_dims.append((dk, final_val))

        all_healthy = len(collapsed_dims) == 0
        if collapsed_dims:
            detail = "; ".join(f"{d}={v:.6f}" for d, v in collapsed_dims)
            results.append(("all kl_dim_* > 0.01", "FAIL", f"collapsed: {detail}"))
        else:
            results.append(("all kl_dim_* > 0.01", "PASS",
                            f"{len(dim_keys)} dims all healthy"))
        if not all_healthy:
            passed = False

    # Check recon_loss decreased
    if "recon_loss" in losses:
        epochs = sorted(losses["recon_loss"].keys())
        if len(epochs) >= 2:
            first = losses["recon_loss"][epochs[0]]
            last = losses["recon_loss"][epochs[-1]]
            decreased = last < first
            results.append(("recon_loss decreased", "PASS" if decreased else "FAIL",
                            f"{first:.4f} -> {last:.4f}"))
            if not decreased:
                passed = False

    return passed, results


def main():
    parser = argparse.ArgumentParser(description="Check BSVAE training losses")
    parser.add_argument("csv_path", help="Path to train_losses.csv")
    parser.add_argument("--mode", choices=["regression", "anticollapse"],
                        default="anticollapse",
                        help="Validation mode (default: anticollapse)")
    args = parser.parse_args()

    losses = parse_losses_csv(args.csv_path)

    if not losses:
        print(f"FAIL: No losses found in {args.csv_path}")
        sys.exit(1)

    print(f"Loaded losses from {args.csv_path}")
    print(f"Loss keys: {sorted(losses.keys())}")
    print(f"Epochs: {min(min(v.keys()) for v in losses.values())} - "
          f"{max(max(v.keys()) for v in losses.values())}")
    print()

    if args.mode == "regression":
        passed, results = check_regression(losses)
    else:
        passed, results = check_anticollapse(losses)

    print(f"=== {args.mode.upper()} VALIDATION ===")
    for name, status, detail in results:
        print(f"  [{status}] {name}: {detail}")
    print()

    if passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
