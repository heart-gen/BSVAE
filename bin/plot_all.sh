#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src"
PYTHON_BIN="${PYTHON:-python}"
OUTPUT_FILE="${BSVAE_SUMMARY_FILE:-${REPO_ROOT}/results/metrics_summary.csv}"

IFS=' ' read -r -a SECTIONS <<< "${BSVAE_SUMMARY_SECTIONS:-VAE_genenet beta_genenet}"
if [[ ${#SECTIONS[@]} -eq 0 ]]; then
    echo "No config sections supplied via BSVAE_SUMMARY_SECTIONS." >&2
    exit 1
fi

EXPERIMENTS=()
for section in "${SECTIONS[@]}"; do
    EXPERIMENTS+=("${BSVAE_EXPERIMENT_PREFIX:-}${section}")
done

LOGGER="${REPO_ROOT}/plot_all.out"
: > "${LOGGER}"

printf "Generating metrics summary for %s\n" "${EXPERIMENTS[*]}" | tee -a "${LOGGER}"

for exp in "${EXPERIMENTS[@]}"; do
    if [[ ! -f "${REPO_ROOT}/results/${exp}/test_losses.pt" ]]; then
        echo "Missing evaluation logs for ${exp}; skipping." | tee -a "${LOGGER}" >&2
    fi
done

"${PYTHON_BIN}" - <<'PY' "${OUTPUT_FILE}" "${REPO_ROOT}" "${LOGGER}" "${EXPERIMENTS[@]}"
import csv
import os
import sys
from typing import Dict, Any

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"torch is required to load evaluation logs: {exc}")

output_path = sys.argv[1]
repo_root = sys.argv[2]
logger_path = sys.argv[3]
experiments = sys.argv[4:]

results_dir = os.path.join(repo_root, "results")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

rows = []
missing = []
keys = set()
for exp in experiments:
    log_path = os.path.join(results_dir, exp, "test_losses.pt")
    if not os.path.exists(log_path):
        missing.append(exp)
        continue
    losses: Dict[str, Any] = torch.load(log_path, map_location="cpu")
    row = {"experiment": exp}
    for key, value in sorted(losses.items()):
        if hasattr(value, "item"):
            value = value.item()
        row[key] = float(value)
        keys.add(key)
    rows.append(row)

fieldnames = ["experiment"] + sorted(keys)
with open(output_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with open(logger_path, "a", encoding="utf-8") as log_file:
    if rows:
        log_file.write(f"\nWrote summary to {output_path}\n")
        for row in rows:
            summary = ", ".join(f"{k}={row.get(k, 'NA')}" for k in fieldnames)
            log_file.write(summary + "\n")
    else:
        log_file.write("\nNo evaluation logs were found.\n")
    if missing:
        log_file.write("Missing experiments: " + ", ".join(missing) + "\n")

if missing and len(missing) == len(experiments):
    raise SystemExit("No metrics available; nothing to summarise.")
PY

printf "Summary written to %s\n" "${OUTPUT_FILE}" | tee -a "${LOGGER}"
