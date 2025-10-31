#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="${BSVAE_CONFIG:-${REPO_ROOT}/src/bsvae/hyperparam.ini}"
BASE_SECTION="${BSVAE_BASE_SECTION:-beta_genenet}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

DATA_ARGS=()
if [[ -n "${BSVAE_GENE_DIR:-}" ]]; then
    if [[ ! -d "${BSVAE_GENE_DIR}" ]]; then
        echo "Gene expression directory not found: ${BSVAE_GENE_DIR}" >&2
        exit 1
    fi
    DATA_ARGS+=("--gene-expression-dir" "${BSVAE_GENE_DIR}")
elif [[ -n "${BSVAE_GENE_FILE:-}" ]]; then
    if [[ ! -f "${BSVAE_GENE_FILE}" ]]; then
        echo "Gene expression file not found: ${BSVAE_GENE_FILE}" >&2
        exit 1
    fi
    DATA_ARGS+=("--gene-expression-filename" "${BSVAE_GENE_FILE}")
else
    cat >&2 <<'MSG'
Set BSVAE_GENE_DIR or BSVAE_GENE_FILE to point to gene expression data.
BSVAE_GENE_DIR should contain X_train.csv and X_test.csv.
MSG
    exit 1
fi

IFS=' ' read -r -a BETA_VALUES <<< "${BSVAE_BETA_VALUES:-0.5 1.0 2.0 4.0 8.0}"
if [[ ${#BETA_VALUES[@]} -eq 0 ]]; then
    echo "No beta values supplied via BSVAE_BETA_VALUES." >&2
    exit 1
fi

for beta in "${BETA_VALUES[@]}"; do
    safe_beta="${beta//./p}"
    exp_name="${BSVAE_EXPERIMENT_PREFIX:-}${BASE_SECTION}_beta${safe_beta}"
    echo "Running beta sweep: beta=${beta} (experiment ${exp_name})"

    "${PYTHON_BIN}" -m bsvae.main "${exp_name}" \
        --config "${CONFIG_PATH}" \
        --section "${BASE_SECTION}" \
        --loss beta \
        --beta "${beta}" \
        "${DATA_ARGS[@]}" \
        "$@"
done
