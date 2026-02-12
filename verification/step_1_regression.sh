#!/bin/bash
#SBATCH --job-name=bsvae-regression
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=verification/step_1_regression_%j.out

# Detect cluster and set up environment
if [[ "$HOSTNAME" == *"quest"* ]] || [[ -d "/projects/p32505" ]]; then
    echo "=== Running on Quest ==="
    #SBATCH --account=p32505
    #SBATCH --partition=gengpu
    #SBATCH --gres=gpu:a100:1
    module load python/3.12.10
    source /projects/p32505/opt/venv/.ai_env/bin/activate
    DATA_PATH="/projects/b1213/users/kynon/projects/gnvae-example/input/simulation_data/_m/results/N150_batch0.0_nl0.0_zi0.0_seed13/log2rpkm.tsv.gz"
elif [[ "$HOSTNAME" == *"bridges"* ]] || [[ -d "/ocean/projects" ]]; then
    echo "=== Running on Bridges-2 ==="
    #SBATCH --partition=GPU-shared
    #SBATCH --gpus=v100-32:1
    module load anaconda3/2024.10-1
    conda activate /ocean/projects/bio250020p/shared/opt/env/network
    DATA_PATH="/ocean/projects/bio250020p/kbenjamin/projects/gnvae-example/input/simulation_data/_m/results/N150_batch0.0_nl0.0_zi0.0_seed13/log2rpkm.tsv.gz"
else
    echo "=== Running locally ==="
    DATA_PATH="/projects/b1213/users/kynon/projects/gnvae-example/input/simulation_data/_m/results/N150_batch0.0_nl0.0_zi0.0_seed13/log2rpkm.tsv.gz"
fi

set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="verification/regression_output"
mkdir -p "$OUTDIR"

echo "=== Regression test: old-equivalent settings ==="
echo "Using data: $DATA_PATH"

bsvae-train regression_test \
    --outdir "$OUTDIR" \
    --gene-expression-filename "$DATA_PATH" \
    --latent-dim 8 \
    --hidden-dims "[128, 64]" \
    --beta 2 \
    --epochs 50 \
    --batch-size 64 \
    --kl-warmup-epochs 0 \
    --free-bits 0.0 \
    --no-batch-norm \
    --no-test \
    --seed 13

echo ""
echo "=== Training complete. Checking losses... ==="
python verification/step_3_check_losses.py "$OUTDIR/regression_test/train_losses.csv" --mode regression

echo ""
echo "=== Regression test done ==="
