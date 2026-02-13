#!/bin/bash
#SBATCH --job-name=bsvae-tests
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=verification/step_0_tests_%j.out

# Detect cluster and set up environment
if [[ "$HOSTNAME" == *"quest"* ]] || [[ -d "/projects/p32505" ]]; then
    echo "=== Running on Quest ==="
    #SBATCH --account=p32505
    #SBATCH --partition=short
    module load python/3.12.10
    source /projects/p32505/opt/venv/.ai_env/bin/activate
elif [[ "$HOSTNAME" == *"bridges"* ]] || [[ -d "/ocean/projects" ]]; then
    echo "=== Running on Bridges-2 ==="
    #SBATCH --partition=RM-shared
    module load anaconda3/2024.10-1
    conda activate /ocean/projects/bio250020p/shared/opt/env/network
else
    echo "=== Running locally ==="
fi

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running BSVAE unit tests..."
python -m pytest tests/ -v --tb=short 2>&1

echo ""
echo "=== All tests passed ==="
