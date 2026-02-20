# High-Performance Computing

## Single SLURM job

```bash
#!/bin/bash
#SBATCH --job-name=bsvae-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/.venv/bsvae/bin/activate

bsvae-train run1 \
  --dataset /scratch/data/expression.csv \
  --epochs 100 \
  --batch-size 256
```

## Array sweep over module counts

```bash
#!/bin/bash
#SBATCH --array=0-3

K_VALUES=(8 12 16 24)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

bsvae-train run_k${K} \
  --dataset /scratch/data/expression.csv \
  --n-modules ${K}
```

## CPU-only fallback

```bash
bsvae-train run_cpu --dataset /scratch/data/expression.csv --no-cuda
```

## Notes

- Output directories are created at `<outdir>/<name>`.
- Checkpoints are written as `model-<epoch>.pt`.
- Post-processing (`bsvae-networks`) can run in separate CPU jobs.
