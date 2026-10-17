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

## Cross-fit fold models

`isoform-specific --folds` needs one model per sample fold, each trained without that fold's samples. Run one array task per fold:

```bash
#!/bin/bash
#SBATCH --array=0-1
#SBATCH --gres=gpu:1

F=${SLURM_ARRAY_TASK_ID}
bsvae-train fold${F} --dataset data/transcripts.tsv.gz --n-modules 36 \
  --exclude-samples folds/fold_${F}.txt --tx2gene data/tx2gene.tsv

bsvae-networks extract-modules --model-path results/fold${F} \
  --dataset data/transcripts.tsv.gz --expr data/transcripts.tsv.gz \
  --tx2gene data/tx2gene.tsv --exclude-samples folds/fold_${F}.txt \
  --output-dir results/fold${F}/modules
```

## Benchmark arrays

The benchmark harness runs one replicate and method per array task, each reading one row of a TSV task table, so a whole study is a handful of `sbatch --array` submissions. See [Benchmarking](benchmarking.md#5-run-it-on-a-cluster) for the stage layout, the environment variables and the account-specific memory rule (`--cpus-per-task` sets the memory; do not pass `--mem` on PSC).

```bash
# from the bsvae-benchmarking repository
sbatch --array=1-5 01_recovery/_h/02a.harness.sh      # rows 1-5 of the task table
EPOCHS=150 SEEDS=10 bash 00_scripts/run_pipeline.sh   # submit every stage
```

## CPU-only fallback

```bash
bsvae-train run_cpu --dataset /scratch/data/expression.csv --no-cuda
```

## Notes

- Output directories are created at `<outdir>/<name>`.
- Checkpoints are written as `model-<epoch>.pt`.
- Post-processing (`bsvae-networks`) can run in separate CPU jobs.
