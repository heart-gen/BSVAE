# High-Performance Computing

Run BSVAE on managed clusters by embedding the CLI into batch submission scripts. This page demonstrates common SLURM patterns, but the same principles apply to PBS, LSF, or other schedulers.

## Single Job Submission
```bash
#!/bin/bash
#SBATCH --job-name=bsvae-beta
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda
source ~/.venv/bsvae/bin/activate

bsvae-train run_beta4 --section beta_genenet --gene-expression-dir data/splits/
```

The experiment results are written to `results/run_beta4/` relative to the launch directory. Keep the working directory at the repository root or pass absolute paths to `--gene-expression-dir`.

## Array Jobs
Queue multiple configurations by leveraging section names or overrides inside an array job:

```bash
#!/bin/bash
#SBATCH --array=0-3
SECTIONS=(beta_genenet VAE_genenet Custom debug)
SECTION=${SECTIONS[$SLURM_ARRAY_TASK_ID]}

bsvae-train sweep_${SECTION} \
  --section ${SECTION} \
  --gene-expression-dir /scratch/datasets/genenet/
```

Each array task produces a separate directory such as `results/sweep_beta_genenet/` containing checkpoints and logs.

## Checkpoints and Resumption
Checkpoints are automatically emitted every `--checkpoint-every` epochs. To resume after a preemption, rerun the same command; the trainer detects existing weights in the experiment directory and continues training.

💡 **Tip:** Combine `--is-eval-only` with array jobs to post-process multiple experiments without reserving GPUs.
