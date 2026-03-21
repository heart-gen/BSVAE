# Hyperparameters

The active training workflow is configured through explicit CLI flags, primarily on `bsvae-train` and `bsvae-sweep-k`.

`hyperparam.ini` is not part of the active CLI path. The files in `src/bsvae/hyperparam.ini` and `docs/hyperparam.ini` are retained only for legacy context.

## Model And Architecture

- `--n-modules` default `20`
- `--latent-dim` default `32`
- `--hidden-dims` default `[512, 256, 128]`
- `--dropout` default `0.1`
- `--sigma-min` default `0.3`
- `--use-batch-norm` / `--no-batch-norm`
- `--normalize-input`

## Optimization And Schedule

- `--epochs` default `100`
- `--batch-size` default `128`
- `--lr` default `5e-4`
- `--checkpoint-every` default `10`
- `--warmup-epochs` default `20`
- `--transition-epochs` default `10`
- `--freeze-gmm-epochs` default `0`
- `--collapse-threshold` default `0.5`
- `--collapse-noise-scale` default `0.5`

## Loss Terms

- `--beta` default `1.0`
- `--free-bits` default `0.0` (bsvae-train), `0.5` (bsvae-sweep-k)
- `--kl-warmup-epochs` default `0`
- `--kl-anneal-mode` `linear` or `cyclical`
- `--kl-cycle-length` default `50`
- `--sep-strength` default `0.1`
- `--sep-alpha` default `2.0`
- `--bal-strength` default `0.1`
- `--bal-ema-blend` default `0.5`
- `--pi-entropy-strength` default `0.0`
- `--hier-strength` default `0.0`
- `--corr-strength` default `0.0`
- `--latent-corr-strength` default `0.0`
- `--masked-recon`

## Hierarchical And Isoform Options

- `--tx2gene` supplies transcript-to-gene mappings
- `--isoform-stratified` changes batch construction so related isoforms co-occur
- `--p-multi` controls the probability of sampling multi-isoform features when stratified sampling is enabled

## Practical Starting Points

Small pilot:

```bash
bsvae-train pilot \
  --dataset data/expression.csv \
  --n-modules 8 \
  --latent-dim 12 \
  --epochs 30
```

Typical baseline:

```bash
bsvae-train study1 \
  --dataset data/expression.csv \
  --n-modules 20 \
  --latent-dim 32 \
  --free-bits 0.5 \
  --sep-strength 0.1 \
  --bal-strength 0.01
```

Model-selection workflow:

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```
