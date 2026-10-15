# Hyperparameters

The active training workflow is configured through explicit CLI flags, primarily on `bsvae-train` and `bsvae-sweep-k`.

`hyperparam.ini` is not part of the active CLI path; the legacy ini files have been removed.

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
- `--collapse-min-usage` default `0.002` (absolute; replaces the relative `--collapse-threshold`, now ignored)
- `--collapse-noise-scale` default `0.5`

## Loss Terms

- `--beta` default `1.0`; the KL is scaled per observed value (`--no-kl-per-element` restores the legacy sum over latent dimensions, whose effective β grows with the number of samples)
- `--free-bits` default `0.0` (both `bsvae-train` and `bsvae-sweep-k`)
- `--kl-warmup-epochs` default `0` (counted from the start of the GMM phase)
- `--kl-anneal-mode` `linear` or `cyclical`
- `--kl-cycle-length` default `50`
- `--sep-strength` default `0.1`
- `--sep-alpha` default `2.0`
- `--bal-strength` default `0.1`
- `--bal-type` default `entropy` (discourages dead components without forcing equal module sizes)
- `--bal-ema-blend` default `0.5`
- `--pi-entropy-strength` default `0.0`
- `--hier-strength` default `0.0`
- `--hier-margin` default `1.0`, `--no-hier-corr-weight`
- `--corr-strength` default `0.0`
- `--latent-corr-strength` default `0.0`
- `--masked-recon`

## Hierarchical And Isoform Options

- `--tx2gene` supplies transcript-to-gene mappings (header optional; version suffixes stripped unless `--keep-tx-version`)
- `--isoform-stratified` builds batches gene by gene so all isoforms of a gene co-occur
- `--p-multi` controls the probability that a batch is filled from complete gene groups
- the hierarchical loss is a hinge (`--hier-margin`) weighted by the positive expression correlation of each isoform pair, so isoforms that switch modules are not forced together

## Post-processing (extract-modules)

- `--min-gamma` default `0.5`: features below this top responsibility are unassigned (grey)
- `--min-module-size` default `10`
- `--merge-cut-height` default `0.4`: merge modules whose eigengene correlation exceeds `0.6` (GMM components that split one module have noisier eigengenes than a WGCNA tree cut, so the cut is looser than WGCNA's 0.25)
- `--kme-reassign` with `--min-kme` `0.3` and `--kme-margin` `0.1`

## Choosing K

`--n-modules` is the number of GMM components, not the final module count. Train with about three times the number of modules you expect (at least 20) and let `extract-modules` merge components whose eigengenes correlate; on the reference simulations this recovered small modules that a K equal to the true count absorbed into large ones, and raised ARI by 0.1–0.2. `bsvae-sweep-k` remains useful for picking the component count by held-out ELBO.

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
  --n-modules 48 \
  --latent-dim 32 \
  --normalize-input \
  --beta 2.0
```

(`--n-modules 48` for roughly 16 expected modules; post-processing merges the rest.)

Transcript-level data:

```bash
bsvae-train iso1 \
  --dataset data/transcripts.csv \
  --tx2gene data/tx2gene.tsv \
  --n-modules 48 \
  --normalize-input \
  --beta 2.0 \
  --hier-strength 0.05 --isoform-stratified
```

On the reference simulations (`scripts/benchmark_sims.py`), `--beta` between 1 and 2 with `--normalize-input` and no latent-correlation loss recovered modules best; `--latent-corr-strength 0.5`, which earlier runs used to counteract the over-weighted KL, lowers ARI now that the KL is scaled per element. The hierarchical loss trades isoform-split detection for within-gene agreement: `--hier-strength 0.05` keeps most switched isoforms separable, while disabling `--hier-corr-weight` forces isoforms together and halves split detection.

Model-selection workflow:

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```
