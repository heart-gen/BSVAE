# Hyperparameters

BSVAE now configures training directly through `bsvae-train` CLI flags.

`hyperparam.ini` is deprecated in both:

- `src/bsvae/hyperparam.ini`
- `docs/hyperparam.ini`

Those files are retained only as compatibility placeholders and are not parsed by the active CLI entry point.

## Model/structure

- `--n-modules` (default `20`)
- `--latent-dim` (default `32`)
- `--hidden-dims` (default `[512, 256, 128]`)
- `--dropout` (default `0.1`)
- `--sigma-min` (default `0.3`)
- `--use-batch-norm` / `--no-batch-norm`

## Optimization

- `--epochs` (default `100`)
- `--batch-size` (default `128`)
- `--lr` (default `5e-4`)
- `--checkpoint-every` (default `10`)
- `--warmup-epochs` (default `20`)
- `--transition-epochs` (default `10`)

## Loss terms

- `--beta` (default `1.0`)
- `--free-bits` (default `0.5`)
- `--kl-warmup-epochs` (default `0`)
- `--kl-anneal-mode` (`linear` or `cyclical`)
- `--kl-cycle-length` (default `50`)
- `--sep-strength` (default `0.1`)
- `--sep-alpha` (default `2.0`)
- `--bal-strength` (default `0.01`)
- `--hier-strength` (default `0.0`)
- `--tx2gene` (required when hierarchical loss is enabled)

## Practical starting points

- Small pilot: `--n-modules 8 --latent-dim 12 --epochs 30`
- Typical run: defaults + adjust `--n-modules`
- Stabilize KL: keep `--free-bits 0.5`, use `--kl-anneal-mode linear`
