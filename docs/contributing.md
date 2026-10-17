# Contributing

## Repository layout

- `src/bsvae/models/`: `GMMModuleVAE`, encoder/decoder/prior/losses
- `src/bsvae/cli/`: command entry points (`train`, `sweep_k`, `networks`, `simulate`, `benchmark`)
- `src/bsvae/networks/`: network and module extraction, post-processing, module quality, targeted-gene tests
- `src/bsvae/latent/`: latent export and analysis
- `src/bsvae/simulation/`: gene- and transcript-level simulators and the grey-aware metrics
- `src/bsvae/benchmark/`: the benchmark harness, comparison methods and the packaged R scripts (`wgcna.R`, `acorde.R`, `twn.R`)
- `src/bsvae/utils/`: datasets, training loops, I/O, tx2gene handling, isoform calls, covariate regression
- `scripts/`: benchmark and calibration scripts (see [Benchmarking](benchmarking.md))
- `tests/`: unit tests

## Development setup

```bash
poetry install --with dev --extras "anndata yaml"
poetry run pytest -q        # slow tests are deselected by default; add -m slow to run them
```

## Change checklist

1. Update code and tests together.
2. Keep CLI docs in sync (`README.md`, `docs/cli.md`, `docs/quickstart.md`).
3. Avoid introducing behavior that diverges from documented defaults. `tests/test_docs_defaults.py` fails when a ``- `--flag` ... default `X` `` line in `docs/cli.md` or `docs/hyperparameters.md` no longer matches the parser, so keep that line format for documented defaults.
4. Add a `CHANGELOG.md` entry under the next version.

## Pull requests

Include:

- What changed
- Why it changed
- How it was tested

## Releasing

Versions come from git tags (`poetry-dynamic-versioning`); `pyproject.toml` keeps `version = "0.0.0"`.

1. Move the `CHANGELOG.md` entries to a new `## X.Y.Z - date` heading and merge to `main`.
2. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The `Release` workflow (`.github/workflows/release.yml`) runs the tests, builds with `poetry build`, checks that the wheel version matches the tag, publishes to PyPI through trusted publishing, and creates the GitHub release. Read the Docs rebuilds `latest` from `main` and `stable` from the tag.

To build locally, tag first (or build on a tagged commit) so the dynamic version resolves, then run `poetry build` and inspect `dist/`.
