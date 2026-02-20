# Contributing

## Repository layout

- `src/bsvae/models/`: `GMMModuleVAE`, encoder/decoder/prior/losses
- `src/bsvae/cli/`: command entry points (`train`, `networks`, `simulate`)
- `src/bsvae/networks/`: extraction and module utilities
- `src/bsvae/utils/`: datasets, training loops, I/O, helpers
- `tests/`: unit tests

## Development setup

```bash
pip install -e .
pytest -q
```

## Change checklist

1. Update code and tests together.
2. Keep CLI docs in sync (`README.md`, `docs/cli.md`, `docs/quickstart.md`).
3. Avoid introducing behavior that diverges from documented defaults.

## Pull requests

Include:

- What changed
- Why it changed
- How it was tested
