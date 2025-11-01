# Contributing

We welcome pull requests that extend the BSVAE modeling framework, add datasets, or improve documentation.

## Repository Layout
- `src/bsvae/models/` – Encoder, decoder, and StructuredFactorVAE implementations.
- `src/bsvae/utils/` – Training loops, evaluation utilities, PPI helpers, and dataset loaders.
- `bin/` – Convenience scripts for plotting and metric aggregation.
- `tests/` – Unit tests covering loaders, training utilities, and CLI argument parsing.

## Coding Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions for Python code.
- Document public functions and classes using NumPy-style docstrings.
- Include type hints for new APIs where practical.

## Testing
Run the full test suite before submitting changes:
```bash
pytest -q
```
Add targeted tests in `tests/` for new functionality.

## Experiment Presets
To add a new preset:
1. Edit `src/bsvae/hyperparam.ini` (or a project-specific copy).
2. Create a new `[section_name]` block describing your experiment.
3. Document the preset in the README or relevant docs page.

## Extending the CLI
When introducing new CLI arguments:
1. Modify `parse_arguments` in `src/bsvae/main.py`.
2. Provide defaults in the relevant config section.
3. Update `docs/cli.md` and `docs/usage.md` to describe the new flag.
4. Add tests under `tests/` that cover parsing and execution paths.

## Submitting Changes
1. Fork the repository and create a feature branch.
2. Commit changes with descriptive messages.
3. Open a pull request summarizing the motivation and testing evidence.
4. Respond to review feedback promptly.

💡 **Tip:** For large features, open an issue first to discuss design choices with maintainers.
