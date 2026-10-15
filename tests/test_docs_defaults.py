"""Documented CLI defaults must match the parsers (docs/cli.md and docs/hyperparameters.md)."""

import re
from pathlib import Path

import pytest

from bsvae.cli import benchmark as bench_cli
from bsvae.cli import networks as networks_cli
from bsvae.cli import simulate as simulate_cli
from bsvae.cli import sweep_k as sweep_cli
from bsvae.cli import train as train_cli

DOCS = Path(__file__).resolve().parents[1] / "docs"
LINE_RE = re.compile(r"^- `(--[a-z0-9-]+)`.*?default `([^`]+)`", re.MULTILINE)


def _parser_defaults(parser):
    out = {}
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                out[opt] = action.default
    return out


def _subparser(parser, name):
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict) and name in action.choices:
            return action.choices[name]
    raise KeyError(name)


def _section(text, heading, next_level="## "):
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = rest.find("\n" + next_level)
    return rest if end < 0 else rest[:end]


def _norm(v):
    text = str(v).replace(" ", "")
    try:
        return repr(float(text))
    except ValueError:
        return text


def _check(doc_text, defaults, where):
    problems = []
    for flag, documented in LINE_RE.findall(doc_text):
        if flag not in defaults:
            problems.append(f"{where}: {flag} is documented but the parser has no such flag")
            continue
        if _norm(documented) != _norm(defaults[flag]):
            problems.append(f"{where}: {flag} documented default {documented!r} != parser default {defaults[flag]!r}")
    assert not problems, "\n".join(problems)


def _train_parser():
    import argparse
    from bsvae.cli._common import add_general_args, add_loss_args, add_model_args, add_schedule_args
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    add_general_args(p); add_schedule_args(p, 10); add_model_args(p); add_loss_args(p)
    return p


def test_cli_md_train_defaults():
    text = (DOCS / "cli.md").read_text()
    _check(_section(text, "## `bsvae-train`"), _parser_defaults(_train_parser()), "cli.md bsvae-train")


def test_cli_md_sweep_defaults():
    text = (DOCS / "cli.md").read_text()
    import argparse
    p = argparse.ArgumentParser()
    # Rebuild the sweep parser without the required positional/dataset arguments.
    sweep_cli.parse_args.__globals__  # noqa: B018  (import side effects only)
    full = sweep_cli.parse_args(["x", "--dataset", "d"])
    defaults = {("--" + k.replace("_", "-")): v for k, v in vars(full).items()}
    _check(_section(text, "## `bsvae-sweep-k`"), defaults, "cli.md bsvae-sweep-k")


def test_cli_md_networks_defaults():
    text = (DOCS / "cli.md").read_text()
    parser = networks_cli.build_parser()
    for sub in ("extract-networks", "extract-modules", "export-latents", "latent-analysis"):
        _check(_section(text, f"### `{sub}`", next_level="##"), _parser_defaults(_subparser(parser, sub)), f"cli.md {sub}")


def test_cli_md_simulate_defaults():
    text = (DOCS / "cli.md").read_text()
    parser = simulate_cli.build_parser()
    for sub in ("generate", "benchmark"):
        _check(_section(text, f"### `{sub}`", next_level="##"), _parser_defaults(_subparser(parser, sub)), f"cli.md {sub}")


def test_hyperparameters_md_defaults():
    text = (DOCS / "hyperparameters.md").read_text()
    defaults = _parser_defaults(_train_parser())
    # Post-processing flags live on extract-modules.
    defaults.update(_parser_defaults(_subparser(networks_cli.build_parser(), "extract-modules")))
    _check(text, defaults, "hyperparameters.md")
