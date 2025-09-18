import os
import ast
import shutil
import random
import argparse
import numpy as np
import configparser
import logging

import torch


def create_safe_directory(directory, logger=None):
    """Create a directory and archive the previous one if it already exists."""
    if os.path.exists(directory):
        if logger is not None:
            warn = "Directory {} already exists. Archiving it to {}.zip"
            logger.warning(warn.format(directory, directory))
        else:
            print(f"Warning: Directory {directory} already exists. Archiving it to {directory}.zip")

        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu=True):
    """Return the appropriate device (CUDA if available and requested)."""
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def get_model_device(model):
    """Return the device on which the model is located."""
    return next(model.parameters()).device


def get_n_params(model):
    """Return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_namespace_(namespace, dictionary):
    """Update an argparse namespace in-place with a dictionary."""
    vars(namespace).update(dictionary)


def get_config_section(filenames, section):
    """
    Return a dictionary of a section from `.ini` config files.
    All values in the section are literally evaluated (e.g., l=[1,"as"] becomes a list).

    Parameters
    ----------
    filenames : list or str
        List of file paths or a single path to `.ini` config files.
    section : str
        Section name to retrieve.

    Returns
    -------
    dict
        Dictionary with config keys and their literal Python values.
    """
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.optionxform = str  # keep case-sensitive keys

    files = parser.read(filenames)
    if len(files) == 0:
        raise ValueError(f"Config files not found: {filenames}")

    if section not in parser:
        raise ValueError(f"Section '{section}' not found in config files: {filenames}")

    try:
        raw_dict = dict(parser[section])
        return {k: ast.literal_eval(v) for k, v in raw_dict.items()}
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing config section '{section}': {e}")


def check_bounds(value, type_cast=float, lb=-float("inf"), ub=float("inf"),
                 is_inclusive=True, name="value"):
    """
    Bound checker for argparse arguments.

    Parameters
    ----------
    value : any
        Input value from argparse.
    type_cast : type
        Type to cast value to (e.g., int or float).
    lb : float
        Lower bound.
    ub : float
        Upper bound.
    is_inclusive : bool
        Whether bounds are inclusive.
    name : str
        Name of the variable for error messages.

    Returns
    -------
    value : type
        The type-casted value if within bounds.

    Raises
    ------
    argparse.ArgumentTypeError
        If value is out of bounds or cannot be cast.
    """
    try:
        value = type_cast(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"{name} could not be cast to {type_cast}: {e}")

    in_bounds = lb <= value <= ub if is_inclusive else lb < value < ub
    if not in_bounds:
        raise argparse.ArgumentTypeError(f"{name}={value} is outside of bounds ({lb}, {ub})")
    return value


class FormatterNoDuplicate(argparse.ArgumentDefaultsHelpFormatter):
    """
    Formatter overriding `argparse.ArgumentDefaultsHelpFormatter` to show
    `-e, --epoch EPOCH` instead of `-e EPOCH, --epoch EPOCH`.

    Source:
    Adapted from CPython: https://github.com/python/cpython/blob/main/Lib/argparse.py
    """

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar

        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append(option_string)
            parts[-1] += f' {args_string}'

        return ', '.join(parts)
