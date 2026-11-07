"""Copy-on-write (the pandas 3 default): no code path may write into a ``.values`` view.

Under copy-on-write ``DataFrame.values`` is a read-only view, so in-place writes such as
``np.fill_diagonal(df.values, ...)`` raise. pandas 2.x can switch copy-on-write on as an option, which lets
the locked-pandas CI rows guard the paths that pandas 3 broke: the Strube correlation in
``combine_target_tests`` and the residual scan without eigengene residualisation.
"""
import contextlib

import pandas as pd
import pytest

from tests import test_isoform_residuals, test_targets


def _copy_on_write():
    if int(pd.__version__.split(".")[0]) >= 3:   # always on, and the option is gone
        return contextlib.nullcontext()
    return pd.option_context("mode.copy_on_write", True)


def test_copy_on_write_makes_values_read_only():
    with _copy_on_write():
        assert not pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).values.flags.writeable


@pytest.mark.filterwarnings("ignore")
def test_combine_target_tests_under_copy_on_write(tmp_path):
    with _copy_on_write():
        test_targets.test_combine_target_tests_pools_weak_evidence(tmp_path)
        test_targets.test_combine_strube_correction_reduces_evidence_for_dependent_cohorts()


@pytest.mark.filterwarnings("ignore")
def test_isoform_scan_without_residualised_eigengenes_under_copy_on_write():
    with _copy_on_write():
        test_isoform_residuals.test_residualised_eigengenes_suppress_leakage_from_correlated_modules()
