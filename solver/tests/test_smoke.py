"""M0 smoke test: the package imports and exposes the global tolerances."""

import sfu_alloc
from sfu_alloc.constants import FEAS_TOL, TOL


def test_package_imports() -> None:
    assert sfu_alloc.__doc__


def test_tolerances() -> None:
    assert TOL == 1e-6
    assert FEAS_TOL == 1e-6
