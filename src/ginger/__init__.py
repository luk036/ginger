"""ginger — Polynomial root-finding algorithms (parallelizable).

This package provides parallel implementations of Bairstow's method and
Aberth-Ehrlich's method for finding all roots of real-coefficient polynomials.
It is pure Python with no NumPy dependency.

Submodules:
    rootfinding     — parallel Bairstow method (pbairstow_even)
    aberth          — Aberth-Ehrlich method (single-threaded and MT)
    autocorr        — Bairstow solver for palindromic/autocorrelation polynomials
    vector2         — 2D vector class for quadratic factor coefficients
    matrix2         — 2x2 matrix class used in Bairstow correction
"""

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.9`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
