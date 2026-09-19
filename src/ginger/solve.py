"""Facade entry points that auto-select the execution policy.

Mirrors ``include/ginger/solve.hpp`` in ginger-cpp: each solver exposes a
facade that dispatches to the single-threaded or multithreaded variant based
on an explicit :class:`SolveMode`.
"""

from enum import Enum
from typing import List, Sequence, Tuple

from .aberth import aberth, aberth_autocorr, aberth_autocorr_mt, aberth_mt
from .rootfinding import (
    Options,
    _bairstow_autocorr_step,
    _bairstow_even_step,
    _bairstow_solve,
)


class SolveMode(Enum):
    """Execution policy selector for the solver facades.

    ``AUTOMATIC`` selects the single-threaded variant. The per-root jobs are
    pure-Python arithmetic that never releases the GIL, so the thread pool is a
    net loss (measured 2-9x slower than sequential). ``MULTI_THREADED`` is kept
    for API parity with ginger-cpp and for callers that supply GIL-releasing
    work.
    """

    AUTOMATIC = "automatic"
    SEQUENTIAL = "sequential"
    MULTI_THREADED = "multi_threaded"


def solve_aberth(
    coeffs: Sequence[float],
    zs: List[complex],
    options: Options = Options(),
    mode: SolveMode = SolveMode.AUTOMATIC,
) -> Tuple[List[complex], int, bool]:
    """Aberth-Ehrlich solver with automatic policy selection.

    Args:
        coeffs: Polynomial coefficients in descending order.
        zs: Initial root guesses.
        options: Algorithm configuration.
        mode: Execution policy. ``AUTOMATIC`` and ``SEQUENTIAL`` both use the
            single-threaded variant; ``MULTI_THREADED`` opts into the thread
            pool. See :class:`SolveMode` for why ``AUTOMATIC`` is sequential.

    Returns:
        Tuple of (refined roots, iterations performed, converged).
    """
    if mode is SolveMode.MULTI_THREADED:
        return aberth_mt(coeffs, zs, options)
    return aberth(coeffs, zs, options)


def solve_aberth_autocorr(
    coeffs: Sequence[float],
    zs: List[complex],
    options: Options = Options(),
    mode: SolveMode = SolveMode.AUTOMATIC,
) -> Tuple[List[complex], int, bool]:
    """Aberth solver for autocorrelation polynomials with policy selection.

    Args:
        coeffs: Polynomial coefficients in descending order.
        zs: Initial root guesses.
        options: Algorithm configuration.
        mode: Execution policy. ``AUTOMATIC`` and ``SEQUENTIAL`` both use the
            single-threaded variant; ``MULTI_THREADED`` opts into the thread
            pool. See :class:`SolveMode` for why ``AUTOMATIC`` is sequential.

    Returns:
        Tuple of (refined roots, iterations performed, converged).
    """
    if mode is SolveMode.MULTI_THREADED:
        return aberth_autocorr_mt(coeffs, zs, options)
    return aberth_autocorr(coeffs, zs, options)


def solve_pbairstow_even(
    coeffs: List[float],
    vrs: List,
    options: Options = Options(),
    mode: SolveMode = SolveMode.AUTOMATIC,
) -> Tuple[List, int, bool]:
    """Parallel Bairstow solver (even degree) with policy selection.

    The Python package currently provides only the single-threaded
    Gauss-Seidel variant, so ``MULTI_THREADED`` falls back to it.

    Args:
        coeffs: Polynomial coefficients in descending order.
        vrs: Initial quadratic-factor estimates.
        options: Algorithm configuration.
        mode: Execution policy (only ``SEQUENTIAL``/``AUTOMATIC`` apply).

    Returns:
        Tuple of (final factor estimates, iterations performed, converged).
    """
    return _bairstow_solve(coeffs, vrs, options, _bairstow_even_step)


def solve_pbairstow_autocorr(
    coeffs: List[float],
    vrs: List,
    options: Options = Options(),
    mode: SolveMode = SolveMode.AUTOMATIC,
) -> Tuple[List, int, bool]:
    """Bairstow solver for autocorrelation polynomials with policy selection.

    The Python package currently provides only the single-threaded
    Gauss-Seidel variant, so ``MULTI_THREADED`` falls back to it.

    Args:
        coeffs: Polynomial coefficients in descending order.
        vrs: Initial quadratic-factor estimates.
        options: Algorithm configuration.
        mode: Execution policy (only ``SEQUENTIAL``/``AUTOMATIC`` apply).

    Returns:
        Tuple of (final factor estimates, iterations performed, converged).
    """
    return _bairstow_solve(coeffs, vrs, options, _bairstow_autocorr_step)
