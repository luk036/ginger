"""Facade entry points that auto-select the execution policy.

Mirrors ``include/ginger/solve.hpp`` in ginger-cpp: each solver exposes a
facade that dispatches to the single-threaded or multithreaded variant based
on the problem size (via :func:`ginger.rootfinding.should_parallelize`) or an
explicit :class:`SolveMode`.
"""

from enum import Enum
from typing import List, Sequence, Tuple

from .aberth import aberth, aberth_autocorr, aberth_autocorr_mt, aberth_mt
from .rootfinding import Options, _bairstow_solve, should_parallelize


class SolveMode(Enum):
    """Execution policy selector for the solver facades."""

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
        mode: Execution policy; ``AUTOMATIC`` dispatches to the
            multithreaded variant when :func:`should_parallelize` holds.

    Returns:
        Tuple of (refined roots, iterations performed, converged).
    """
    if mode is SolveMode.SEQUENTIAL:
        return aberth(coeffs, zs, options)
    if mode is SolveMode.MULTI_THREADED:
        return aberth_mt(coeffs, zs, options)
    return (
        aberth_mt(coeffs, zs, options) if should_parallelize(len(zs)) else aberth(coeffs, zs, options)
    )


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
        mode: Execution policy; ``AUTOMATIC`` dispatches to the
            multithreaded variant when :func:`should_parallelize` holds.

    Returns:
        Tuple of (refined roots, iterations performed, converged).
    """
    if mode is SolveMode.SEQUENTIAL:
        return aberth_autocorr(coeffs, zs, options)
    if mode is SolveMode.MULTI_THREADED:
        return aberth_autocorr_mt(coeffs, zs, options)
    return (
        aberth_autocorr_mt(coeffs, zs, options)
        if should_parallelize(len(zs))
        else aberth_autocorr(coeffs, zs, options)
    )


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
    return _bairstow_solve(coeffs, vrs, options, autocorr=False)


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
    return _bairstow_solve(coeffs, vrs, options, autocorr=True)
