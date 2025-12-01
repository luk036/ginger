# -*- coding: utf-8 -*-
from __future__ import print_function

from ginger.autocorr import initial_autocorr, pbairstow_autocorr
from ginger.rootfinding import initial_guess, pbairstow_even
from typing import Any


def run_autocorr() -> int:
    """Run autocorrelation benchmark function.

    Returns:
        int: number of iterations
    """
    coeffs = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_autocorr(coeffs)
    _, niter, _ = pbairstow_autocorr(coeffs, vr0s)
    return niter


def run_pbairstow() -> int:
    """Run pbairstow benchmark function.

    Returns:
        int: number of iterations
    """
    coeffs = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_guess(coeffs)
    _, niter, _ = pbairstow_even(coeffs, vr0s)
    return niter


def test_autocorr(benchmark: Any) -> None:
    """Benchmark autocorrelation function.

    Arguments:
        benchmark: pytest-benchmark fixture
    """
    result = benchmark(run_autocorr)
    assert result <= 12


def test_pbairstow(benchmark: Any) -> None:
    """Benchmark pbairstow function.

    Arguments:
        benchmark: pytest-benchmark fixture
    """
    result = benchmark(run_pbairstow)
    assert result <= 11
