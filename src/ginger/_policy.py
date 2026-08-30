"""Execution policies for parallelizable root-finding solvers.

Strategy + Template-Method decomposition:

- Each algorithm supplies a per-root *job* function ``job(i, state)`` that
  computes one Newton correction and returns ``(tol, i, new_value)``.
- Each execution policy owns the iteration loop, the scheduling of the jobs,
  and the convergence aggregation.

This mirrors ``source/execution_policy.hpp`` in ginger-cpp: the algorithm
(Aberth/Bairstow variant) is decoupled from the execution mode
(sequential Gauss-Seidel vs. multithreaded Jacobi).
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from .rootfinding import Options

Job = Tuple[float, int, object]


def sequential_run(
    job: "object", state: List, options: Options
) -> Tuple[List, int, bool]:
    """Sequential Gauss-Seidel execution: roots are updated in-place in
    ascending index order within each iteration.

    Args:
        job: Per-root step ``job(i, state) -> (tol, i, new_value)``.
        state: Mutable list of current root estimates.
        options: Algorithm configuration.

    Returns:
        Tuple of (final roots, iterations performed, converged).
    """
    for niter in range(options.max_iters):
        tolerance = 0.0
        for i in range(len(state)):
            tol_i, i, new_value = job(i, state)  # type: ignore[operator]
            state[i] = new_value
            tolerance = max(tolerance, tol_i)
        if tolerance < options.tolerance:
            return state, niter, True
    return state, options.max_iters, False


def jacobi_mt_run(
    job: "object", state: List, options: Options
) -> Tuple[List, int, bool]:
    """Jacobi multithreaded execution: each iteration reads a frozen snapshot
    of the roots and submits one job per root to a thread pool; results are
    written back into the live state.

    Args:
        job: Per-root step ``job(i, snapshot) -> (tol, i, new_value)``.
        state: Mutable list of current root estimates.
        options: Algorithm configuration.

    Returns:
        Tuple of (final roots, iterations performed, converged).
    """
    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            snapshot = state[:]
            futures = [
                executor.submit(job, i, snapshot) for i in range(len(state))  # type: ignore[operator]
            ]
            for future in futures:
                tol_i, i, new_value = future.result()
                state[i] = new_value
                tolerance = max(tolerance, tol_i)
            if tolerance < options.tolerance:
                return state, niter, True
    return state, options.max_iters, False
