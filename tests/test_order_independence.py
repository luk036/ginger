"""Order-independence tests mirroring the Rust ginger-rs suite.

Covers:
1. Jacobi (frozen-snapshot) sweep: order-independent, bit-identical next state
   regardless of factor processing order.
2. Suppression order within a job: only machine-epsilon drift.
3. Gauss-Seidel (pbairstow_even): order-DEPENDENT iteration count when
   initial guesses are permuted.

Note: this Python package has no `pbairstow_even_mt`; the Bairstow entry
points are Gauss-Seidel. The Jacobi sweep is therefore reconstructed here
from the public primitives (`horner`, `suppress_old`, `delta`), mirroring
the Rust `pbairstow_even_mt` frozen-snapshot loop exactly.
"""

from ginger.rootfinding import (
    Options,
    delta,
    horner,
    initial_guess,
    pbairstow_even,
    roots_from_quadratic,
    suppress_old,
)
from ginger.vector2 import Vector2

H8 = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]


def deepcopy_vrs(vrs: list[Vector2]) -> list[Vector2]:
    """Deep-copy a factor list (shallow copies alias the mutable Vector2s)."""
    return [Vector2(v.x, v.y) for v in vrs]


def sorted_roots(vrs: list[Vector2]) -> list[tuple[float, float]]:
    """All roots of the converged factors, sorted for set comparison."""
    roots: list[tuple[float, float]] = []
    for vr in vrs:
        for z in roots_from_quadratic(vr):
            roots.append((z.real, z.imag))
    roots.sort()
    return roots


def max_root_set_diff(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Maximum coordinate-wise difference between two sorted root sets."""
    return max(
        (
            max(abs(x[0] - y[0]), abs(x[1] - y[1]))
            for x, y in zip(a, b)
        ),
        default=0.0,
    )


def jacobi_job(
    coeffs: list[float], i: int, vri: Vector2, vrsc: list[Vector2]
) -> tuple[Vector2, float] | None:
    """Mirror of Rust pbairstow_even_job: reads the frozen snapshot only."""
    coeffs1 = coeffs.copy()
    degree = len(coeffs1) - 1
    vA = horner(coeffs1, degree, vri)
    tol_i = max(abs(vA.x), abs(vA.y))
    if tol_i < 1e-15:
        return None
    vA1 = horner(coeffs1, degree - 2, vri)
    for j, vrj in enumerate(vrsc):
        if j != i:
            suppress_old(vA, vA1, vri, vrj)
    dt = delta(vA, vri, vA1)
    return vri - dt, tol_i


def jacobi_sweep(
    coeffs: list[float], vrs: list[Vector2], processing_order: list[int]
) -> tuple[list[Vector2], float]:
    """One Jacobi sweep: frozen snapshot, jobs read it, write own slot.

    Returns (next state, max tolerance) — mirrors the _mt variant which
    checks ``tolerance < options.tolerance`` for convergence.
    """
    vrsc = vrs[:]  # frozen snapshot, as in the Rust _mt variant
    next_vrs = vrs[:]
    tolerance = 0.0
    for i in processing_order:
        res = jacobi_job(coeffs, i, vrs[i], vrsc)
        if res is not None:
            next_vrs[i], tol_i = res
            tolerance = max(tolerance, tol_i)
    return next_vrs, tolerance


def test_jacobi_sweep_order_independent_bit_identical() -> None:
    """Jacobi sweep: every order yields a BIT-IDENTICAL next state."""
    vrs0 = initial_guess(H8)
    orders = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]]

    next_states = [
        jacobi_sweep(H8, vrs0, order)[0] for order in orders
    ]

    ref = next_states[0]
    for k, state in enumerate(next_states[1:], start=1):
        for a, b in zip(ref, state):
            assert a.x == b.x
            assert a.y == b.y


def test_jacobi_mt_order_independent_full_convergence() -> None:
    """Jacobi full convergence: permuted initial guesses give same root SET."""
    base = initial_guess(H8)
    perms = [
        deepcopy_vrs(base),
        deepcopy_vrs(base[::-1]),
        deepcopy_vrs([base[1], base[3], base[0], base[2]]),
        deepcopy_vrs([base[2], base[0], base[3], base[1]]),
    ]

    rootsets: list[list[tuple[float, float]]] = []
    for perm in perms:
        vrs = perm
        converged = False
        for _ in range(2000):
            next_vrs, tolerance = jacobi_sweep(H8, vrs, list(range(len(vrs))))
            if tolerance < 1e-12:
                converged = True
                break
            vrs = next_vrs
        assert converged
        rootsets.append(sorted_roots(vrs))

    for k, rs in enumerate(rootsets[1:], start=1):
        assert max_root_set_diff(rootsets[0], rs) < 1e-9


def test_suppression_order_machine_epsilon() -> None:
    """Suppression order within a job: drift stays at machine-epsilon level."""
    vrs = initial_guess(H8)
    orders = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]]

    worst = 0.0
    for i in range(len(vrs)):
        ref_vri: Vector2 | None = None
        for order in orders:
            coeffs1 = H8.copy()
            degree = len(coeffs1) - 1
            vri = vrs[i]
            vA = horner(coeffs1, degree, vri)
            if max(abs(vA.x), abs(vA.y)) < 1e-15:
                continue
            vA1 = horner(coeffs1, degree - 2, vri)
            for j in order:
                if j != i:
                    suppress_old(vA, vA1, vri, vrs[j])
            dt = delta(vA, vri, vA1)
            new_vri = vri - dt
            if ref_vri is None:
                ref_vri = new_vri
            else:
                worst = max(
                    worst,
                    abs(new_vri.x - ref_vri.x),
                    abs(new_vri.y - ref_vri.y),
                )
    assert worst < 1e-12


def test_gs_order_dependent_iterations() -> None:
    """Gauss-Seidel (pbairstow_even): niter varies with guess order."""
    base = initial_guess(H8)
    perms = [
        deepcopy_vrs(base),
        deepcopy_vrs(base[::-1]),
        deepcopy_vrs([base[1], base[3], base[0], base[2]]),
        deepcopy_vrs([base[2], base[0], base[3], base[1]]),
    ]

    niters: list[int] = []
    for perm in perms:
        _, niter, found = pbairstow_even(H8, perm, Options())
        assert found
        niters.append(niter)

    assert not all(n == niters[0] for n in niters)
