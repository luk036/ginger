from concurrent.futures import ThreadPoolExecutor
from math import cos, sin, pow, pi
from typing import List, Tuple

# Assuming Options class is defined elsewhere and imported with proper annotations
from .rootfinding import Options, horner_eval_f

TWO_PI: float = 2.0 * pi


# def horner_eval_f(coeffs: List[float], zval: float) -> float:
#     return sum(coeff * (zval**i) for i, coeff in enumerate(coeffs[::-1]))


def horner_eval_c(coeffs: List[float], zval: complex) -> complex:
    return sum(coeff * (zval**i) for i, coeff in enumerate(coeffs[::-1]))


def initial_aberth(coeffs: List[float]) -> List[complex]:
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (coeffs[0] * degree)
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    k: float = TWO_PI / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (k * (0.25 + i) for i in range(degree))
    ]


def aberth_job(
    coeffs: List[float],
    i: int,
    zi: complex,
    converged: List[bool],
    zsc: List[complex],
    coeffs1: List[float],
) -> float:
    pp = horner_eval_c(coeffs, zi)
    tol_i = abs(pp)
    if tol_i < 1e-15:
        converged[i] = True
        return 0.0
    pp1 = horner_eval_c(coeffs1, zi)
    for j, zj in enumerate(zsc):
        if i != j:
            pp1 -= pp / (zi - zj)
    zi -= pp / pp1
    return tol_i


def aberth_mt(
    coeffs: List[float], zs: List[complex], options: Options
) -> Tuple[int, bool]:
    degree = len(coeffs) - 1
    coeffs1 = [coeff * (degree - i) for i, coeff in enumerate(coeffs[:degree])]
    converged = [False] * len(zs)
    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            futures = []
            zsc = zs.copy()

            for i in range(len(zs)):
                if not converged[i]:
                    futures.append(
                        executor.submit(
                            aberth_job, coeffs, i, zs[i], converged, zsc, coeffs1
                        )
                    )

            for future in futures:
                tol_i = future.result()
                if tol_i > tolerance:
                    tolerance = tol_i

            if tolerance < options.tolerance:
                return niter, True

    return options.max_iters, False


# Example usage (assuming Options and necessary setup exist with proper annotations)
# options = Options(max_iters=1000, tolerance=1e-10)
# coeffs = [1, 0, -1]  # Coefficients for a simple polynomial x^2 - 1
# zs = initial_aberth(coeffs)
# iterations, converged = aberth_mt(coeffs, zs, options)
# print(f"Iterations: {iterations}, Converged: {converged}")
