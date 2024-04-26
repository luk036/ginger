from math import acos, cos, sqrt
from typing import List, Tuple

from .lds import VdCorput
from .matrix2 import Matrix2
from .robin import Robin
from .vector2 import Vector2

PI = acos(-1.0)


class Options:
    max_iters: int = 2000
    tolerance: float = 1e-12
    tol_ind: float = 1e-15
    # tol_suppress: float = 1e-1


# def horner_eval(coeffs1: List[float], z):


def delta(vA: Vector2, vr: Vector2, vp: Vector2) -> Vector2:
    """[summary]

    r * p + s   -p
    -q * p      -s

    Args:
        vA (Vector2): [description]
        vr (Vector2): [description]
        vp (Vector2): [description]

    Returns:
        Vector2: [description]

    Examples:
        >>> d = delta(Vector2(1, 2), Vector2(-2, 0), Vector2(4, 5))
        >>> print(d)
        <0.2, -0.4>
    """
    r, q = vr.x, vr.y
    p, s = vp.x, vp.y
    mp = Matrix2(Vector2(-s, p), Vector2(p * q, p * r + s))
    return mp.mdot(vA) / mp.det()  # 6 mul's + 2 div's


def suppress_old(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2) -> Vector2:
    """[summary]

    Args:
        vA (Vector2): [description]
        vr (Vector2): [description]
        vp (Vector2): [description]

    Returns:
        Vector2: [description]

    Examples:
        >>> vA = Vector2(3, 3)
        >>> vA1 = Vector2(1, 2)
        >>> vri = Vector2(-2, 0)
        >>> vrj = Vector2(4, -5)
        >>> dr = suppress_old(vA, vA1, vri, vrj)
        >>> print(dr)
        <-16.78082191780822, 1.4383561643835616>
    """
    A, B = vA.x, vA.y
    A1, B1 = vA1.x, vA1.y
    vp = vri - vrj
    r, q = vri.x, vri.y
    p, s = vp.x, vp.y
    f = r * p - s
    qp = q * p
    e = -f * s + qp * p
    a = -A * s - B * p
    b = B * f + A * qp
    c = A1 * e - a
    d = B1 * e - b - a * p
    A = a * e
    B = b * e
    A1 = -c * s - d * p
    B1 = d * f + c * qp
    return delta(Vector2(A, B), vri, Vector2(A1, B1))


def suppress(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2) -> Vector2:
    """[summary]

    Args:
        vA (Vector2): [description]
        vr (Vector2): [description]
        vp (Vector2): [description]

    Returns:
        Vector2: [description]

    Examples:
        >>> vA = Vector2(3, 3)
        >>> vA1 = Vector2(1, 2)
        >>> vri = Vector2(-2, 0)
        >>> vrj = Vector2(4, -5)
        >>> dr = suppress(vA, vA1, vri, vrj)
        >>> print(dr)
        <-16.780821917808282, -1.4383561643835616>
    """
    vp = vri - vrj
    vA1 -= delta(vA, vrj, vp)
    return delta(vA, vri, vA1)


def horner_eval(coeffs: List, degree: int, val):
    """Polynomial evaluation using Horner's scheme

    Note: coeffs becomes the quotient after calling this function

    Args:
        coeffs (List): List of coefficients of polynomial
        val (float or complex): value to be evaluated

    Returns:
        float or complex

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> horner_eval(coeffs, 5, 3)
        960
        >>> coeffs
        [1, -5, -87, 121, 1090, 960]
    """
    for i in range(degree):
        coeffs[i + 1] += coeffs[i] * val
    return coeffs[degree]


def horner_backward(coeffs: List, degree: int, val):
    """Polynomial evaluation using Horner's scheme

    Note: coeffs becomes the quotient after calling this function

    Args:
        coeffs (List): List of coefficients of polynomial
        val (float or complex): value to be evaluated

    Returns:
        float or complex

    Examples:
        >>> coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
        >>> degree = len(coeffs) - 1
        >>> alpha = 6.3256
        >>> p_eval = horner_backward(coeffs, 4, alpha)
        >>> -p_eval * (alpha ** 5)
        -0.013355264987140483
        >>> coeffs[3]
        0.006920331351966613
    """
    for i in range(2, degree + 2):
        coeffs[-i] -= coeffs[-(i - 1)]
        coeffs[-i] /= -val
    return coeffs[-(degree + 1)]


def horner(coeffs: List[float], degree: int, vr: Vector2) -> Vector2:
    """[summary]

    Note: coeffs1 becomes the quotient after calling this function

    Args:
        coeffs (List[float]): [description]
        vr (Vector2): [description]

    Returns:
        Vector2: [description]

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> vp = horner(coeffs, 5, Vector2(-1, -6))  # x^2 + x - 6
        >>> coeffs
        [1, -9, -57, 385, 0, 0]
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> vp = horner(coeffs, 5, Vector2(2, -3))  # x^2 - 2x - 3
        >>> coeffs
        [1, -6, -81, 202, 888, -1704]
    """
    coeffs[1] += coeffs[0] * vr.x
    for i in range(2, degree):
        coeffs[i] += coeffs[i - 1] * vr.x - coeffs[i - 2] * vr.y
    coeffs[degree] -= coeffs[degree - 2] * vr.y
    return Vector2(coeffs[degree - 1], coeffs[degree])


def initial_guess_orig(coeffs: List[float]) -> List[Vector2]:
    """[summary]

    Args:
        coeffs (List[float]): [description]

    Returns:
        List[Vector2]: [description]

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
    """
    degree = len(coeffs) - 1
    center = -coeffs[1] / (degree * coeffs[0])
    # p_eval = np.poly1d(coeffs)
    p_center = horner_eval(coeffs.copy(), degree, center)
    radius = abs(p_center) ** (1.0 / degree)
    m = center * center + radius * radius
    vr0s = []
    degree //= 2
    degree *= 2  # make even
    k = PI / degree
    for i in range(1, degree, 2):
        temp = radius * cos(k * i)
        r0 = 2 * (center + temp)
        t0 = m + 2 * center * temp  # ???
        vr0s += [Vector2(r0, t0)]
    return vr0s


def initial_guess(coeffs: List[float]) -> List[Vector2]:
    """[summary]

    Args:
        coeffs (List[float]): [description]

    Returns:
        List[Vector2]: [description]

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
    """
    degree = len(coeffs) - 1
    center = -coeffs[1] / (degree * coeffs[0])
    # p_eval = np.poly1d(coeffs)
    p_center = horner_eval(coeffs.copy(), degree, center)
    radius = abs(p_center) ** (1.0 / degree)
    m = center * center + radius * radius
    vr0s = []
    degree //= 2
    degree *= 2  # make even
    # k = PI / degree
    vgen = VdCorput(2)
    vgen.reseed(1)
    for i in range(1, degree, 2):
        temp = radius * cos(PI * vgen.pop())
        r0 = 2 * (center + temp)
        t0 = m + 2 * center * temp  # ???
        vr0s += [Vector2(r0, t0)]
    return vr0s


def pbairstow_even(
    coeffs: List[float], vrs: List[Vector2], options=Options()
) -> Tuple[List[Vector2], int, bool]:
    """[summary]

    Args:
        coeffs (List[float]): [description]
        vrs (List[Vector2]): [description]
        options (Options, optional): [description]. Defaults to Options().

    Returns:
        [type]: [description]

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
        >>> vrs, niter, found = pbairstow_even(h, vr0s)
    """
    M = len(vrs)
    degree = len(coeffs) - 1
    converged = [False] * M
    robin = Robin(M)
    for niter in range(1, options.max_iters):
        tolerance = 0.0
        # exclude converged
        for i in filter(lambda i: converged[i] is False, range(M)):
            # for i in range(M):
            coeffs1 = coeffs.copy()
            vA = horner(coeffs1, degree, vrs[i])
            tol_i = max(abs(vA.x), abs(vA.y))
            if tol_i < options.tol_ind:
                converged[i] = True
                continue
            vA1 = horner(coeffs1, degree - 2, vrs[i])
            tolerance = max(tol_i, tolerance)
            # for j in filter(lambda j: j != i, range(M)):  # exclude i
            for j in robin.exclude(i):
                vA1 -= delta(vA, vrs[j], vrs[i] - vrs[j])
            vrs[i] -= delta(vA, vrs[i], vA1)
        if tolerance < options.tolerance:
            return vrs, niter, True
    return vrs, options.max_iters, False


def find_rootq(vr: Vector2) -> Tuple[float, float]:
    """Solve x^2 - r*x + q = 0

    (x - x1)(x - x2) = x^2 - (x1 + x2) x + x1 * x2

    Args:
        vr (Vector2): [description]

    Returns:
        Tuple[float, float]: [description]

    Examples:
        >>> vr = find_rootq(Vector2(5, 6))
        >>> print(vr)
        (3.0, 2.0)
    """
    # r, q = vr.x, vr.y
    hr = vr.x / 2.0
    d = hr * hr - vr.y
    if d < 0.0:
        x1 = hr + sqrt(-d) * 1j
    else:
        x1 = hr + (sqrt(d) if hr >= 0.0 else -sqrt(d))
    x2 = vr.y / x1
    return x1, x2
