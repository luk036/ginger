from pytest import approx

from ginger.aberth import horner_backward, horner_eval


def test_backward2() -> None:
    coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
    degree = len(coeffs) - 1
    alpha = 6.3256
    p_eval = horner_backward(coeffs, degree, alpha)
    assert -p_eval * (alpha**5) == approx(-0.013355264987140483)
    assert coeffs[3] == approx(0.006920331351966613)


def test_backward1() -> None:
    coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
    # degree = len(coeffs) - 1
    p_eval, coeffs1 = horner_eval(coeffs, 6.3256)
    assert p_eval == approx(-0.012701469838522064)
    assert coeffs1[3] == approx(-0.0020220560640132265)
