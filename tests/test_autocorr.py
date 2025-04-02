from ginger.autocorr import extract_autocorr, initial_autocorr, pbairstow_autocorr
from ginger.rootfinding import Options, find_rootq
from ginger.vector2 import Vector2


def test_extract_autocorr():
    vr = extract_autocorr(Vector2(1, -4))
    assert vr.x == 0.25
    assert vr.y == -0.25


def test_autocorr():
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_autocorr(h)
    opts = Options()
    # opts.tolerance = 1e-30
    vrs, niter, found = pbairstow_autocorr(h, vr0s, opts)
    print([niter, found])

    for vr in vrs:
        vr = extract_autocorr(vr)
        print(find_rootq(vr))

    assert niter <= 12
