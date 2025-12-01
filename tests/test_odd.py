from ginger.rootfinding import find_rootq, initial_guess, pbairstow_even

# def test_odd() -> None:
#     h = [5.0, 2.5, 9.2, 6.9, 2.6, 0.2]
#     vr0s = initial_guess(h)
#     vrs, niter, found = pbairstow_even(h, vr0s)
#     print([niter, found])
#     print([find_rootq(vr) for vr in vrs])
#     assert niter <= 19


def test_odd2() -> None:
    h = [5.0, 2.5, 9.2, 6.9, 2.6, 0.2, 0]
    vr0s = initial_guess(h)
    vrs, niter, found = pbairstow_even(h, vr0s)
    print([niter, found])
    print([find_rootq(vr) for vr in vrs])
    assert niter <= 113
