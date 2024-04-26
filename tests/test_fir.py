from ginger.autocorr import extract_autocorr, initial_autocorr, pbairstow_autocorr
from ginger.rootfinding import Options, find_rootq, initial_guess, pbairstow_even

r = [
    -0.00196191,
    -0.00094597,
    -0.00023823,
    0.00134667,
    0.00380494,
    0.00681596,
    0.0097864,
    0.01186197,
    0.0121238,
    0.00985211,
    0.00474894,
    -0.00281751,
    -0.01173923,
    -0.0201885,
    -0.02590168,
    -0.02658216,
    -0.02035729,
    -0.00628271,
    0.01534627,
    0.04279982,
    0.0732094,
    0.10275561,
    0.12753013,
    0.14399228,
    0.15265722,
    0.14399228,
    0.12753013,
    0.10275561,
    0.0732094,
    0.04279982,
    0.01534627,
    -0.00628271,
    -0.02035729,
    -0.02658216,
    -0.02590168,
    -0.0201885,
    -0.01173923,
    -0.00281751,
    0.00474894,
    0.00985211,
    0.0121238,
    0.01186197,
    0.0097864,
    0.00681596,
    0.00380494,
    0.00134667,
    -0.00023823,
    -0.00094597,
    -0.00196191,
]


def test_fir_even():
    vr0s = initial_guess(r)
    opts = Options()
    opts.tolerance = 4e-8
    # opts.tol_suppress = 0.5e-1
    vrs, niter, found = pbairstow_even(r, vr0s, opts)
    print([niter, found])
    for vr in vrs:
        print(find_rootq(vr))
    assert niter <= 181


def test_fir_auto():
    vr0s = initial_autocorr(r)
    print("vrs: {}".format(len(vr0s)))
    opts = Options()
    opts.tolerance = 4e-8
    vrs, niter, found = pbairstow_autocorr(r, vr0s, opts)
    print([niter, found])
    for vr in vrs:
        vr = extract_autocorr(vr)
        print(find_rootq(vr))

    assert niter <= 15
