=========
Changelog
=========

Version 0.4 (2026-07-16)
========================

Bug Fixes
---------
- **VDC harmonization**: Replaced evenly-spaced angles with VDC-based angles in ``initial_autocorr``, matching Rust/C++. Added missing self-reciprocal suppression in ``pbairstow_autocorr``. (#8a3daee)
- **suppress_old e^2 fix**: Changed division by ``e`` to multiplication (``a_val*e`` → ``a_val/e``) and fixed missing ``/e`` in vA1 computation. (#4fa4b24)
- **initial_guess table index**: Fixed COS_PI_VDC2_TABLE index offset to match Rust/C++. (#4fa4b24)

Performance
-----------
- **One zs snapshot per iteration**: Reduced allocation churn in ``aberth_mt`` and ``aberth_autocorr_mt`` from O(N²) to O(N) copies. (#7818e8f)

Testing & Code Quality
----------------------
- **Coverage raised 98%→99%**: Added non-convergence and large-radius coverage tests. (#7665d8b)

Code Cleanup
------------
- **Removed PyScaffold boilerplate**: Deleted ``skeleton.py`` and ``test_skeleton.py``, removed stale files (``aberth_ai.py``, ``leja_order_ai.py``, ``CONTRIBUTING.md.bak``). (#7e18fe4)
- **Dropped Python < 3.9 compat**: Removed ``importlib-metadata`` conditional dependency. (#7e18fe4)
