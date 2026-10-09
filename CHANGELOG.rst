=========
Changelog
=========

Version 0.5 (2026-10-09)
========================

Features
--------
- **Solve facade + execution policies**: Added ``src/ginger/solve.py``
  (``SolveMode`` + ``solve_*`` facades, ``PARALLEL_THRESHOLD`` /
  ``should_parallelize``) and ``src/ginger/_policy.py``
  (``sequential_run`` / ``jacobi_mt_run`` execution policies, Strategy).
  (#e126a57)

Performance
-----------
- **Scale-aware convergence, fused Horner, O(n^2) Leja order**:
  ``residual_scale()`` normalizes the polynomial so ``Options.tolerance`` is a
  *relative* test — the old absolute test was unreachable for large-coefficient
  polynomials (degree 16 burned all 2000 iterations while its roots were
  accurate to 4.7e-14). ``horner_eval_pp()`` evaluates P and P' in one pass;
  float cores ``_delta_step`` / ``_suppress_old_step`` / ``_horner_quadratic``
  replace ``Vector2`` / ``Matrix2`` churn in the Bairstow hot loop;
  ``leja_order`` is O(n^2) (was O(n^3), bit-identical output); ``AUTOMATIC`` no
  longer dispatches to the GIL-bound thread pool (measured 2-9x slower);
  round-robin neighbours are precomputed. (#f532f89)

Bug Fixes
---------
- **Unused ``suppress`` import**: Removed. (#0f2cb9c)

Testing & Code Quality
----------------------
- **Order-independence tests**: Added 4 tests verifying bit-identical Jacobi
  next state under any processing order, root-set invariance under permuted
  guesses, machine-epsilon suppression-order drift, and Gauss-Seidel
  iteration-count order-dependence. (#a6205ae)
- **Facade tests**: Added ``test_solve.py``. (#e126a57)
- **Scaling benchmark**: Added ``bench/test_scaling.py`` guarding niter /
  converged. (#f532f89)

Code Cleanup
------------
- **Refactor**: Split the Bairstow steps into even/autocorr variants (dropping
  the autocorr flag) and deduped the job functions. (#20d545d, #e126a57)
- **Removed AI slop**: Stripped boilerplate from docstrings and comments;
  formatted test files and a signature with black. (#dc6e27c, #95fd9df, #503d51c)
- **Config**: Moved flake8 config into a dedicated ``.flake8`` (fixing the stale
  PyScaffold project URLs) and stripped the empty ``entry_points`` section from
  ``setup.cfg``. (#a9970e1, #fd051a2)

Documentation
-------------
- **Order-independence analysis**: Added the corrected ``order-matter.md``
  report, documenting that bit-for-bit order-independence holds only for
  Jacobi-style iteration while the single-threaded variants are
  order-dependent. (#fc38e01)
- **AGENTS.md**: Added agent guidelines. (#f5b3ca2)

Build & CI
----------
- **Updated GitHub Actions**: checkout->v4, setup-python->v5,
  codecov-action->v4; removed the stale ``.bak`` workflow. (#c43302f, #c9d3b43)

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
