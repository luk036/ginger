# AGENTS.md - Agent Guidelines for ginger

## Build, Lint, and Test Commands

### Testing
```bash
# Run all tests (coverage + verbose enabled by default in setup.cfg)
pytest

# Run a single test file
pytest tests/test_aberth.py

# Run a single test function
pytest tests/test_aberth.py::test_aberth

# Run tests matching a pattern
pytest -k "autocorr"

# With explicit coverage
pytest --cov ginger --cov-report term-missing
```

### Linting and Formatting
```bash
# Run all pre-commit hooks (recommended before committing)
pre-commit run --all-files

# Individual tools
black src/ginger tests/    # Format code
isort src/ginger tests/    # Sort imports (black profile)
flake8 src/ tests/         # Lint (max line length: 256)
mypy src/                  # Type check (Python 3.12 target)
```

### Build
```bash
tox -e build          # Build sdist + wheel
tox -e clean          # Remove build artifacts
```

### Documentation
```bash
tox -e docs           # Build HTML docs
tox -e doctests       # Run doctests
tox -e linkcheck      # Check broken links
```

## Code Style Guidelines

### Imports
- **Order**: stdlib → third-party → local (enforced by isort)
- **Tooling**: isort with Black profile (`.isort.cfg`, `known_first_party = ginger`)
- **Third-party deps**: `lds_gen` (VdCorput LDS), `mywheel` (Robin)
- **Local imports**: Relative (`from .matrix2 import Matrix2`) or intra-package (`from .aberth import COS_PI_VDC2_TABLE`)

### Formatting
- **Formatter**: Black (23.7.0)
- **Line length**: 256 characters (configured in setup.cfg)
- **Linting**: flake8 ignores E203, W503; `.flake8` additionally ignores E1/E2/E3/E501/W1/W2/W3/W5 (formatting rules off)
- **Pre-commit**: Enforced via `.pre-commit-config.yaml`

### Type Hints
- **Required**: All function parameters and return types
- **Union aliases**: Define module-level aliases (e.g., `Num = Union[float, complex]`)
- **Algorithm config**: `Options` class with annotated class attributes
```python
class Options:
    max_iters: int = 2000
    tolerance: float = 1e-12
    tol_ind: float = 1e-15
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `Options`, `Vector2`, `Matrix2`)
- **Functions**: snake_case (e.g., `pbairstow_even`, `initial_guess`, `aberth_mt`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `COS_PI_VDC2_TABLE`, `TABLE_SIZE`)
- **Private helpers**: Prefix with underscore (e.g., `_vgen`)

### Docstrings
- **Style**: Sphinx/reStructuredText with `:param:`, `:type:`, `:return:`
- **Math**: Use `.. math::` (LaTeX) for formulas and `.. svgbob::` for diagrams
- **Examples**: Include `>>>` doctest examples in every public function
```python
def delta(vA: Vector2, vr: Vector2, vp: Vector2) -> Vector2:
    r"""Calculate adjustment vector for Bairstow's method.

    Solves the 2×2 linear system to find the optimal adjustment to current
    quadratic factor estimates :math:`(r,q)`:

    .. math::

       \begin{bmatrix} r p + s & p \\ q p & s \end{bmatrix}
       \begin{bmatrix} \Delta r \\ \Delta q \end{bmatrix}
       = \begin{bmatrix} A \\ B \end{bmatrix}

    :param vA: Residual vector :math:`(A,B)` from polynomial division
    :return: Correction vector :math:`(\Delta r, \Delta q)`

    Examples:
        >>> d = delta(Vector2(1, 2), Vector2(-2, 0), Vector2(4, 5))
        >>> print(d)
        <0.2, 0.4>
    """
```

### Error Handling
- **Convergence**: Return `(roots, iterations, converged_bool)` tuples rather than raising
- **No exceptions** for algorithmic non-convergence — report via the `converged` flag
- **Preconditions**: Use `assert` where appropriate (rare in this codebase)

### Testing Patterns
- **Framework**: pytest (coverage and verbose on by default in setup.cfg)
- **Naming**: `test_*` prefix, one file per module (`test_rootfind.py` → `rootfinding.py`)
- **Round-trip**: Reconstruct polynomials from found roots and verify coefficients
- **Coverage**: `.coveragerc` with branch coverage, `source = ginger`

### Pre-commit Hooks
- trailing-whitespace, check-added-large-files, check-ast, check-json,
  check-merge-conflict, check-xml, check-yaml, debug-statements,
  end-of-file-fixer, requirements-txt-fixer, mixed-line-ending,
  isort (5.12.0), black (23.7.0), flake8 (6.1.0)

### Configuration Files
- `setup.cfg`: Package metadata, pytest options, flake8 settings
- `pyproject.toml`: Build system (setuptools_scm)
- `tox.ini`: Test environments (default, build, clean, docs, doctests, linkcheck, publish)
- `.isort.cfg`: Import sorting (Black profile, `known_first_party = ginger`)
- `mypy.ini`: Type checking (Python 3.12, ignores `lds_gen` imports)
- `.coveragerc`: Branch coverage config

## Key Project Context

ginger is a parallel polynomial root-finding library:
- **Bairstow's method**: `rootfinding.py` — `pbairstow_even` factors a polynomial into quadratic factors with zero-suppression, plus `initial_guess`, `horner`, `suppress`, `roots_from_quadratic`
- **Aberth-Ehrlich method**: `aberth.py` — `aberth`, `aberth_mt` (ThreadPoolExecutor), `aberth_autocorr`/`aberth_autocorr_mt` for palindromic polynomials, `initial_aberth`, `poly_from_roots` (Leja ordering)
- **Geometry helpers**: `vector2.py` / `matrix2.py` for quadratic-factor coefficients
- **Key dependencies**: `lds_gen` (van der Corput initial guesses), `mywheel` (Robin indexing); pure Python, no numpy
- **Used by**: [multiplierless](https://github.com/luk036/multiplierless) for spectral factorization
