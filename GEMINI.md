# Gemini Code Guide: ginger

## Project Overview

`ginger` is a pure Python library for finding the roots of polynomials. It provides implementations of the Aberth-Ehrlich method and a parallelized version of Bairstow's method. The library is designed to be lightweight and has no dependency on `numpy`.

The main features of `ginger` are:

*   **Pure Python:** No external libraries like `numpy` are required.
*   **Low Memory Footprint:** The storage requirement is O(N), where N is the degree of the polynomial.
*   **Structure Preservation:** Preserves structures like auto-correlation functions.

The core logic is located in `src/ginger/rootfinding.py`, which contains the implementations of the root-finding algorithms.

## Building and Running

This project uses `tox` for managing testing, building, and other development tasks.

### Dependencies

The project's dependencies are managed in `setup.cfg`. The testing dependencies are:

*   `pytest`
*   `pytest-cov`
*   `pytest-benchmark`

### Testing

To run the tests, execute the following command:

```bash
tox
```

This will run the test suite using `pytest` and generate a coverage report.

### Building

To build the project, use the following `tox` environments:

*   **Clean:** To remove previous build artifacts:
    ```bash
    tox -e clean
    ```
*   **Build:** To create a new build:
    ```bash
c    tox -e build
    ```

### Publishing

To publish the package to PyPI, use the `publish` environment in `tox`:

```bash
tox -e publish
```

## Development Conventions

*   **Code Style:** The project uses `flake8` for linting and `pre-commit` for running static analysis and format checkers. The configuration for `flake8` can be found in `.flake8`.
*   **Typing:** The codebase is type-hinted.
*   **Documentation:** The project uses Sphinx for documentation. The documentation source is in the `docs/` directory. To build the documentation, run:
    ```bash
    tox -e docs
    ```
*   **Continuous Integration:** The project uses GitHub Actions for CI. The workflow is defined in `.github/workflows/ci.bak`. The CI pipeline runs tests on Python 3.9 and 3.11 across Ubuntu, macOS, and Windows.
*   **Coverage:** The project uses `coveralls` to track code coverage.
