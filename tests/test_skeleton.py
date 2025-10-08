from unittest.mock import patch

import pytest

from ginger.skeleton import fib, main, run

__author__ = "Wai-Shing Luk"
__copyright__ = "Wai-Shing Luk"
__license__ = "MIT"


def test_fib():
    """API Tests"""
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out


def test_run(capsys):
    """CLI Tests"""
    with patch("sys.argv", ["fibonacci", "7"]):
        run()
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out
