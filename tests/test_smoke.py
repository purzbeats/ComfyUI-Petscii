"""Scaffold check: the core package imports without ComfyUI or torch present."""

import petscii_core


def test_package_imports():
    assert petscii_core.__version__
