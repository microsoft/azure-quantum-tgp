# Copyright (c) Microsoft Corporation. All rights reserved.

import nox

_ARGS = ("-vvv", "--nbmake", "--overwrite")


_NOTEBOOKS = [
    "notebooks/paper-figures.ipynb",
    "notebooks/stage-one-analysis.ipynb",
    "notebooks/stage-two-analysis.ipynb",
    "notebooks/fridge-calibration.ipynb",
    "notebooks/PRL130_207001.ipynb",
]


@nox.session(python=["3.8", "3.9", "3.10", "3.11"], venv_backend="uv")
def pytest(session):
    session.install(".[test]")
    session.run("coverage", "erase")
    session.run("pytest", *_ARGS, *_NOTEBOOKS)


@nox.session(python="3.10")
def pytest_typeguard(session):
    session.install(".[test]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=tgp", *_ARGS, *_NOTEBOOKS)


@nox.session(python="3.8")
def coverage(session):
    session.install("coverage")
    session.install(".[test]")
    session.run("pytest", *_ARGS, *_NOTEBOOKS)
    session.run("coverage", "report")
    session.run("coverage", "xml")
