"""Shared pytest set-up for the whole repository.

On macOS, PyTorch and the tree libraries (LightGBM, XGBoost) each load their own copy of the
OpenMP runtime, and once a process has imported PyTorch, LightGBM crashes or hangs in it. So on
macOS, when a run would mix the two, the PyTorch tests (all of them live in src/urbanwind/dl/tests)
are taken out of the main run and run afterwards in a second pytest process with the same options.
On Linux, including CI, everything runs in one process as usual.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

DL_TESTS = pathlib.Path(__file__).resolve().parent / "src" / "urbanwind" / "dl" / "tests"
CHILD = "URBANWIND_DL_TESTS_ONLY"
DEFERRED = pytest.StashKey[list]()  # targets for the second process; set only when the run is split
OPTIONS = pytest.StashKey[list]()


def _in_dl_tests(path: pathlib.Path) -> bool:
    path = path.resolve()
    return path == DL_TESTS or DL_TESTS in path.parents


def pytest_configure(config):
    if sys.platform != "darwin" or os.environ.get(CHILD):
        return
    root = config.invocation_params.dir
    dl = [arg for arg in config.args if _in_dl_tests(root / arg.split("::")[0])]
    if len(dl) == len(config.args):  # only PyTorch tests were asked for, so nothing to keep apart
        return
    # pytest never asks pytest_ignore_collect about paths given on the command line, so explicit
    # PyTorch paths are taken out of the arguments here
    given = set(config.args) if config.args_source == pytest.Config.ArgsSource.ARGS else set()
    config.args[:] = [arg for arg in config.args if arg not in dl]
    config.stash[DEFERRED] = dl
    config.stash[OPTIONS] = [arg for arg in config.invocation_params.args if arg not in given]


def pytest_ignore_collect(collection_path, config):
    # a folder above the PyTorch tests (for example "src") reaches them during collection
    deferred = config.stash.get(DEFERRED, None)
    if deferred is None or not _in_dl_tests(collection_path):
        return None
    if str(DL_TESTS) not in deferred:
        deferred.append(str(DL_TESTS))
    return True


def pytest_sessionfinish(session):
    config = session.config
    deferred = config.stash.get(DEFERRED, None)
    if not deferred or session.exitstatus == pytest.ExitCode.INTERRUPTED:
        return
    print("\n\nPyTorch tests, in their own process (see conftest.py):", flush=True)
    argv = [sys.executable, "-m", "pytest", *config.stash[OPTIONS], *deferred]
    child = subprocess.run(argv, cwd=config.invocation_params.dir, env={**os.environ, CHILD: "1"}).returncode
    # "no tests ran" (5) on one side is no result; any failure on either side fails the run
    if session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED or (session.exitstatus == 0 and child != 5):
        session.exitstatus = child
