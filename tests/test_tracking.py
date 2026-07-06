"""Fast, offline tests for the experiment-tracking abstraction.

The MLflow tests use a temporary local SQLite backend, so they run fully
offline with no network calls -- the same private setup you'd use on-prem.
"""

import pytest

from src.tracking import (
    MLflowTracker,
    NoOpTracker,
    WandbTracker,
    build_tracker,
)


def test_build_tracker_returns_noop_by_default():
    assert isinstance(build_tracker({}), NoOpTracker)
    assert isinstance(build_tracker({"tracker": {"backend": "none"}}), NoOpTracker)


def test_build_tracker_selects_wandb():
    tracker = build_tracker({"tracker": {"backend": "wandb", "project": "unit-test"}})
    assert isinstance(tracker, WandbTracker)


def test_build_tracker_selects_mlflow(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = build_tracker(
        {"tracker": {"backend": "mlflow", "project": "unit-test", "tracking_uri": uri}}
    )
    assert isinstance(tracker, MLflowTracker)


def test_build_tracker_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_tracker({"tracker": {"backend": "does-not-exist"}})


def test_mlflow_tracker_logs_locally(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = build_tracker(
        {"tracker": {"backend": "mlflow", "project": "unit-test", "tracking_uri": uri}}
    )
    tracker.start_run({"lr": 0.001, "epochs": 1, "note": "hello"})
    tracker.log_metrics({"accuracy": 0.5}, step=1)
    tracker.finish()
