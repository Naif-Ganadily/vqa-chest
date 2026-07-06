"""Experiment-tracking abstraction.

The pipelines log params / metrics / figures / models through a small, uniform
interface instead of hard-coding a vendor. You switch backends via the
``tracker`` block in the YAML config -- no pipeline code changes required.

Backends
--------
- ``wandb``  : Weights & Biases. Cloud SaaS by default (data leaves your machine).
- ``mlflow`` : MLflow. Self-hostable and fully private -- with an empty
               ``tracking_uri`` it logs to a local, offline SQLite backend
               (``sqlite:///mlflow.db``) with zero network calls (ideal for
               regulated / on-prem environments).
- ``none``   : No-op. Nothing is logged (handy for tests and quick local runs).

Example config block::

    tracker:
      backend: mlflow        # wandb | mlflow | none
      project: vqa-chest     # W&B project / MLflow experiment name
      run_name: local-run
      tracking_uri: ""       # mlflow only; "" -> local private SQLite (mlflow.db)

NOTE (healthcare): regardless of backend, never log PHI into params, metric
names, or artifact filenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimal experiment-tracking interface shared by every backend."""

    def start_run(self, config: dict) -> None:
        """Begin a run and record the config as params."""
        ...

    def log_params(self, params: dict) -> None:
        """Log hyperparameters / static config values."""
        ...

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log scalar metrics, optionally at a given step (e.g. epoch)."""
        ...

    def log_figure(self, name: str, fig: Any) -> None:
        """Log a matplotlib figure as an image artifact."""
        ...

    def log_model(self, path: str, name: str, metadata: Optional[dict] = None) -> None:
        """Log a saved model file as a versioned artifact."""
        ...

    def finish(self) -> None:
        """Close the run and flush anything buffered."""
        ...


class WandbTracker:
    """Weights & Biases backend (cloud SaaS by default)."""

    def __init__(self, project: str, run_name: Optional[str] = None):
        import wandb  # imported lazily so the dep is only needed when used

        self._wandb = wandb
        self._project = project
        self._run_name = run_name
        self._run = None

    def start_run(self, config: dict) -> None:
        self._run = self._wandb.init(project=self._project, name=self._run_name, config=config)

    def log_params(self, params: dict) -> None:
        # W&B captures config at init(); update() keeps it in sync for late params.
        self._wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_figure(self, name: str, fig: Any) -> None:
        self._wandb.log({name: self._wandb.Image(fig)})

    def log_model(self, path: str, name: str, metadata: Optional[dict] = None) -> None:
        artifact = self._wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_file(path)
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()


class MLflowTracker:
    """MLflow backend -- self-hostable and private.

    ``tracking_uri``:
      - ``""`` / ``None`` -> local private SQLite (``sqlite:///mlflow.db``), offline
      - ``"sqlite:///my.db"`` / ``"postgresql://..."`` -> your own DB backend
      - ``"http://host:5000"`` -> a self-hosted MLflow tracking server
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        import mlflow  # imported lazily

        self._mlflow = mlflow
        # MLflow's plain file store ("./mlruns") is in maintenance mode as of
        # MLflow 3.x, so default to a local, private SQLite backend. Metadata
        # (SQLite) and artifacts (./mlartifacts) both stay on disk and offline --
        # the same shape you scale to Postgres + object store on-prem.
        uri = tracking_uri or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(project) is None:
            mlflow.create_experiment(
                project, artifact_location=Path("mlartifacts").absolute().as_uri()
            )
        mlflow.set_experiment(project)
        self._run_name = run_name

    def start_run(self, config: dict) -> None:
        self._mlflow.start_run(run_name=self._run_name)
        self.log_params(config)

    def log_params(self, params: dict) -> None:
        # MLflow params must be scalars; drop nested/non-scalar values.
        flat = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
        if flat:
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_figure(self, name: str, fig: Any) -> None:
        self._mlflow.log_figure(fig, f"{name}.png")

    def log_model(self, path: str, name: str, metadata: Optional[dict] = None) -> None:
        # Log the checkpoint file under an artifact folder; metadata is logged as tags.
        if metadata:
            self._mlflow.set_tags({f"model.{k}": v for k, v in metadata.items()})
        self._mlflow.log_artifact(path, artifact_path="model")

    def finish(self) -> None:
        self._mlflow.end_run()


class NoOpTracker:
    """Discards everything. Used for tests and quick offline runs."""

    def start_run(self, config: dict) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None: ...
    def log_figure(self, name: str, fig: Any) -> None: ...
    def log_model(self, path: str, name: str, metadata: Optional[dict] = None) -> None: ...
    def finish(self) -> None: ...


def build_tracker(config: dict) -> Tracker:
    """Return a tracker chosen by ``config['tracker']['backend']``.

    Falls back to the legacy flat keys (``wandb_project`` / ``run_name``) so old
    configs keep working.
    """
    tcfg = config.get("tracker", {}) or {}
    backend = str(tcfg.get("backend", "none")).lower()
    project = tcfg.get("project") or config.get("wandb_project", "vqa-chest")
    run_name = tcfg.get("run_name") or config.get("run_name")

    if backend == "wandb":
        return WandbTracker(project=project, run_name=run_name)
    if backend == "mlflow":
        return MLflowTracker(
            project=project, run_name=run_name, tracking_uri=tcfg.get("tracking_uri", "")
        )
    if backend in ("none", "noop", ""):
        return NoOpTracker()
    raise ValueError(
        f"Unknown tracker backend: {backend!r} (expected 'wandb', 'mlflow', or 'none')"
    )
