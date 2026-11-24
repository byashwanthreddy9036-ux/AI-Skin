"""Configuration loading.

The YAML file is parsed into small dataclasses so that typos in section names fail
fast, while the open-ended parameter tables (degradation severities, enhancement
settings) stay as plain dictionaries that are passed straight to the transforms.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "configs/config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent directory containing ``pyproject.toml``.

    ``SKIN_RESEARCH_ROOT`` overrides the search, which keeps the code free of
    machine-specific paths.
    """
    env_root = os.environ.get("SKIN_RESEARCH_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    candidates = [start or Path.cwd(), Path(__file__).resolve()]
    for base in candidates:
        for directory in [base, *base.parents]:
            if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
                return directory
    return Path.cwd().resolve()


@dataclass
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    variations_dir: Path
    sample_dir: Path
    models_dir: Path
    outputs_dir: Path

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def charts_dir(self) -> Path:
        return self.outputs_dir / "charts"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"


@dataclass
class SplitConfig:
    test_fraction: float = 1.0
    group_column: str = "lesion_id"
    evaluate_split: str = "test"


@dataclass
class DatasetConfig:
    name: str
    metadata_csv: Path
    image_root: Path
    labels: list[str]
    max_images: int | None = None
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class ModelConfig:
    adapter: str
    repo_id: str
    revision: str | None
    weights_file: str
    config_file: str
    device: str = "cpu"
    batch_size: int = 16
    num_threads: int = 4
    label_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    save_images: bool
    image_format: str
    degradations: list[str]
    severities: list[str]
    enhancements_on_original: list[str]
    degrade_then_enhance: list[list[str]]


@dataclass
class EvaluationConfig:
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    calibration_bins: int = 10
    primary_metric: str = "balanced_accuracy"


@dataclass
class AppConfig:
    seed: int
    root: Path
    source_file: Path | None
    paths: PathsConfig
    dataset: DatasetConfig
    model: ModelConfig
    quality: dict[str, Any]
    degradations: dict[str, dict[str, dict[str, Any]]]
    enhancements: dict[str, dict[str, Any]]
    experiment: ExperimentConfig
    evaluation: EvaluationConfig
    raw: dict[str, Any]

    def resolve(self, path: str | Path) -> Path:
        """Resolve a (possibly relative) path against the project root."""
        path = Path(path).expanduser()
        return path if path.is_absolute() else (self.root / path)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' is missing or is not a mapping.")
    return value


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    try:
        return cls(**values)
    except TypeError as exc:  # unknown or missing keys
        raise ConfigError(f"Invalid '{section}' section: {exc}") from exc


def config_from_dict(data: dict[str, Any], root: Path, source_file: Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig` from an already-parsed mapping."""
    raw = copy.deepcopy(data)  # pristine copy for snapshots; `data` is consumed below
    data = copy.deepcopy(data)
    root = root.resolve()

    def rp(value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else root / p

    paths_raw = _section(data, "paths")
    paths = _build(PathsConfig, {k: rp(v) for k, v in paths_raw.items()}, "paths")

    ds_raw = _section(data, "dataset")
    split = _build(SplitConfig, ds_raw.pop("split", {}) or {}, "dataset.split")
    ds_raw["metadata_csv"] = rp(ds_raw["metadata_csv"])
    ds_raw["image_root"] = rp(ds_raw["image_root"])
    dataset = _build(DatasetConfig, {**ds_raw, "split": split}, "dataset")
    if not 0 < dataset.split.test_fraction <= 1:
        raise ConfigError("dataset.split.test_fraction must be in (0, 1].")

    model = _build(ModelConfig, _section(data, "model"), "model")
    if model.batch_size < 1:
        raise ConfigError("model.batch_size must be >= 1.")

    experiment = _build(ExperimentConfig, _section(data, "experiment"), "experiment")
    evaluation = _build(EvaluationConfig, data.get("evaluation", {}) or {}, "evaluation")

    degradations = _section(data, "degradations")
    enhancements = _section(data, "enhancements")
    for name in experiment.degradations:
        if name not in degradations:
            raise ConfigError(f"experiment.degradations lists '{name}' but it has no parameters.")
        missing = [s for s in experiment.severities if s not in degradations[name]]
        if missing:
            raise ConfigError(f"Degradation '{name}' has no parameters for severities {missing}.")
    for name in experiment.enhancements_on_original:
        if name not in enhancements:
            raise ConfigError(f"Enhancement '{name}' has no parameters.")
    for entry in experiment.degrade_then_enhance:
        if len(entry) != 3:
            raise ConfigError(f"degrade_then_enhance entries need 3 items, got {entry}.")
        deg, sev, enh = entry
        if deg not in degradations or sev not in degradations[deg] or enh not in enhancements:
            raise ConfigError(f"Unknown degrade_then_enhance combination {entry}.")

    return AppConfig(
        seed=int(data.get("seed", 42)),
        root=root,
        source_file=source_file,
        paths=paths,
        dataset=dataset,
        model=model,
        quality=data.get("quality", {}) or {},
        degradations=degradations,
        enhancements=enhancements,
        experiment=experiment,
        evaluation=evaluation,
        raw=raw,
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (mappings merge, other values replace)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(config_path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config, resolving ``extends: <file>`` (relative to this file) recursively."""
    if config_path in seen:
        raise ConfigError(f"Circular 'extends' chain: {' -> '.join(map(str, (*seen, config_path)))}")
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {config_path} must contain a YAML mapping.")
    parent = data.pop("extends", None)
    if parent:
        base = _read_yaml((config_path.parent / parent).resolve(), (*seen, config_path))
        data = deep_merge(base, data)
    return data


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the YAML configuration (default: ``configs/config.yaml`` in the project root)."""
    root = find_project_root()
    config_path = Path(path) if path else root / DEFAULT_CONFIG_NAME
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path) if config_path.exists() else root / config_path
    data = _read_yaml(config_path.resolve())
    return config_from_dict(data, root=root, source_file=config_path.resolve())


def save_config_snapshot(config: AppConfig, destination: Path) -> Path:
    """Write the exact configuration used for a run next to its outputs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.raw, fh, sort_keys=False)
    return destination
