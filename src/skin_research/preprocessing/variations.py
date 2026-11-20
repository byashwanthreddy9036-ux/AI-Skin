"""Experimental conditions and traceable generation of image variations.

A *condition* is a fixed recipe (optional degradation, then optional
enhancement). Every generated image gets a manifest row recording its source
image, recipe, parameters, seed, label, output path and quality metrics, so any
result can be traced back to the exact original pixels.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skin_research.config import AppConfig
from skin_research.data.loader import load_image, save_image
from skin_research.preprocessing.degradation import apply_degradation
from skin_research.preprocessing.enhancement import apply_enhancement
from skin_research.quality.assessment import assess_array
from skin_research.utils.logging import get_logger
from skin_research.utils.reproducibility import derived_seed

log = get_logger(__name__)

ORIGINAL = "original"


@dataclass(frozen=True)
class Condition:
    """One experimental condition."""

    name: str
    group: str  # original | degradation | enhancement | degrade_enhance
    degradation: str | None = None
    severity: str | None = None
    degradation_params: dict[str, Any] = field(default_factory=dict)
    enhancement: str | None = None
    enhancement_params: dict[str, Any] = field(default_factory=dict)

    def apply(self, image: np.ndarray, seed: int) -> np.ndarray:
        out = image
        if self.degradation:
            out = apply_degradation(out, self.degradation, self.degradation_params, seed)
        if self.enhancement:
            out = apply_enhancement(out, self.enhancement, self.enhancement_params)
        return out

    def recipe(self) -> dict[str, Any]:
        return asdict(self)


def build_conditions(cfg: AppConfig) -> list[Condition]:
    """Expand the experiment section of the config into a list of conditions."""
    exp = cfg.experiment
    conditions = [Condition(ORIGINAL, "original")]
    for deg in exp.degradations:
        for sev in exp.severities:
            conditions.append(Condition(f"{deg}__{sev}", "degradation", deg, sev, dict(cfg.degradations[deg][sev])))
    for enh in exp.enhancements_on_original:
        conditions.append(Condition(f"enhance__{enh}", "enhancement", enhancement=enh,
                                    enhancement_params=dict(cfg.enhancements[enh])))
    for deg, sev, enh in exp.degrade_then_enhance:
        conditions.append(Condition(f"{deg}__{sev}+{enh}", "degrade_enhance", deg, sev,
                                    dict(cfg.degradations[deg][sev]), enh, dict(cfg.enhancements[enh])))
    names = [c.name for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate condition names in the experiment configuration.")
    return conditions


def select_conditions(conditions: list[Condition], names: list[str] | None) -> list[Condition]:
    """Filter conditions; ``original`` (the baseline) is always kept.

    A name that exactly matches a condition selects only that condition;
    otherwise it is treated as a prefix (``gaussian_blur`` → all its severities
    and combinations).
    """
    if not names:
        return conditions
    all_names = {c.name for c in conditions}
    wanted: set[str] = {ORIGINAL}
    unknown = []
    for n in names:
        matches = {n} if n in all_names else {c for c in all_names if c.startswith(n)}
        if not matches:
            unknown.append(n)
        wanted |= matches
    if unknown:
        raise ValueError(f"Unknown condition(s) {unknown}. Known: {', '.join(c.name for c in conditions)}")
    return [c for c in conditions if c.name in wanted]


def _prepare_condition_dirs(root: Path, conditions: list[Condition], fmt: str, force: bool) -> None:
    """Create one folder per condition with its recipe; refuse to mix recipes.

    If a folder exists with a *different* recipe the user must pass ``force``,
    which deletes only that generated folder (never source data).
    """
    for cond in conditions:
        if cond.name == ORIGINAL:
            continue
        folder = root / cond.name
        recipe_file = folder / "condition.json"
        recipe = {**cond.recipe(), "image_format": fmt}
        if recipe_file.is_file() and json.loads(recipe_file.read_text()) != recipe:
            if not force:
                raise RuntimeError(
                    f"{folder} was generated with different parameters. Re-run with --force to regenerate "
                    "it, or change experiment.name to keep both versions.")
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        recipe_file.write_text(json.dumps(recipe, indent=2))


def iter_variations(
    cfg: AppConfig,
    table: pd.DataFrame,
    conditions: list[Condition],
    save_images: bool | None = None,
    force: bool = False,
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Yield ``(manifest_row, image)`` for every image × condition.

    Images are processed one source image at a time, so memory use is bounded
    by a single image and its variants. Existing identical outputs are reused
    (generation is deterministic), which makes interrupted runs resumable.
    """
    save = cfg.experiment.save_images if save_images is None else save_images
    fmt = cfg.experiment.image_format
    out_root = cfg.paths.variations_dir / cfg.experiment.name
    if save:
        _prepare_condition_dirs(out_root, conditions, fmt, force)

    for n, row in enumerate(table.itertuples(index=False), 1):
        source = Path(row.abs_path)
        original = load_image(source)
        for cond in conditions:
            seed = derived_seed(cfg.seed, row.image_id, cond.name)
            output_path: Path | None = None
            if cond.name == ORIGINAL:
                image, output_path = original, source
            else:
                target = (out_root / cond.name / str(row.image_id)).with_suffix(f".{fmt}")
                if save and target.is_file():
                    image, output_path = load_image(target), target
                else:
                    image = cond.apply(original, seed)
                    if save:
                        output_path = save_image(image, target, fmt)
            record = {
                "variation_id": f"{row.image_id}::{cond.name}",
                "image_id": row.image_id,
                "lesion_id": getattr(row, "lesion_id", None),
                "label": row.label,
                "condition": cond.name,
                "group": cond.group,
                "degradation": cond.degradation,
                "severity": cond.severity,
                "degradation_params": json.dumps(cond.degradation_params, sort_keys=True) if cond.degradation else None,
                "enhancement": cond.enhancement,
                "enhancement_params": json.dumps(cond.enhancement_params, sort_keys=True) if cond.enhancement else None,
                "seed": seed,
                "source_path": str(source),
                "output_path": str(output_path) if output_path else None,
                **assess_array(image, cfg.quality),
            }
            yield record, image
        if n % 25 == 0 or n == len(table):
            log.info("  processed %d/%d source images × %d conditions", n, len(table), len(conditions))


def generate_variations(cfg: AppConfig, table: pd.DataFrame, conditions: list[Condition],
                        force: bool = False) -> pd.DataFrame:
    """Generate (and save) all variations, returning the manifest."""
    rows = [record for record, _ in iter_variations(cfg, table, conditions, force=force)]
    return pd.DataFrame(rows)


def manifest_path(cfg: AppConfig) -> Path:
    return cfg.paths.variations_dir / cfg.experiment.name / "manifest.csv"
