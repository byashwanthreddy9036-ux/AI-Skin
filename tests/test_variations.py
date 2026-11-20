import json
from pathlib import Path

import numpy as np
import pytest

from skin_research.data.loader import load_image, prepare_dataset
from skin_research.preprocessing.variations import build_conditions, generate_variations, select_conditions


def test_conditions_cover_required_experiment_groups(tmp_config):
    conditions = build_conditions(tmp_config)
    groups = {c.group for c in conditions}
    assert groups == {"original", "degradation", "enhancement", "degrade_enhance"}
    names = {c.name for c in conditions}
    for required in ("original", "gaussian_blur__moderate", "underexposure__moderate", "overexposure__moderate",
                     "low_resolution__moderate", "low_contrast__moderate", "enhance__sharpen", "enhance__clahe"):
        assert required in names


def test_manifest_traces_every_variation(tmp_config):
    prepared = prepare_dataset(tmp_config)
    conditions = select_conditions(build_conditions(tmp_config), ["gaussian_blur__mild", "enhance__clahe",
                                                                  "gaussian_noise__moderate+denoise"])
    originals = {p: p.read_bytes() for p in prepared.paths}
    manifest = generate_variations(tmp_config, prepared.table, conditions)
    assert len(manifest) == len(prepared.table) * len(conditions)
    assert {p: p.read_bytes() for p in prepared.paths} == originals  # originals untouched
    row = manifest[manifest.condition == "gaussian_blur__mild"].iloc[0]
    assert json.loads(row.degradation_params) == tmp_config.degradations["gaussian_blur"]["mild"]
    assert row.label == prepared.table.set_index("image_id").loc[row.image_id, "label"]
    assert row.source_path.endswith(".jpg") and row.output_path.endswith(".png")
    combo = manifest[manifest.condition == "gaussian_noise__moderate+denoise"].iloc[0]
    assert combo.degradation == "gaussian_noise" and combo.enhancement == "denoise"
    orig = manifest[manifest.condition == "original"].iloc[0]
    assert orig.output_path == orig.source_path


def test_regeneration_is_deterministic_and_guards_changed_params(tmp_config):
    prepared = prepare_dataset(tmp_config)
    conds = select_conditions(build_conditions(tmp_config), ["gaussian_noise__mild"])
    m1 = generate_variations(tmp_config, prepared.table, conds)
    path = m1[m1.condition == "gaussian_noise__mild"].output_path.iloc[0]
    first = load_image(Path(path))
    m2 = generate_variations(tmp_config, prepared.table, conds)  # reuses identical files
    assert np.array_equal(first, load_image(Path(path)))
    assert (m1.output_path == m2.output_path).all()

    tmp_config.degradations["gaussian_noise"]["mild"] = {"sigma": 99}
    changed = select_conditions(build_conditions(tmp_config), ["gaussian_noise__mild"])
    with pytest.raises(RuntimeError, match="different parameters"):
        generate_variations(tmp_config, prepared.table, changed)
    generate_variations(tmp_config, prepared.table, changed, force=True)


def test_unknown_condition_rejected(tmp_config):
    with pytest.raises(ValueError):
        select_conditions(build_conditions(tmp_config), ["fog"])


def test_exact_name_does_not_pull_in_combinations(tmp_config):
    conds = build_conditions(tmp_config)
    exact = {c.name for c in select_conditions(conds, ["gaussian_blur__moderate"])}
    assert exact == {"original", "gaussian_blur__moderate"}
    prefix = {c.name for c in select_conditions(conds, ["gaussian_blur"])}
    assert {"gaussian_blur__mild", "gaussian_blur__severe", "gaussian_blur__moderate+sharpen"} <= prefix
