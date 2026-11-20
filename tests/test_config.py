import copy

import pytest

from skin_research.config import ConfigError, config_from_dict, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.model.batch_size >= 1
    assert cfg.dataset.metadata_csv.is_absolute()
    assert set(cfg.model.label_map.values()) == set(cfg.dataset.labels)
    for name in cfg.experiment.degradations:
        assert set(cfg.experiment.severities) <= set(cfg.degradations[name])


def test_unknown_degradation_rejected(tmp_path):
    raw = copy.deepcopy(load_config().raw)
    raw["experiment"]["degradations"].append("does_not_exist")
    with pytest.raises(ConfigError):
        config_from_dict(raw, root=tmp_path)


def test_relative_paths_resolve_against_root(tmp_path):
    cfg = config_from_dict(load_config().raw, root=tmp_path)
    assert cfg.paths.outputs_dir == tmp_path.resolve() / "outputs"


def test_config_snapshot_is_serialisable(tmp_path):
    import yaml

    from skin_research.config import save_config_snapshot

    cfg = load_config()
    out = save_config_snapshot(cfg, tmp_path / "snap.yaml")
    assert yaml.safe_load(out.read_text())["model"]["repo_id"] == cfg.model.repo_id
