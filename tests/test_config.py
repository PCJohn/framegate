"""Config behaviour: the shipped YAML must match the dataclass defaults, and the
three construction paths must work."""

from dataclasses import asdict, fields
from importlib import resources

import pytest
import yaml

from framegate import GateConfig


def test_default_yaml_matches_dataclass_defaults():
    text = resources.files("framegate").joinpath("default.yaml").read_text()
    data = yaml.safe_load(text)
    defaults = asdict(GateConfig())
    # every dataclass field is present in the yaml and equal
    for f in fields(GateConfig):
        assert f.name in data, f"{f.name} missing from default.yaml"
        assert data[f.name] == defaults[f.name], f"{f.name} differs"
    assert set(data) == set(defaults)   # and nothing extra


def test_in_code_override():
    cfg = GateConfig(min_scene_len=6, thumb=96)
    assert cfg.min_scene_len == 6 and cfg.thumb == 96


def test_from_yaml_with_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("min_scene_len: 30\nthumb: 64\n")
    cfg = GateConfig.from_yaml(str(p), thumb=96)     # file value overridden in code
    assert cfg.min_scene_len == 30 and cfg.thumb == 96


def test_from_yaml_default_is_loadable():
    assert GateConfig.from_yaml().min_scene_len == GateConfig().min_scene_len


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_real_key: 1\n")
    with pytest.raises(ValueError):
        GateConfig.from_yaml(str(p))


def test_config_is_immutable():
    cfg = GateConfig()
    with pytest.raises(Exception):
        cfg.thumb = 99
    assert cfg.replace(thumb=99).thumb == 99 and cfg.thumb != 99
