"""Config behaviour: the generated YAML template stays in sync with the dataclass by
construction, and the construction paths work."""

from dataclasses import asdict

import pytest
import yaml

from framegate import GateConfig


def test_to_yaml_template_is_complete_and_roundtrips(tmp_path):
    text = GateConfig.to_yaml()
    data = yaml.safe_load(text)
    assert set(data) == set(asdict(GateConfig()))        # every field, nothing extra (cannot drift)
    p = tmp_path / "t.yaml"
    p.write_text(text)
    assert GateConfig.from_yaml(str(p)) == GateConfig()  # template loads back to the defaults


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
