"""Config behaviour: the generated YAML template stays in sync with the dataclass by
construction, and the construction paths work."""

from dataclasses import FrozenInstanceError, asdict

import pytest
import yaml

from framegate import GateConfig


def test_to_yaml_template_is_complete_and_roundtrips(tmp_path):
    text = GateConfig.to_yaml()
    data = yaml.safe_load(text)
    assert set(data) == set(
        asdict(GateConfig())
    )  # every field, nothing extra (cannot drift)
    p = tmp_path / "t.yaml"
    p.write_text(text)
    assert (
        GateConfig.from_yaml(str(p)) == GateConfig()
    )  # template loads back to the defaults


def test_in_code_override():
    cfg = GateConfig(min_scene_len=6, thumb=96)
    assert cfg.min_scene_len == 6 and cfg.thumb == 96


def test_from_yaml_with_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("min_scene_len: 30\nthumb: 64\n")
    cfg = GateConfig.from_yaml(str(p), thumb=96)  # file value overridden in code
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
    with pytest.raises(FrozenInstanceError):
        cfg.thumb = 99
    assert cfg.replace(thumb=99).thumb == 99 and cfg.thumb != 99


def test_thumb_is_a_side_or_a_policy(tmp_path):
    """The default thumbnail follows the frame ("pow2"); a side in pixels is a fixed
    square; the size, the stride cap and the samples per cell come from the shape."""
    cfg = GateConfig()
    assert cfg.thumb == "pow2-fit"
    assert cfg.thumb_hw((720, 1280, 3)) == (
        320,
        512,
    )  # 288 rows wanted: 1.6:1 is nearest
    assert cfg.thumb_hw((1080, 1920)) == (576, 1024)
    assert cfg.thumb_hw((1440, 2560, 3)) == (576, 1024)
    assert cfg.thumb_hw((1920, 1080, 3)) == (1024, 576)  # portrait
    assert cfg.thumb_hw((2160, 3840, 3)) == (1152, 2048)
    assert cfg.thumb_hw((40, 50, 3)) == (64, 64)  # floored at the grid
    assert cfg.thumb_hw((60, 2000, 3)) == (64, 64)  # a banner: the shape gives way
    assert cfg.samples_per_cell_for((1080, 1920, 3)) == 9
    assert cfg.samples_per_cell_for((720, 1280, 3)) == 5
    assert cfg.samples_per_cell_for((2160, 3840, 3)) == 18
    assert cfg.samples_per_cell == 9  # a 1080p frame's, under a policy
    square = GateConfig(thumb="pow2")
    assert square.thumb_hw((720, 1280, 3)) == (512, 512)
    assert square.thumb_hw((1080, 1920)) == (1024, 1024)
    assert square.samples_per_cell_for((720, 1280, 3)) == 8
    fixed = GateConfig(thumb=1024)
    assert fixed.thumb_hw((720, 1280, 3)) == (1024, 1024)  # a fixed square, upscaled
    assert fixed.samples_per_cell == 16 == fixed.samples_per_cell_for((720, 1280, 3))
    coarse = GateConfig(stride=4)
    assert coarse.stride_for((1024, 1024)) == 4
    assert coarse.stride_for((128, 128)) == 2  # capped at the 2 px cell
    assert coarse.stride_for((576, 1024)) == 4
    assert coarse.stride_for((128, 256)) == 2  # the shorter side's cell
    assert coarse.samples_per_cell_for((240, 420, 3)) == 1
    for bad in ("pow4", "", 0, True, 2.5):
        with pytest.raises(ValueError, match="thumb="):
            GateConfig(thumb=bad)
    with pytest.raises(
        ValueError, match="no samples"
    ):  # a fixed square is checked up front
        GateConfig(thumb=64, stride=2).pyramid_exps  # noqa: B018
    p = tmp_path / "c.yaml"
    p.write_text("thumb: pow2\n")
    assert GateConfig.from_yaml(str(p)).thumb == "pow2"
    p.write_text("thumb: 512\n")
    assert GateConfig.from_yaml(str(p)).thumb == 512
