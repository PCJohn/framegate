"""Learned maps: fastdet models scored on the gate's own imfeat pass (models.py).

The model under test is random -- splits, thresholds and leaves are drawn and written
by fastdet's real exporter into a real ``.fdt`` -- so no dataset or fit is needed. What
matters is the plumbing: the map fastdet computes from the gate's pass must be the map
fastdet computes on its own from the same frame, to the bit."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from framegate import Gate, GateConfig
from framegate.models import resolve_models

fastdet = pytest.importorskip("fastdet")

from fastdet.artifact import ModelArtifact  # noqa: E402
from fastdet.exporter import build_blob  # noqa: E402
from fastdet.features import FeatureExtractor, feature_level_bits  # noqa: E402

N_FEATURES = 96


def write_random_model(path, seed=0, n_trees=120, depth=4):
    """A loadable fastdet model of random trees over a random subset of the columns
    fastdet's default front-end produces (the front-end the gate matches)."""
    rng = np.random.default_rng(seed)
    cfg = fastdet.Config()
    cfg = replace(cfg, model=replace(cfg.model, n_trees=n_trees, depth=depth))
    names = FeatureExtractor(cfg.train).base_names
    keep = sorted(rng.choice(len(names), N_FEATURES, replace=False).tolist())
    columns = [names[i] for i in keep]
    counts = rng.integers(2, 12, N_FEATURES)
    borders = [np.sort(rng.uniform(0.05, 0.95, int(c))).tolist() for c in counts]
    trees = []
    for _ in range(n_trees):
        splits = []
        for _ in range(depth):
            f = int(rng.integers(N_FEATURES))
            splits.append(
                {
                    "float_feature_index": f,
                    "border": borders[f][int(rng.integers(len(borders[f])))],
                }
            )
        trees.append(
            {"splits": splits, "leaf_values": rng.normal(0.0, 0.3, 1 << depth).tolist()}
        )
    model_json = {
        "features_info": {
            "float_features": [
                {"feature_index": f, "borders": borders[f]} for f in range(N_FEATURES)
            ]
        },
        "oblivious_trees": trees,
    }
    blob, _info = build_blob(
        model_json,
        level_shift=[feature_level_bits(n)[1] for n in columns],
        leaf_bits=cfg.model.leaf_bits,
        leaf_chunk=cfg.model.leaf_chunk,
        coarse_trees=0,
    )
    ModelArtifact(config=cfg, feature_names=columns, blob=blob).save(path)
    return path


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    return write_random_model(tmp_path_factory.mktemp("models") / "text.fdt")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    img[100:160, 80:400] = 240  # a bright block: some structure for the trees to see
    return img


def test_model_map_matches_fastdet_on_the_same_frame(model_file, frame):
    """The gate's pass fed to the model == fastdet's own predict_proba: one front-end."""
    gate = Gate(GateConfig(models={"text": str(model_file)}))
    assert gate.models == ["text"]
    fs = gate.image(frame)
    assert "text" in fs.model_maps
    assert fs.text.shape == (64, 64) and fs.text.dtype == np.float32
    assert fs.text.min() >= 0.0 and fs.text.max() <= 1.0
    det = fastdet.Detector.load(model_file)
    np.testing.assert_array_equal(fs.text, det.predict_proba(frame))
    det.close()
    gate.close()


def test_video_frames_and_duplicates(model_file, frame):
    gate = Gate(GateConfig(models={"text": str(model_file)}))
    fs1, _ = gate.frame(frame)
    fs2, sig = gate.frame(frame.copy())  # byte-identical: stats reused, map included
    assert fs2 is fs1 and sig.freeze
    fs3, _ = gate.frame(np.roll(frame, 40, axis=1))
    assert fs3.text.shape == (64, 64) and not np.array_equal(fs3.text, fs1.text)
    gate.close()


def test_grayscale_frame_gets_a_map(model_file, frame):
    gate = Gate(GateConfig(models={"text": str(model_file)}))
    fs = gate.image(frame[:, :, 1])
    assert fs.text.shape == (64, 64) and np.isfinite(fs.text).all()
    gate.close()


def test_heuristic_without_a_model(frame):
    gate = Gate()  # nothing bundled in the repo: the heuristic text cue
    fs = gate.image(frame)
    assert gate.models == [] and fs.model_maps == {}
    assert fs.text.shape == (64, 64)
    assert fs.text.max() > 1.0  # an unnormalised score, not a probability


def test_bundled_directory_and_overrides(model_file, tmp_path, frame):
    bundled = tmp_path / "bundle"
    bundled.mkdir()
    (bundled / "text.fdt").write_bytes(model_file.read_bytes())
    (bundled / "face.fdt").write_bytes(model_file.read_bytes())
    cfg = GateConfig(models_dir=str(bundled))
    assert sorted(resolve_models(cfg)) == ["face", "text"]
    gate = Gate(cfg)
    fs = gate.image(frame)
    assert sorted(fs.model_maps) == ["face", "text"]
    np.testing.assert_array_equal(fs.model_maps["face"], fs.text)  # same random model
    gate.close()
    # None drops a bundled model; a path replaces it
    cfg = GateConfig(models_dir=str(bundled), models={"face": None})
    assert list(resolve_models(cfg)) == ["text"]
    cfg = GateConfig(models_dir=str(bundled), models={"text": str(model_file)})
    assert resolve_models(cfg)["text"] == model_file


def test_front_end_mismatch_is_refused(model_file):
    with pytest.raises(ValueError, match="stride"):
        Gate(GateConfig(stride=2, models={"text": str(model_file)}))
    with pytest.raises(ValueError, match="resize_interp"):
        Gate(GateConfig(resize_interp="nearest", models={"text": str(model_file)}))
    # the model records fastdet's default thumbnail rule ("pow2-fit"): a fixed square, or
    # another rule, is not it
    with pytest.raises(ValueError, match="thumb: model 'pow2-fit' vs gate 1024"):
        Gate(GateConfig(thumb=1024, models={"text": str(model_file)}))
    with pytest.raises(ValueError, match="thumb: model 'pow2-fit' vs gate 'pow2'"):
        Gate(GateConfig(thumb="pow2", models={"text": str(model_file)}))


def test_model_follows_the_frame_size(model_file):
    """Under the policy the thumbnail, and so the pass the model scores, follows each
    frame: the gate's map still equals fastdet's own on frames of several sizes."""
    gate = Gate(GateConfig(models={"text": str(model_file)}))
    det = fastdet.Detector.load(model_file)
    assert det.front_end_spec["thumb"] == "pow2-fit" == gate._gate.cfg.thumb
    rng = np.random.default_rng(4)
    for shape, size in (
        ((720, 1280, 3), (320, 512)),
        ((300, 500, 3), (128, 256)),
        ((1080, 1920, 3), (576, 1024)),
        ((1920, 1080, 3), (1024, 576)),
    ):
        frame = rng.integers(0, 256, shape, dtype=np.uint8)
        assert gate._gate.cfg.thumb_hw(frame.shape) == size
        fs = gate.image(frame)
        assert fs.thumb.shape == (*size, 3)
        np.testing.assert_array_equal(fs.text, det.predict_proba(frame))
    det.close()
    gate.close()


def test_missing_model_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="text"):
        Gate(GateConfig(models={"text": str(tmp_path / "nope.fdt")}))


def test_config_round_trip(model_file, tmp_path):
    yaml_text = GateConfig.to_yaml().replace(
        "models: {}", f"models: {{text: {model_file}}}"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = GateConfig.from_yaml(path)
    assert cfg.models == {"text": str(model_file)}
    assert resolve_models(cfg) == {"text": model_file}


def test_no_warning_without_models():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Gate().close()
