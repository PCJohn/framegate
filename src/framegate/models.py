"""Learned per-cell maps from fastdet models, on the gate's own imfeat pass.

A model is a `fastdet <https://github.com/PCJohn/fastdet>`_ ``.fdt`` file whose front-end
is the gate's -- the same thumbnail (a fixed side or the same sizing policy), filter,
stride and grid pyramid -- so the imfeat result the gate already holds feeds it straight
through ``Detector.predict_from_imfeat``:
one pass for the gate's signals and for every model, and each model adds only its
scorer (well under a millisecond) per frame.

Models are found by name. ``cfg.models_dir`` (the package's ``models/`` folder by
default) supplies ``<name>.fdt`` for every file it holds, and ``cfg.models`` maps a name
to a path explicitly on top (``None`` drops a bundled one). ``FrameStats`` exposes the
maps as ``model_maps[name]``, a (G, G) float32 probability map each; the ``text``
property reads its model when one is loaded and falls back to the heuristic cue when
not, so there is always a text map. Any other name (``face``, ``person``, ...) is simply
a new map -- the library knows nothing about what the models detect.

fastdet is an optional dependency: a model named in ``cfg.models`` requires it, while
bundled models are skipped with a warning when it is missing, so an install without
fastdet keeps working on the heuristics.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from .config import GateConfig

MODEL_SUFFIX = ".fdt"
_INSTALL = "pip install git+https://github.com/PCJohn/fastdet"


def bundled_dir(cfg: GateConfig) -> Path:
    """Where bundled models live: ``cfg.models_dir``, relative to the package."""
    d = Path(cfg.models_dir)
    return d if os.path.isabs(cfg.models_dir) else Path(__file__).parent / d


def resolve_models(cfg: GateConfig) -> dict[str, Path]:
    """Model files by name: the bundled ``<name>.fdt`` files, then ``cfg.models`` on top
    (a path adds or replaces a name, ``None`` removes it)."""
    found: dict[str, Path] = {}
    d = bundled_dir(cfg)
    if d.is_dir():
        for f in sorted(d.glob(f"*{MODEL_SUFFIX}")):
            found[f.stem] = f
    for name, path in cfg.models.items():
        if path is None:
            found.pop(name, None)
        else:
            found[name] = Path(path)
    return found


def check_front_end(name: str, path: Path, spec: dict, cfg: GateConfig) -> None:
    """Raise unless the model's front-end (``Detector.front_end_spec``) is the gate's.

    ``thumb`` compares as the model records it: a side in pixels, or a policy name that
    both resolve per frame the same way (imfeat's rule, the size floored at the grid and
    the stride capped at the cell side)."""
    want = {
        "thumb": cfg.thumb,
        "stride": cfg.stride,
        "resize_interp": cfg.resize_interp,
        "space": "hsv",
        "input_space": "bgr",  # the gate hands imfeat the BGR thumbnail; imfeat converts
        "levels": [2**e for e in cfg.pyramid_exps],
        "extra_scales": [],
    }
    bad = {k: (spec.get(k), v) for k, v in want.items() if spec.get(k) != v}
    if bad:
        detail = ", ".join(
            f"{k}: model {m!r} vs gate {g!r}" for k, (m, g) in bad.items()
        )
        raise ValueError(
            f"model {name!r} ({path}) was not trained on this gate's front-end -- "
            f"{detail}. Match GateConfig to the model (or the model to the gate); "
            "extra scales are not supported."
        )


class ModelBank:
    """The loaded models of one gate. ``maps(result, shape)`` runs each of them on an
    imfeat result; ``close()`` joins their scorer threads."""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.detectors: dict = {}
        paths = resolve_models(cfg)
        if not paths:
            return
        try:
            from fastdet import Detector
        except ImportError as exc:
            explicit = sorted(n for n in paths if cfg.models.get(n))
            if explicit:
                raise ImportError(
                    f"models {explicit} need fastdet, which is not installed: {_INSTALL}"
                ) from exc
            warnings.warn(
                f"fastdet is not installed; bundled models {sorted(paths)} are skipped "
                f"and the heuristic maps are used ({_INSTALL})",
                stacklevel=2,
            )
            return
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"model {name!r}: no file at {path}")
            det = Detector.load(path, threads=cfg.feat_threads)
            check_front_end(name, path, det.front_end_spec, cfg)
            self.detectors[name] = det

    @property
    def names(self) -> list[str]:
        return list(self.detectors)

    def __bool__(self) -> bool:
        return bool(self.detectors)

    def maps(self, result, shape: tuple) -> dict[str, np.ndarray]:
        """Every model's (G, G) probability map for one imfeat result of a frame of
        source shape ``(H, W)``."""
        return {
            name: np.asarray(det.predict_from_imfeat(result, shape), dtype=np.float32)
            for name, det in self.detectors.items()
        }

    def close(self) -> None:
        for det in self.detectors.values():
            det.close()
        self.detectors = {}
