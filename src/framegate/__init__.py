"""framegate -- a fast, generic pre-pipeline gate for any vision input.

Run it first on an image or video frame to get cheap, broadly-useful signals
(blank, exposure, saliency, text regions, an ROI box) plus, for video, temporal
signals (shot cut, freeze, fade, flicker) -- so a heavy model runs only where and
when it's worth it.

Quick start:
    from framegate import Gate
    gate = Gate()
    stats = gate.image(img)                       # single image
    stats, signals = gate.frame(frame)            # video frame
"""

from .config import GateConfig
from .gate import Gate
from .stats import FrameGate, FrameStats
from .stream import StreamAnalyzer, TemporalSignals

__all__ = ["Gate", "GateConfig", "FrameGate", "FrameStats",
           "StreamAnalyzer", "TemporalSignals"]
__version__ = "0.1.0"
