"""framegate -- a fast, generic pre-pipeline gate for any vision input.

Run it first on an image or video frame to get cheap, broadly-useful signals
(blank, exposure, saliency / fine-texture maps) plus, for video, a motion map and
temporal signals (shot cut, freeze, fade, flicker) -- so a heavy model runs only
where and when it's worth it.

Quick start:
    from framegate import Gate
    gate = Gate()
    stats = gate.image(img)                       # single image
    stats, signals = gate.frame(frame)            # video frame

Or drive it as a publishing node that drops blank/frozen/duplicate frames and emits
a Packet (frame + stats + signals + shot_id + shot_group_id) only when work is worth
doing; shot_group_id is shared by recurrences of the same shot (re-identification):

    from framegate import Publisher
    pub = Publisher()
    for frame in source:
        pkt = pub.publish(frame)                  # None if dropped
        if pkt is not None:
            run_detector(pkt.frame, saliency=pkt.stats.saliency)
"""

from .config import GateConfig
from .gate import Gate
from .publish import Packet, Publisher
from .shotmem import Group, Shot, ShotMemory, ShotTracker
from .stats import FrameGate, FrameStats
from .stream import StreamAnalyzer, TemporalSignals

__all__ = [
    "Gate",
    "GateConfig",
    "FrameGate",
    "FrameStats",
    "StreamAnalyzer",
    "TemporalSignals",
    "Publisher",
    "Packet",
    "ShotMemory",
    "ShotTracker",
    "Shot",
    "Group",
]
__version__ = "0.1.0"
