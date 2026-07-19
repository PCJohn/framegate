"""Publisher: the drop policy (blank/freeze/duplicate) and the inferred stream
metadata (frame_id over all inputs, shot_id bumping on cuts) that a subscriber
relies on. Driven exactly as a downstream user would."""

from framegate import Packet, Publisher
import synth


def test_blank_frame_is_dropped():
    pub = Publisher()
    assert pub.publish(synth.black()) is None
    assert pub.publish(synth.white()) is None  # solid white is blank too


def test_first_content_frame_publishes():
    pub = Publisher()
    pkt = pub.publish(synth.noisy(synth.hsv_scene(60, 2)))
    assert isinstance(pkt, Packet)
    assert pkt.frame_id == 0 and pkt.shot_id == 0


def test_duplicate_and_freeze_are_dropped():
    c = synth.hsv_scene(60, 2)
    pub = Publisher()
    assert pub.publish(c) is not None  # first sighting publishes
    assert pub.publish(c) is None  # byte-identical -> freeze -> dropped
    assert pub.publish(c) is None


def test_frame_id_counts_dropped_frames():
    """frame_id is the true input index, so a published packet after some dropped
    frames reports the real position (gaps are visible), not a compacted count."""
    pub = Publisher()
    ids = []
    seq = [synth.black(), synth.black(), synth.noisy(synth.hsv_scene(60, 2))]
    for f in seq:
        pkt = pub.publish(f)
        if pkt is not None:
            ids.append(pkt.frame_id)
    assert ids == [2]  # two blanks dropped; the content frame is input index 2


def test_shot_id_increments_on_cut():
    a = [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(20)]
    b = [synth.noisy(synth.hsv_scene(60, 3)) for _ in range(8)]
    pub = Publisher()
    shots = [p.shot_id for p in map(pub.publish, a + b) if p is not None]
    assert shots[0] == 0  # opens on shot 0
    assert shots[-1] == 1  # exactly one cut seen -> ends on shot 1
    assert set(shots) == {0, 1}


def test_subscribers_receive_published_packets():
    got = []
    pub = Publisher()
    pub.subscribe(got.append)
    frames = [synth.black()] + [synth.noisy(synth.hsv_scene(60, 2)) for _ in range(3)]
    published = [p for p in map(pub.publish, frames) if p is not None]
    assert got == published  # callback sees exactly what publish() returned
    assert len(got) == 3  # the leading blank was dropped


def _shot(hue, seed, n=12):
    return [synth.noisy(synth.hsv_scene(hue, seed)) for _ in range(n)]


def _per_shot_groups(seq):
    """Drive a Publisher over `seq`; return the group id of each distinct shot_id."""
    pub = Publisher()
    seen = {}
    for f in seq:
        p = pub.publish(f)
        if p is not None:
            seen.setdefault(p.shot_id, p.shot_group_id)
    return [seen[k] for k in sorted(seen)]


def test_packet_carries_shot_group_id():
    pub = Publisher()
    pkt = pub.publish(synth.noisy(synth.hsv_scene(60, 2)))
    assert pkt.shot_group_id == 0  # first shot opens the first group


def test_cross_cut_reidentifies_recurring_shots():
    a, b = _shot(60, 2), _shot(120, 3)
    assert _per_shot_groups(a + b + a + b) == [0, 1, 0, 1]  # ABAB -> 2 groups
    c = _shot(30, 7)
    assert _per_shot_groups(a + b + c + a + b + c) == [0, 1, 2, 0, 1, 2]  # ABCABC


def test_distinct_shots_do_not_merge():
    seq = sum((_shot(h, s) for h, s in [(60, 2), (120, 3), (30, 7), (90, 5)]), [])
    groups = _per_shot_groups(seq)
    assert groups == [0, 1, 2, 3]  # four different shots -> four different groups
