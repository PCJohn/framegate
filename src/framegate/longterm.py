"""L3 long-term memory: a read-only prototype index, built offline and loaded at the
start of a session (a folder of common/reference images -- document scans, known film
shots, or images whose presence should trigger loading specialised weights).

Unlike the L2 shot store, L3 never mutates during a session: it is built once by an
offline script (walk a folder -> imfeat pHash per image -> framestore -> serialise) and
memory-mapped at load, so it costs nothing to bring up and can be shared read-only
across processes. It is therefore a different, simpler object than framestore.Store --
query-only, no tail, no rebuild, no id reuse -- wrapping the same Hamming query kernel.

This module is a stub. The query seam is defined so the pipeline can call it, and the
loader/builder are marked for a later chat. When implemented, `load_prototypes(path)`
should mmap the saved arrays and return a `FrozenStore`; the C++ side needs a
save/mmap-load path on framestore (documented there as future work).
"""

from typing import List, Optional, Tuple


class FrozenStore:
    """Read-only pHash index over a fixed set of prototype images. Built offline,
    mmap-loaded at runtime. Payloads (source path, label, a weights pointer, ...) live
    in a parallel metadata list the caller owns, indexed by the ids this returns --
    same split as the live side: the store returns ids, Python owns meaning."""

    def __init__(self) -> None:
        # Populated by load_prototypes: the mmap'd framestore arrays + parallel metadata.
        self._store = None
        self._meta: List[dict] = []

    def query(self, phash: int, k: int = 1, max_dist: float = 0.1) -> Tuple:
        """Nearest prototypes to `phash`: (distances, ids) into `self._meta`."""
        raise NotImplementedError("L3 FrozenStore.query -- implement with mmap load")

    def meta(self, i: int) -> dict:
        return self._meta[i]

    def __len__(self) -> int:
        return len(self._meta)


def build_prototypes(image_dir: str, out_path: str) -> None:
    """Offline: walk `image_dir`, compute one pHash per image via imfeat, build a
    framestore index, and serialise it plus a parallel metadata table to `out_path`.
    Run once, ahead of time; the result is what load_prototypes mmaps."""
    raise NotImplementedError("L3 offline builder -- implement in a later pass")


def load_prototypes(path: str) -> Optional[FrozenStore]:
    """Runtime: mmap a prototype index built by build_prototypes. Cheap enough to call
    at session start; returns a query-only FrozenStore."""
    raise NotImplementedError("L3 loader -- implement with framestore mmap support")
