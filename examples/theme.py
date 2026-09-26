"""The examples' shared look and controls: the dark dashboard theme of fastdet's demo,
applied through matplotlib's rcParams so every panel inherits it, plus the few pieces the
live viewers share -- a titled time-series axis, a monospace readout, and a window
wrapper with fastdet-demo's keys (``q``/``Esc`` quit, ``space`` pause, ``s`` save) that
also survives the user closing the window mid-frame.

Pure viewer code: nothing here is imported by the library.
"""

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

BG = "#181818"  # figure and panel background
EDGE = "#444444"  # spines, legend frame
TICK = "#bbbbbb"  # ticks, axis labels, grid
TEXT = "#eeeeee"  # titles, readouts
LEGEND_BG = "#242424"
# series colours, in the order the property cycle hands them out
AMBER, GREEN, BLUE, ROSE, VIOLET, YELLOW = (
    "#ffb347",
    "#78dc78",
    "#7fb8ff",
    "#ff7f9f",
    "#c9a0ff",
    "#ffe066",
)
SERIES = [AMBER, GREEN, BLUE, ROSE, VIOLET, YELLOW]


def apply() -> None:
    """Install the theme for every figure made afterwards."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": EDGE,
            "axes.labelcolor": TICK,
            "axes.labelsize": 8,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 9,
            "axes.titlelocation": "left",
            "axes.prop_cycle": cycler(color=SERIES),
            "xtick.color": TICK,
            "ytick.color": TICK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": TEXT,
            "grid.color": TICK,
            "grid.alpha": 0.25,
            "legend.facecolor": LEGEND_BG,
            "legend.edgecolor": EDGE,
            "legend.labelcolor": TEXT,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "figure.titlesize": 11,
        }
    )


def series_axis(ax, title, labels, history, colors=None):
    """A time-series panel over the last ``history`` frames: gridded, legend top-left.
    Returns one line per label, to be fed with ``set_ydata``."""
    ax.set_title(title)
    ax.set_xlim(0, history)
    ax.grid(alpha=0.25)
    colors = colors or SERIES
    lines = [
        ax.plot(np.zeros(history), color=c, label=lab)[0]
        for lab, c in zip(labels, colors, strict=False)
    ]
    if len(labels) > 1:
        ax.legend(loc="upper left", ncol=len(labels))
    return lines


def readout(ax, fontsize=10):
    """A monospace text block filling an axis (which is switched off)."""
    ax.set_axis_off()
    return ax.text(
        0.0,
        1.0,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=fontsize,
        linespacing=1.35,
    )


def heat_axis(ax, title, cmap, clim=None):
    """An image panel for a (G, G) map: no ticks, optional fixed colour range."""
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(
        np.zeros((2, 2), np.float32), cmap=cmap, aspect="auto", interpolation="nearest"
    )
    if clim:
        im.set_clim(*clim)
    return im


def interactive_backend() -> bool:
    """False under Agg and the other file-only backends (no window will ever open)."""
    name = mpl.get_backend().lower()
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        headless = backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    except ImportError:  # matplotlib < 3.9
        headless = getattr(mpl.rcsetup, "non_interactive_bk", [])
    return name not in {b.lower() for b in headless}


class Window:
    """A live figure with fastdet-demo's controls. ``closed`` turns true when the user
    closes the window or presses ``q``/``Esc``; ``space`` pauses inside :meth:`pump`;
    ``s`` asks for a save, which :meth:`pump` writes to ``save_prefix_<n>.png``. Under
    a non-interactive backend there is no window: drawing still works (for saving) and
    :meth:`block` returns at once."""

    def __init__(self, fig, title, save_prefix="framegate"):
        self.fig = fig
        self.quit = False
        self.paused = False
        self.save_requested = False
        self.save_prefix = save_prefix
        self.interactive = interactive_backend()
        self._saves = 0
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(title)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect(
            "close_event", lambda _event: setattr(self, "quit", True)
        )
        if self.interactive:
            plt.ion()
            plt.show(block=False)

    @property
    def closed(self) -> bool:
        return self.quit or not plt.fignum_exists(self.fig.number)

    def _on_key(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.quit = True
        elif event.key == " ":
            self.paused = not self.paused
        elif event.key == "s":
            self.save_requested = True

    def draw(self) -> float:
        """Redraw now; returns the milliseconds it took (0 once the window is gone)."""
        if self.closed:
            self.quit = True
            return 0.0
        t = time.perf_counter()
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:  # the backend tears the canvas down mid-call on close
            self.quit = True
            return 0.0
        return (time.perf_counter() - t) * 1e3

    def pump(self, seconds=0.001) -> bool:
        """Process window events (keys, close, a requested save); blocks while paused.
        Returns False once the window is closed, so the caller can stop cleanly."""
        try:
            plt.pause(seconds)
            while self.paused and not self.closed:
                plt.pause(0.05)
        except Exception:  # see draw()
            self.quit = True
        if self.save_requested and not self.closed:
            self.save_requested = False
            self._saves += 1
            path = Path(f"{self.save_prefix}_{self._saves:03d}.png")
            self.fig.savefig(path, dpi=110)
            print(f"saved {path}")
        return not self.closed

    def block(self) -> None:
        """Hold the final frame until the window is closed or ``q`` is pressed."""
        while self.interactive and self.pump(0.05):
            pass

    def close(self) -> None:
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
