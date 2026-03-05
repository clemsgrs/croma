import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from tqdm.auto import tqdm

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text as RichText

    _rich_console = Console(stderr=True)
    _rich_available = True
except ImportError:  # pragma: no cover
    _rich_available = False
    _rich_console = None


def resolve_progress_mode(mode: str) -> bool:
    value = str(mode).strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    if value != "auto":
        raise ValueError("progress mode must be one of {'auto', 'on', 'off'}")
    return bool(sys.stderr.isatty())


@contextmanager
def progress_bar(
    *,
    total: int,
    desc: str,
    enabled: bool,
    unit: str = "it",
    leave: bool = True,
) -> Iterator[tqdm]:
    bar = tqdm(
        total=int(total),
        desc=str(desc),
        unit=str(unit),
        dynamic_ncols=True,
        disable=not bool(enabled),
        leave=bool(leave),
    )
    try:
        yield bar
    finally:
        bar.close()


def progress_write(message: str, *, enabled: bool) -> None:
    if bool(enabled):
        tqdm.write(str(message))
    else:
        print(str(message))


# ---------------------------------------------------------------------------
# Rich-based step ticker (used by benchmark.py)
# ---------------------------------------------------------------------------


class _NoopStepTicker:
    """Plain-text fallback used when rich is unavailable or progress is off."""

    def __init__(self) -> None:
        self._t0: float = 0.0

    def start(self, step: str) -> None:
        self._t0 = time.perf_counter()

    def done(self, step: str, *, cached: bool = False) -> None:
        elapsed = time.perf_counter() - self._t0
        if cached:
            print(f"  \u2713 {step:<8}  (cached)")
        else:
            print(f"  \u2713 {step:<8}  {elapsed:.1f}s")

    def log(self, msg: str) -> None:
        print(str(msg))


class StepTicker:
    """Animated rich step ticker for use inside a :func:`model_block` context."""

    def __init__(self, live: "Live") -> None:
        self._live = live
        self._completed: list = []
        self._current: str | None = None
        self._t0: float = 0.0

    def start(self, step: str) -> None:
        self._t0 = time.perf_counter()
        self._current = step
        self._live.update(self._renderable())

    def done(self, step: str, *, cached: bool = False) -> None:
        elapsed = time.perf_counter() - self._t0
        if cached:
            self._completed.append(RichText(f"  \u2713 {step:<8}  (cached)", style="dim"))
        else:
            self._completed.append(RichText(f"  \u2713 {step:<8}  {elapsed:.1f}s"))
        self._current = None
        self._live.update(self._renderable())

    def log(self, msg: str) -> None:
        self._live.console.print(str(msg))

    def _renderable(self):
        parts = list(self._completed)
        if self._current is not None:
            parts.append(Spinner("dots", text=f" {self._current}"))
        return Group(*parts)


@contextmanager
def model_block(model: str, idx: int, total: int, *, enabled: bool):
    """Context manager that renders an animated per-model progress block.

    Yields a :class:`StepTicker` for reporting per-step progress.
    Falls back to :class:`_NoopStepTicker` when rich is unavailable or
    *enabled* is ``False``.
    """
    header = f"\n=== {model} ({idx}/{total}) ==="
    if not enabled or not _rich_available:
        print(header)
        yield _NoopStepTicker()
        return
    _rich_console.print(header)
    with Live(console=_rich_console, refresh_per_second=10) as live:
        yield StepTicker(live=live)
