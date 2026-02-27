import sys
from collections.abc import Iterator
from contextlib import contextmanager

from tqdm.auto import tqdm


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
