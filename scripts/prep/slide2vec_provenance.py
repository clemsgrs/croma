"""Provenance of a slide2vec run, as the tileset sidecars record it.

A slide2vec run directory holds the resolved ``config.yaml`` and, next to every slide
embedding ``slide_embeddings/<sample_id>.pt``, a ``<sample_id>.meta.json`` whose
``compatibility`` block names the encoder identity (including a slide encoder's
``aggregation_revision``). slide2vec logs neither its own version nor whether the slide
encoder ran in eval mode, so the launcher records both in ``run_record.json`` before the
run starts::

    python scripts/prep/slide2vec_provenance.py record <run_dir> --encoder <preset>

:func:`source_runs` reads those three records back; the tileset writers
(``prepare_benchmark_embeddings.py``, ``prepare_pcabiop.py``) copy them into each
``<Model>.npy.json`` so a matrix says which runs, at which settings, produced it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

#: Written by the launcher into the run directory, beside slide2vec's ``config.yaml``.
RUN_RECORD_NAME = "run_record.json"


def record_run(run_dir: Path, *, slide2vec_version: str, slide_encoder_eval_mode: bool) -> Path:
    """Write ``run_dir/run_record.json``: the slide2vec version and the encoder's mode.

    Re-recording a run (a resume) must find the same record: a resume under another
    slide2vec version or encoder mode would mix two extractions in one run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_RECORD_NAME
    record = {
        "slide2vec_version": str(slide2vec_version),
        "slide_encoder_eval_mode": bool(slide_encoder_eval_mode),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError(f"{run_dir} was recorded as {existing}; refusing to resume it as {record}")
        return path
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def slide_encoder_eval_mode(encoder) -> bool:
    """Whether every torch module a loaded slide2vec encoder holds is in eval mode."""
    import torch

    modules = [value for value in vars(encoder).values() if isinstance(value, torch.nn.Module)]
    if not modules:
        raise ValueError(f"{type(encoder).__name__} holds no torch module to inspect")
    return not any(sub.training for module in modules for sub in module.modules())


def run_provenance(run_dir: Path, embedding_paths: Iterable[Path]) -> dict:
    """The provenance of the slide embeddings ``embedding_paths`` read from ``run_dir``."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    record_path = run_dir / RUN_RECORD_NAME
    # A run launched before the record existed never logged these: unknown, not guessed.
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    params = cfg["tiling"]["params"]
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "slide2vec_version": record.get("slide2vec_version"),
        "encoder_preset": cfg["model"]["name"],
        "tile_size_px": params["requested_tile_size_px"],
        "spacing_um": params["requested_spacing_um"],
        "aggregation_revision": _aggregation_revision(run_dir, embedding_paths),
        "slide_encoder_eval_mode": record.get("slide_encoder_eval_mode"),
    }


def source_runs(embedding_paths: Iterable[Path]) -> list[dict]:
    """One provenance record per slide2vec run among ``embedding_paths``, in first-use
    order, each with the number of embeddings it supplied. Paths outside a run are skipped.
    """
    by_run: dict[Path, list[Path]] = {}
    for path in embedding_paths:
        run_dir = run_dir_of(path)
        if run_dir is not None:
            by_run.setdefault(run_dir, []).append(Path(path))
    return [
        {**run_provenance(run_dir, paths), "n_samples": len(paths)}
        for run_dir, paths in by_run.items()
    ]


def run_dir_of(embedding_path: Path) -> Path | None:
    """The slide2vec run directory an embedding was read from, or None if it is not one's.

    slide2vec writes ``<run_dir>/slide_embeddings/<sample_id>.pt`` beside
    ``<run_dir>/config.yaml``.
    """
    path = Path(embedding_path)
    if path.parent.name != "slide_embeddings":
        return None
    run_dir = path.parent.parent
    return run_dir if (run_dir / "config.yaml").exists() else None


def _aggregation_revision(run_dir: Path, embedding_paths: Iterable[Path]) -> int | None:
    """The one ``aggregation_revision`` the slides' own sidecars record (None: unrecorded)."""
    revisions = set()
    for path in embedding_paths:
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text(encoding="utf-8"))
        revisions.add(meta.get("compatibility", {}).get("aggregation_revision"))
    if len(revisions) != 1:
        raise ValueError(
            f"{run_dir}: slides disagree on aggregation_revision ({sorted(map(str, revisions))})"
        )
    return next(iter(revisions))


def _record_installed(run_dir: Path, encoder_name: str) -> dict:
    """Record the installed slide2vec and the mode its ``encoder_name`` loads in."""
    from importlib.metadata import version

    from slide2vec.encoders.registry import encoder_registry

    # Constructed as slide2vec's workers construct it (on CPU, before .to(device)).
    encoder = encoder_registry.require(encoder_name)()
    path = record_run(
        run_dir,
        slide2vec_version=version("slide2vec"),
        slide_encoder_eval_mode=slide_encoder_eval_mode(encoder),
    )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="write <run_dir>/run_record.json before a run starts")
    record.add_argument("run_dir", type=Path)
    record.add_argument("--encoder", required=True, help="slide2vec preset name, e.g. titan")
    args = parser.parse_args(argv)
    result = _record_installed(args.run_dir, args.encoder)
    print(f"[run_record] {args.run_dir}: {json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
