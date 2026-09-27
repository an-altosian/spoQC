"""The one transcript loader of a spoQC run.

Each column is read from the points parquet once, kept until `release()`, and
handed to every step that needs it. The frame equals what `.compute()` on the
dask element that cli.py sets up returns:
- rows are in the parquet parts' natural sort order (dask's order), which is the
  order of the deduplicated RangeIndex;
- `cell_id` is mapped with the cell id -> obs index map cli.py applies to the dask
  element (registered with `set_cell_id_map`), -1 where no cell matches;
- `feature_name` is cast to string, nulls become 'NaN', and it is a `pl.Enum`
  over the sorted names, the categories pandas' `astype('category')` gives.
A SpatialData without a path (the --dev_test / unittest crop) has no parquet of its
own; its columns come from one `.compute()` of the cropped dask element instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from dask.utils import natural_sort_key
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray

_frames: dict[str, pl.DataFrame] = {}
_cell_id_maps: dict[str, dict] = {}
_global_coordinates: dict[str, pl.DataFrame] = {}


def parquet_parts(zarr_path: str | Path) -> list[str]:
    parts = sorted(
        (
            str(p)
            for p in Path(zarr_path, "points", "transcripts", "points.parquet").glob(
                "part.*.parquet"
            )
        ),
        key=natural_sort_key,
    )
    if not parts:
        raise FileNotFoundError(f"no transcript parquet parts under {zarr_path}")
    return parts


def _key(sdata) -> str:
    return f"memory:{id(sdata)}" if sdata.path is None else str(sdata.path)


def _compute(sdata, columns: list[str]) -> pl.DataFrame:
    computed = sdata.points["transcripts"][columns].compute()
    frame = pl.from_pandas(computed)
    if "feature_name" in columns:
        categories = list(computed["feature_name"].cat.categories)
        frame = frame.with_columns(pl.col("feature_name").cast(pl.Enum(categories)))
    return frame


def _read(sdata, columns: list[str]) -> pl.DataFrame:
    if sdata.path is None:
        return _compute(sdata, columns).rechunk()
    frame = pl.read_parquet(parquet_parts(sdata.path), columns=columns)
    if "cell_id" in columns:
        mapping = _cell_id_maps[_key(sdata)]
        frame = frame.with_columns(
            pl.col("cell_id").replace_strict(
                np.array(list(mapping.keys())),
                np.array(list(mapping.values()), dtype=np.int64),
                default=-1,
                return_dtype=pl.Int64,
            )
        )
    if "feature_name" in columns:
        names = frame["feature_name"].cast(pl.String).fill_null("NaN")
        frame = frame.with_columns(names.cast(pl.Enum(sorted(names.unique()))))
    return frame.rechunk()


def set_cell_id_map(sdata, mapping: dict) -> None:
    """Register the cell id -> integer obs index map; cli.py captures it before obs is re-indexed."""
    _cell_id_maps[_key(sdata)] = mapping


def load_transcripts(sdata, columns: list[str]) -> pl.DataFrame:
    """Return `columns` of the transcripts; each column is read from disk once per run."""
    key = _key(sdata)
    frame = _frames.get(key)
    missing = [c for c in columns if frame is None or c not in frame.columns]
    if missing:
        new = _read(sdata, missing)
        frame = new if frame is None else frame.hstack(new)
        _frames[key] = frame
    return frame.select(columns)


def global_coordinates(sdata) -> pl.DataFrame:
    """The coordinates in the 'global' coordinate system, equal to `sd.get_centroids(transcripts, 'global').compute()`.

    Computed once per run with spatialdata's own transformation code, and kept until `release()`.
    """
    key = _key(sdata)
    if key in _global_coordinates:
        return _global_coordinates[key]
    axes = list(get_axes_names(sdata.points["transcripts"]))
    coords = load_transcripts(sdata, axes)
    data = DataArray(
        coords.to_numpy(), coords={"points": range(coords.height), "dim": axes}
    )
    moved = get_transformation(
        sdata.points["transcripts"], "global"
    )._transform_coordinates(data)
    _global_coordinates[key] = pl.DataFrame({ax: moved.sel(dim=ax).data for ax in axes})
    return _global_coordinates[key]


def release(sdata) -> None:
    """Drop the cached transcripts of this SpatialData (a no-op if no step loaded them)."""
    _frames.pop(_key(sdata), None)
    _global_coordinates.pop(_key(sdata), None)
