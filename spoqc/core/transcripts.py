"""The one transcript loader of a spoQC run.

The columns of the transcripts element that cli.py sets up (deduplicated index,
cell_id mapped to the obs index, feature_name recast to sorted categories, and the
--dev_test crop) are computed once, kept until `release()`, and handed to every
step that needs them. cli.py's dask setup is the only implementation of those
transforms; this module only materialises its result, as polars.
`feature_name` becomes a `pl.Enum` over the same categories, in the same order.
"""

from __future__ import annotations

import polars as pl
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray

_frames: dict[int, pl.DataFrame] = {}
_global_coordinates: dict[int, pl.DataFrame] = {}


def _read(sdata, columns: list[str]) -> pl.DataFrame:
    computed = sdata.points["transcripts"][columns].compute()
    frame = pl.from_pandas(computed)
    if "feature_name" in columns:
        categories = list(computed["feature_name"].cat.categories)
        frame = frame.with_columns(pl.col("feature_name").cast(pl.Enum(categories)))
    return frame.rechunk()


def load_transcripts(sdata, columns: list[str]) -> pl.DataFrame:
    """Return `columns` of the transcripts, in index order; each column is computed once per run."""
    key = id(sdata)
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
    key = id(sdata)
    if key in _global_coordinates:
        return _global_coordinates[key]
    axes = list(get_axes_names(sdata.points["transcripts"]))
    coords = load_transcripts(sdata, axes)
    data = DataArray(
        coords.to_numpy(), coords={"points": range(coords.height), "dim": axes}
    )
    # Private API: exact for every transformation type. spatialdata is pinned to 0.7.3
    # in pyproject.toml and requirements.txt; re-check this call when the pin moves.
    moved = get_transformation(
        sdata.points["transcripts"], "global"
    )._transform_coordinates(data)
    _global_coordinates[key] = pl.DataFrame({ax: moved.sel(dim=ax).data for ax in axes})
    return _global_coordinates[key]


def release(sdata) -> None:
    """Drop the cached transcripts of this SpatialData (a no-op if no step loaded them)."""
    _frames.pop(id(sdata), None)
    _global_coordinates.pop(id(sdata), None)
