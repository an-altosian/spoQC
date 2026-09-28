# ambientqc and hqtr transcript images: changes from origin/dev

Everything in `subworkflows/qc_ambient.py` and the hqtr-specific steps (transcript density, qv and ac images, global and local Moran's I, the qv/ac priors) gives the same values as origin/dev db00d98, bit for bit, except for the changes below, which the user decided on 2026-09-28.
Measurements are on a real 4000 x 4000 crop of `breast2.zarr` (x 16000-20000, y 10000-14000: 16,000,000 pixels, 547,140 transcripts, 2,222 cells), `-n 4`, with the tmp folder on NFS (`/home`).
Full-slide numbers (912,950,048 pixels) are linear extrapolations, x57.06.

## 1. Prior parquet layout: 1,000,000-row parts, only the columns that are read

The qv and ac steps write `{tmp}/hqtr_output_qv_prob` and `{tmp}/hqtr_output_ac_prob`.

**origin/dev:** dask part files of 10,000 rows, each with three float64 columns: `{x}_density`, `d_{x}_density` and `norm_p_{x}_density`.

**Now:** part files of 1,000,000 rows (`priors.hqtr.ac_or_qv.PART_ROWS`), with two columns: `{x}_density` and `norm_p_{x}_density`.
The index is unchanged (int64 pixel position, `__null_dask_index__`), and so are the values.

Readers (all found by grep over `spoqc/` and `tests/`):

| Reader | Columns it reads | Change |
| --- | --- | --- |
| `priors/combine_priors.combine_priors_hqtr` (hqtr clustering) | `norm_p_{qv,ac}_density` | reads any layout; no longer returns the prior divisions |
| `additional_analysis/analysis_funcs` (with an annotation file) | `qv_density`, `ac_density` | none (`dd.read_parquet(columns=...)`) |
| `unittests/test_all.py` (`-s unittest` against stored reference data) | whole directory, row-hash sum | its reference data must be regenerated: `d_*` is gone |

`d_{x}_density` had no reader, so it is no longer written.

| Measure (crop) | origin/dev | now |
| --- | --- | --- |
| qv files, size | 1,600 files, 133.6 MB | 16 files, 95.0 MB |
| ac files, size | 1,600 files, 146.4 MB | 16 files, 105.4 MB |
| Write time per prior (dask, 2 runs) | 6.8-9.1 s | 0.31-0.35 s (write_parts, 3.3-3.6 cores) |
| Same layout through write_parts | 2.5-5.0 s (NFS latency, 0.5-1.0 cores) | |
| Read time of `norm_p` (random data, 16 M rows) | 2.8-3.1 s | 0.22-0.24 s |

Full slide (extrapolated): 91,295 files and ~7.6-8.4 GB per prior become 913 files and ~5.4-6.0 GB; the dask write of ~390-520 s per prior becomes ~18-20 s.

Tests: `tests/test_hqtr_ambient_differential.py::TestPrior` (values, columns and divisions against origin/dev's prior), and the real crop in `tests/realdata_hqtr_ambient.py`.

### hqtr mask_raw keeps origin/dev's layout

origin/dev's hqtr `mask_raw` frame was split at the union of its own `chunk_size` divisions and the priors' 10,000-row divisions, because adding the priors aligned the frames.
`pixel_scoring_dask` now adds those 10,000-row divisions itself (`ORIGIN_PRIOR_PART_ROWS`), so hqtr `mask_raw` is byte-identical to origin/dev whatever the prior layout.
It therefore still has the file-count problem: 91,295 part files at full slide (hqpr `mask_raw`, at the default `--pixel_qc_chunk_size` of 200,000, has 4,565).
`{hqpr,hqtr}_output_mask_smoothed_raw` (the refinement, `chunk_size=10000`) has 91,295 files at full slide too.
Neither layout was changed; that needs a decision.

## Writers: one implementation

`core/parquet.write_parts` is spoQC's only writer of these per-pixel parquet directories: the qv/ac priors, `mask_raw` (`pixel_scoring_dask`) and `mask_smoothed_raw` (`pixel_scoring_refinement`).
For a given part layout it writes the bytes dask's `to_parquet` wrote, including NaN written as null.
`helperfuncs.ddf_to_parquet` is deleted; its verbatim copy in `tests/legacy/parquet_writer.py` is the reference in the tests.
