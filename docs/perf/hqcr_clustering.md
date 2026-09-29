# hqcr clustering: changes from origin/dev

Measurements are on the full `breast2.zarr` table (167,780 cells, 313 genes), with the QC metrics from the integration2 `-s all` run's tmp files, on 4 threads.
The rebuilt input reproduces that run: its per-cell 3-level `qc_cluster_str` and `hqcr_traffic_light` match the run's on all 167,780 cells.

## 1. Performance [bit-identical]

`clustering_for_hqcr` ran `sc.pp.neighbors`, `sc.tl.umap` and `sc.tl.leiden`.
The kNN graph, `.uns['neighbors']`, the Leiden labels and numpy's global RNG state are unchanged.

| Change | Saving |
| --- | --- |
| `sc.tl.umap` removed. Nothing in hqcr reads `X_umap`, which is also not reproducible: two runs with `random_state` set differ in all 335,560 values. It still runs under `test_res=True`, for `test_resolutions_leiden`. | 81.6 s (py-spy, real run) |
| `core.knn.neighbors` skips `PyNNDescentTransformer`'s `compress_index()`: it builds search trees scanpy never queries, after the graph is returned. On sparse input pynndescent 0.6.0 allocates (n_nodes, 2, n_obs) float32 there, 24.7 GB for this table. | 23.2 s (py-spy); on 40,000 cells peak RSS 2.37 -> 0.97 GB |
| The random-projection forest is built on the thread budget. Tree RNG states are drawn before the joblib call, and NN-descent keeps `n_jobs=1`. | 66.3 -> 49.6 s (warm, 4 threads) |

Leiden (leidenalg, `n_iterations=-1`) is unchanged: the user kept it exact.

Tests: `tests/test_hqcr_clustering_differential.py`, `tests/realdata_hqcr_clustering.py`.

## 2. Correctness fix: hqcr no longer overwrites 13 genes of the expression matrix

**origin/dev:** `load_data_for_hqcr` set `X` on the view `sdata['table'][:, 0:13]`.
anndata writes that into the parent, so the 13 QC metrics replaced the first 13 genes of `sdata['table'].X`.
Since `qc_ambient`, `X` is the same object as `layers['normlog']`, so the normlog layer was overwritten too, with float32 QC values, for every step after hqcr.

**Now:** the view is taken of a copy of those 13 columns.
The clustering input is the same float32 CSR as before: data, indices and indptr are identical on the full table.

Table after hqcr, origin/dev vs now (`tests/realdata_hqcr_view_write.py`):

| Matrix | Entries that differ | Genes | Cells | Max abs difference |
| --- | --- | --- | --- | --- |
| `X` | 1,796,545 (nnz 11,894,972 vs 10,604,415) | 13: ABCC11, ACTA2, ACTG2, ADAM9, ADGRE5, ADH1B, ADIPOQ, AGR3, AHSP, AIF1, AKR1C1, AKR1C3, ALDH1A3 | 167,780 | 72,056 |
| `layers['normlog']` | the same (same object) | | | |
| `layers['raw']` | 0 | | | |
| `layers['normlogscale']` | 0 | | | |

Now `X` and `layers['normlog']` after hqcr equal the table before hqcr, entry for entry.

Later steps that read these matrices (cli.py order, integration2):

| Step | Reads | Effect on breast2 `-s all` (no annotation file) |
| --- | --- | --- |
| hqpr, combine_masks | pixels, not the expression matrix | none |
| cellcycleqc (`sc.tl.score_genes_cell_cycle` on `X`) | `X` = normlog | none: the panel has 0 S-phase genes, so the step stops before scoring. On a panel with S and G2M genes, its scores and phases were computed from the overwritten matrix: score_genes draws control genes from all genes. |
| modelqc | `layers['normlogscale']` | none: unchanged layer |
| markerqc (annotation file, `-s markerqc`) | `X` = normlog, marker columns | any marker among the 13 genes had QC values as expression |
| analysis_overview / cluster / category (annotation file) | `X` = normlog | `rna_qc_annotated.h5ad` and `rna_cluster.h5ad` stored the overwritten X and normlog. Neighbours use modelqc's `X_pca` (from normlogscale), so they are unaffected. |

Tests: `tests/test_hqcr_view_write.py` (clustering input identical, table untouched, and a mutation check that the original overwrote it).

## 3. Correctness fix: the hqcr tmp-file read no longer fails silently

**origin/dev:** `read_sdata_parquet_tmp_files` skips a tmp file when all its columns are already in `obs`.
It compared the parquet schema's names, which include the stored pandas index (`index`, or `__index_level_0__`), so no file was ever skipped.
In `-s all` the steps' columns are already in `obs`, so the join raised "columns overlap", and a `try/except` printed a warning and returned.
Every file after the failing one was then not even tried.
It did no harm only because every column was already in memory.

**Now:** the index columns (from the schema's pandas metadata) are excluded, only the columns `obs` does not have yet are read, and there is no `try/except`, so any failure raises.
Reading only the missing columns matters for a step run on its own: cli's mandatory block (`correct_for_valid_geometries`) has already set the four valid-geometry columns of generalqc's file.
origin/dev failed there too: with the listdir order of the real tmp folder it joined bubbleqc, cellqc and doubletqc, then swallowed the overlap at generalqc, so `canorm_transcript_counts` and voidqc's column were never read (`load_cell_df` would then raise a KeyError).
In a rerun with a stale `traffic_light_output_hqcr.parquet` in the tmp folder, hqcr's first read now joins it; `combine_priors_hqcr` overwrites the column before anything reads it.

Checks on the run's real tmp files (`tests/realdata_read_tmp_files.py`):
- `-s all`: all six `*_hqcr.parquet` files are skipped. `obs` is unchanged, and identical to what the original left.
- A step run on its own: all six files are read, identical to the original's read.
- A step run on its own, with the valid-geometry columns already set: the other columns are read, and the result is identical to the full read.

Tests: `tests/test_read_sdata_parquet_tmp_files.py` (against the verbatim original: the in-process case, the step on its own, the step with recomputed columns; a missing folder raises).
