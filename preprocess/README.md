# Preprocessing: the training slice cache

Two steps, from the competition DICOMs to the flat cache that training (and the image reads of the label pipeline)
read with `mmap`:

| step | command | output | time / size |
|---|---|---|---|
| 1 | `python preprocess/prepare_data.py --zip <competition zip>` (or without `--zip`: `RAW_DATA_DIR/train_series/`) | `CACHE_DIR/studies/<StudyInstanceUID>.npz`, 4,407 studies (training set + the 58 gold studies) | about 7 min with 48 processes; 89 GB |
| 2 | `python preprocess/build_cache.py` | `CACHE_DIR/slices.npy` (646,182 slices of 384 x 384, uint8) + `CACHE_DIR/index.csv` (21,334 series) | 95 GB |

Per study, one series per slot {sagittal, coronal, axial} x {fat-suppressed, other} (`Fat_Suppression` in
`train_series.csv`; the longest series when there are several). Each series keeps every slice in geometric order (more
than 48 slices: even subsampling to 48), is put in a canonical orientation from `ImageOrientationPatient`, right knees
are mirrored into the left-knee frame, and every slice is a 140 mm physical centre crop resized to 384 px with a
per-series 1-99 percentile window. `index.csv` keeps per series the slice count before and after subsampling, the
slice thickness and pixel spacing, and per study the laterality, sex and scanner ("Manufacturer|Model|field
strength"); the scanner is the hospital proxy of the Synovitis correction in `labels/`.

`inference/preprocess.py` is the same processing for the test layout (one study at a time, no files written).

**Checks.** On 6 training studies (a right and a left knee, a series of 60 slices that is subsampled, a study with 4
series, a gold study), both input modes (zip and extracted folder) give npz arrays and meta identical to the cache
the competition models were trained on, and the flat cache built from them has identical index rows and slices.
