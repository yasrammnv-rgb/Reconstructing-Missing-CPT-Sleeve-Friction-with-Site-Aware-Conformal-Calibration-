# Quick start
1. Clone public repository after author publishes it.
2. Install pinned requirements in an isolated virtual environment.
3. Download the source dataset V1 from https://doi.org/10.17603/ds2-gqjm-t836 and place its MAT file under `dataset/`.
4. Build `features.pkl` from `src/` using the scripts listed in README.
5. Run the trained-model and site-calibration scripts after checking their relative working directories.
6. Compare result CSV rows with those in `results/`.  Do not interpret expected numbers as external field-validation evidence.
