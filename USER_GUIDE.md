# User guide

The source `mat` file provides sounding ID, depth in metres, tip resistance `qc` (MPa), sleeve friction `fs` (MPa), WGS84 latitude/longitude, groundwater depth (m), and predrill depth (m). Main model target `y=log10(fs)` in MPa. Features are calculated from `qc`, depth, water and site coordinates only, but the `fs`-dependent target filter is used for evaluation and must be disclosed.

### Code outputs
features.py`: feature table and grouping labels (800 site groups at nominal 1 km DBSCAN spatial clustering).  
ml.py`: GroupKFold benchmark (five folds), model predictions and site-boostrap summaries.  
cqr.py`: original pointwise conformal calibration and diagnostic coverage.  
regionout_rf.py`: geographic holdouts without training on the held-out coordinate-defined region.  
`site_aware_cqr.py`: 75/25 site-calibrated quantile models, score calibration at the depth-point level, 90th-percentile site level and site-maximum level.

### Output interpretation
`RMSE` is in log10(MPa), not direct MPa error. `SBT accuracy` is relative to computed CPT-derived reference Ic, not independently logged USCS soil identity. The calibration methods report distinct estimands: a depth-point coverage fraction, site proportion meeting within-site >=90% coverage, and proportion of fully covered grouped sites. None guarantees new-region performance without exchangeability and external validation.

### Computational environment
Python, NumPy, pandas, SciPy, scikit-learn and matplotlib; see exact pinned versions in requirements. Models can take substantial CPU time, and the original random forest uses a 120,000-point subsample per fold.
