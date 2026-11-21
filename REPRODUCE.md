# Reproduction and limitations

- Freeze source dataset version as DOI 10.17603/ds2-gqjm-t836, V1, n=984; cross-check no V2 records are merged into V1.
- Group using 1-km haversine DBSCAN, min_samples=1, then five-fold GroupKFold. The grouped site is not an independent geological unit, and region boxes are proxies for geography only.
- Distinguish `ml.py` 80/20 calibration from `cqr.py`/`site_aware_cqr.py` 75/25 calibration.
- The exact conformal finite-sample `k=ceil((m+1)*(1-alpha))` order statistic differs from NumPy's historical interpolated percentile and can change observed fractions.
- Pointwise, 90/90 site, and maximum-score site calibration use the same proper-training quantile predictions within the 75/25 experiment.
- Reference Ic uses derived unit weights and estimated stresses from observed fs; deployment-oriented rerun recalculates stresses using predicted fs. Neither procedure replaces independent ground truth.
- Predrill infill and `fs > 0.5kPa` filtering can bias real-missing-channel use. Evaluation excludes target-filtered values by construction.
- The `results` directory has compact numeric summaries and split manifest; full raw predictions remain in the larger author analysis package and are not reproduced here.
- To make a defensible public release, test each command from a fresh clone, document any path repairs and capture exact software/version/seed output. This package has not been verified as a clean-clone runnable release.
