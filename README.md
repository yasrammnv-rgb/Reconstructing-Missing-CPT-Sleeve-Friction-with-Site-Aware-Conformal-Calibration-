# Site-aware reconstruction of missing CPT sleeve friction

**Manuscript:** Reconstructing Missing CPT Sleeve Friction with Site-Aware Conformal Calibration and Regional Validation
**Author:** Yasemin Manav, Department of Civil Engineering, Pamukkale University, Denizli, Türkiye (ORCID: 0000-0002-7065-3280)
**Contact:** ymanav@pau.edu.tr
**License:** MIT (see `LICENSE`)

This code accompanies a site-grouped, region-held-out CPT sleeve-friction reconstruction and uncertainty workflow. It is **not** a new random-forest algorithm, and the observed 90/90 test fraction is **not** a guaranteed performance level under geographic shift.

## Source data
- Public research dataset: Sanger et al. (2024), North America CPT database, **version 1** (984 soundings), https://doi.org/10.17603/ds2-gqjm-t836
- Download `NorthAmerica_Dataset_V1.mat` under the provider's access and usage terms. **Do not confuse it with version 2 (1,839 soundings).**
- The dataset is not redistributed in this repository.

## Repository layout
All scripts, documentation and archived result tables are in the repository root:
- Analysis scripts: `sbt.py`, `features.py`, `load.py`, `ml.py`, `cqr.py`, `regionout_rf.py`, `regional_stress_test.py`, `stats.py`, `extra_stats.py`, `end_to_end_audit.py`, `make_site_splits.py`, `site_aware_cqr.py`
- Figure scripts: `figs.py`, `fig6.py`, `create_site_aware_figure.py`
- Synthetic example: `site_score_demo.py`
- Archived results reported in the manuscript: `*.csv`
- Documentation: `QUICK_START.md`, `USER_GUIDE.md`, `REPRODUCE.md`, `ASSUMPTIONS_AND_BOUNDARIES.md`

## Installation
```bash
python -m venv .venv
# activate the virtual environment
python -m pip install -r requirements.txt
```
Exact package versions are listed in `requirements.txt`. `.pkl` files are Python pickle files; load them only from trusted sources.

## Reproduction outline
1. Clone or download this repository and work inside its root folder.
2. Place the V1 dataset at `dataset/NorthAmerica_Dataset_V1.mat`, or set the `CPT_DATA_PATH` environment variable to its absolute path.
3. Run `python sbt.py` and then `python features.py` to build the derived feature table (`features.pkl`).
4. Run `python ml.py` for the five-fold site-disjoint model benchmark, `python cqr.py` for the archived pointwise quantile analysis and `python regionout_rf.py` for the regional holdouts.
5. Run `python make_site_splits.py` and `python site_aware_cqr.py` for the site-aware 75/25 CQR calibration.
6. Compare the regenerated tables with the archived CSV files (for example `site_split_manifest.csv` and `site_aware_CQR_calibration.csv`). **Rerunning the scripts overwrites the archived CSV files of the same name; copy them to a separate folder first if you want to compare.**

The original analysis scripts were not re-engineered into a single command-line application; run them from the repository root in the order above.

## Synthetic usage example
Run `python site_score_demo.py`; it needs no external CPT data. It demonstrates the exact finite-sample order statistic and why three calibration sites cannot support a 90% finite-sample quantile. This is an educational toy example, not the manuscript benchmark.

## Input/output guide
See `USER_GUIDE.md` and `REPRODUCE.md` for file formats, outputs, assumptions and limitations.

## Scope of claims
**No external independent region-repository validation or laboratory/USCS reference labels.** `fs > 0.5 kPa` is a target-dependent filter that cannot be applied when fs is truly missing; density/stress reconstruction is a diagnostic and not field validation. Coordinates may encode archive-specific effects.

## Citation
If you use this code, please cite the accompanying article (citation details will be added after publication).
