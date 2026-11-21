## Important scientific caveats
- Dataset-derived labels, not independent lithological ground truth.
- Variable `fs` is the regression target, not an input feature; nevertheless its cutoff defines which rows remain in the training/evaluation population.
- Reference stress, prior preprocessing, predrill filling, and regional shift create residual deployment limitations.
- Spatial near-neighbours may remain correlated despite 1-km clustering.
- Finite-sample site-coverage guarantees require exchangeability of entire site units, a condition not established under domain shift.
