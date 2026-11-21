"""Illustrative, synthetic site-block calibration (no CPT observations)."""
import math
import numpy as np


def score_quantile(scores, alpha=0.1):
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def within_site_score(residuals, fraction=0.9):
    s = np.sort(np.asarray(residuals, dtype=float))
    return float(s[math.ceil(len(s) * fraction) - 1])


if __name__ == "__main__":
    # Each row represents already computed nonconformity scores in one site.
    example_sites = [np.array([0.1, 0.2, 0.3, 0.5]),
                     np.array([0.3, 0.4, 0.5, 0.6]),
                     np.array([0.1, 0.4, 0.4, 0.8])]
    scores_90 = [within_site_score(x, 0.9) for x in example_sites]
    q = score_quantile(scores_90, alpha=0.1)
    assert len(scores_90) == 3
    assert q == float('inf'), "Small-site finite-sample rule should return infinity."
    print("Synthetic toy calibration shows +infinity at alpha=0.10 for only 3 sites.")
    print("A practical 90/90 analysis needs substantially more exchangeable site units.")
