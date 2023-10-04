import itertools
import numpy as np
from scipy.stats import anderson_ksamp
from polyleven import levenshtein


def reward_properties(base_rs, d, name, resolution=0.25):
  """ Populates d with reward statistics.

    Samples rewards from base_rs with prob. propto r.

    Parameters
    ----------
    base_rs: List of reward floats, for each unique x.
    d: dict, to be populated.
    name: String, names statistics.
    resolution: Float, divides range [0.0, 1.0] into equal parts.
  """
  z = sum(base_rs)
  expr = sum([r**2 for r in base_rs]) / z

  # Compute cdf
  p = {r: r/z for r in base_rs}
  quantiles = {}
  cum = 0
  threshold = 0.0
  for r in sorted(base_rs):
    cum += p[r]
    if cum >= threshold:
      quantiles[round(threshold, 2)] = r
      threshold += resolution
  quantiles[1.0] = max(base_rs)

  d[f'Z {name}'] = z
  d[f'rs {name}'] = base_rs
  d[f'AD samples {name}'] = np.random.choice(base_rs,
      size=min(len(base_rs), 100000),
      p=np.array(base_rs)/z)
  d[f'Expected r {name}'] = expr
  d[f'Quantiles {name}'] = quantiles
  print(f'Expected r {name}: {expr}')
  return


def dist_calibration_error(samples, target_quantiles):
  quantile_errors = []
  quantiles = sorted(list(target_quantiles.keys()))
  for i in range(len(quantiles) - 1):
    upper, lower = quantiles[i + 1], quantiles[i]
    target_frac = quantiles[i + 1] - quantiles[i]
    sampled_frac = len([s for s in samples if lower <= s <= upper]) / len(samples)
    quantile_errors.append(sampled_frac - target_frac)
  return sum([qe**2 for qe in quantile_errors])


def anderson_darling(sampled_rs, expected_rs):
  """ Compute Anderson-Darling statistic, normalized
      to p=0.05 threshold.

      Lower statistic (is better) means we accept the null hypothesis (p > 0.05)
      that the distributions are the same.
      above 1 means p < 0.05 - less than 5% chance the distributions are the same
      below 1 means p > 0.05 - more than 5% chance the distributions are the same
      Statistic can be negative.

      Parameters
      ----------
      sampled_rs: List
        A list of sampled reward floats
      expected_rs: List
        A list of reward floats from the ground-truth distribution,
        sampled with probability proportional to r
  """
  ad, crits, _ = anderson_ksamp([sampled_rs, expected_rs])
  ad_score = ad / crits[2]
  return ad_score


def diversity(data, dist_func=levenshtein):
  """ Average pairwise distance between data. """
  dists = [dist_func(*pair) for pair in itertools.combinations(data, 2)]
  n = len(data)
  return sum(dists) / (n*(n-1))


def novelty(new_data, old_data, dist_func=levenshtein):
  scores = [min([dist_func(d, od) for od in old_data]) for d in new_data]
  return np.mean(scores)


