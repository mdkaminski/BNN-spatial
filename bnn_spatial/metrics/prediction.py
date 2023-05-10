"""
Prediction performance metrics
"""

import numpy as np
from scipy.special import ndtri

# Note: ndtri stands for normal (n) distribution (dtr) inverse (i)

# bnn_preds = [n_samples, n_test]
# y = [n_train]
# bnn_preds[:, inds] = [n_samples, n_train]

def rmspe(preds, obs, return_all=False):
    """
    Compute root mean-squared prediction error.

    :param preds: np.ndarray, shape (n_samples, n_train), validation set predictions
    :param obs: np.ndarray, shape (n_train), validation set targets / observations
    :param return_all: bool, specify if RMSPE is given for each MCMC sample (otherwise average is given)
    :return: np.ndarray or float, validation set RMSPE
    """
    sq_diffs = (preds - np.expand_dims(obs, 0)) ** 2
    sample_rmspe = np.sqrt(np.average(sq_diffs, axis=1))
    if return_all:
        return sample_rmspe
    else:
        return np.average(sample_rmspe)

def perc_coverage(preds, obs, pred_var, percent=90, return_all=False):
    """
    Compute X-percent coverage (default X = 90).

    :param preds: np.ndarray, shape (n_samples, n_train), validation set predictions
    :param obs: np.ndarray, shape (n_train), validation set targets / observations
    :param pred_var: np.ndarray, shape (n_train), validation set predictive variance
    :param percent: float, specify X for X-percent coverage (default 90)
    :param return_all: bool, specify if X-percent coverage is given for each MCMC sample (otherwise average is given)
    :return: np.ndarray or float, validation set X-percent coverage
    """
    tail_prob = 1 - percent / 100
    c = ndtri(1 - tail_prob / 2)  # quantile c with P(|Z| > c) = tail_prob
    pred_sd = np.expand_dims(np.sqrt(pred_var), 0)
    lower_bound = preds - c * pred_sd
    upper_bound = preds + c * pred_sd
    obs = np.expand_dims(obs, 0)
    indicator = (obs >= lower_bound) & (obs <= upper_bound)
    sample_coverage = np.average(indicator, axis=1)
    if return_all:
        return sample_coverage
    else:
        return np.average(sample_coverage)

def interval_score(preds, obs, pred_var, alpha=0.1, return_all=False):
    """
    Compute negatively-oriented interval score.

    :param preds: np.ndarray, shape (n_samples, n_train), validation set predictions
    :param obs: np.ndarray, shape (n_train), validation set targets / observations
    :param pred_var: np.ndarray, shape (n_train), validation set predictive variance
    :param alpha: float, probability of tail region
    :param return_all: bool, specify if interval score is given for each MCMC sample (otherwise average is given)
    :return: np.ndarray or float, validation set interval score
    """
    c = ndtri(1 - alpha / 2)  # quantile c with P(|Z| > c) = alpha
    pred_sd = np.expand_dims(np.sqrt(pred_var), 0)
    l = preds - c * pred_sd
    u = preds + c * pred_sd
    x = np.expand_dims(obs, 0)
    score = (u - l) + (2/alpha) * ((l - x) * (x < l) + (x - u) * (x > u))
    sample_score = np.average(score, axis=0)
    if return_all:
        return sample_score
    else:
        return np.average(sample_score)


