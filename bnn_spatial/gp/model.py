"""
Gaussian process prior and posterior
"""

import torch
import numpy as np
from . import base
from torch.distributions.multivariate_normal import MultivariateNormal


class GP(torch.nn.Module):
    def __init__(self, kern, jitter_level=1e-6):
        """
        Implementation of GP prior, and posterior after incorporating data.

        :param kern: instance of Kern child, covariance function or kernel
        :param jitter_level: float, for preventing non-PD error in Cholesky decompositions
        """
        super(GP, self).__init__()
        self.mean_function = base.Zero()  # always use zero mean
        self.kern = kern
        self.jitter_level = jitter_level
        self.X, self.Y, self.sn2 = None, None, None  # to be initialised with assign_data
        self.data_assigned = False  # status of whether data assigned

    def sample_functions(self, X, n_samples):
        """
        Produce samples from the prior latent functions.

        :param X: torch.Tensor, size (n_inputs, input_dim), inputs at which to generate samples
        :param n_samples: int, number of sampled functions
        :return: torch.Tensor, size (n_inputs, n_samples), with samples in columns
        """
        # X = X.reshape((-1, self.kern.input_dim))
        mu = self.mean_function(X)  # compute mean at inputs X

        var = self.kern.K(X)  # compute covariances at inputs X
        jitter = torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device) * self.jitter_level

        # Ensure sufficient jitter for successful Cholesky factorisation
        multiplier = 1
        while True:
            try:
                L = torch.linalg.cholesky(var + multiplier * jitter)  # Cholesky factor for sampling
                break
            except RuntimeError as err:
                multiplier *= 2.
                if float(multiplier) == float("inf"):
                    raise RuntimeError("increase to inf jitter")

        # Populate (n_inputs, n_samples) tensor with random numbers drawn from standard normal
        V = torch.randn(L.size(0), n_samples, dtype=L.dtype, device=L.device)

        # Generate output using Cholesky factor for sampling (using MultivariateNormal severely exhausts memory)
        return mu + torch.matmul(L, V)

    def assign_data(self, X, Y, sn2):
        """
        Assign data to enable posterior fit.

        :param X: torch.Tensor, training inputs
        :param Y: torch.Tensor, corresponding targets (observations)
        :param sn2: float, measurement error variance (could be initial estimate)
        """
        self.X = X.cpu()
        self.Y = Y.cpu()
        self.sn2 = sn2
        self.data_assigned = True

    def update_kernel(self, kernel, sn2):
        """
        For updating GP kernel following optimisation.

        :param kernel: instance of Kern child, new kernel with optimised hyperparameters
        :param sn2: float, measurement error variance estimate from optimisation
        """
        self.kern = kernel
        self.sn2 = sn2

    def predict_f(self, Xnew):
        """
        Compute predictive mean and variance for GP.

        :param Xnew: torch.Tensor, test inputs at which to fit
        :return: tuple, predictive mean, predictive variance
        """
        # Throw exception if data not assigned
        if not self.data_assigned:
            raise Exception('Assign data first')

        # Ensure devices match
        Xnew = Xnew.to(self.X.device)

        # Compute covariance matrices for test/training inputs (s for test, t for train)
        K_st = self.kern.K(Xnew, self.X)
        K_ts = K_st.T
        K_ss = self.kern.K(Xnew)
        K_tt = self.kern.K(self.X) + torch.eye(self.X.size(0), dtype=self.X.dtype, device=self.X.device) * self.sn2

        # Compute predictive mean
        L = torch.linalg.cholesky(K_tt).to(dtype=self.Y.dtype)
        A0 = torch.linalg.solve(L, self.Y)
        A1 = torch.linalg.solve(L.T, A0).to(dtype=K_st.dtype)
        fmean = torch.mm(K_st, A1)

        # Compute predictive variance
        L = L.to(dtype=K_ts.dtype)
        V = torch.linalg.solve(L, K_ts)
        fvar = K_ss - torch.mm(V.T, V)

        return fmean, fvar

    def predict_f_samples(self, Xnew, n_samples):
        """
        Produce samples from posterior latent functions.

        :param Xnew: torch.Tensor, inputs at which to generate samples
        :param n_samples: int, number of sampled functions
        :return: torch.Tensor, size (n_inputs, n_samples), function samples in columns
        """
        # Throw exception if data not assigned
        if not self.data_assigned:
            raise Exception('Assign data first')

        mu, var = self.predict_f(Xnew)
        jitter = torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device) * self.jitter_level

        # Ensure sufficient jitter for successful Cholesky factorisation
        multiplier = 1
        while True:
            try:
                L = torch.linalg.cholesky(var + multiplier * jitter)  # Cholesky factor for sampling
                break
            except RuntimeError as err:
                multiplier *= 2.
                if float(multiplier) == float("inf"):
                    raise RuntimeError("increase to inf jitter")

        V = torch.randn(L.size(0), n_samples, dtype=L.dtype, device=L.device)

        return mu + torch.matmul(L, V)

    def marginal_loglik(self, hyper=(None, None, None)):
        """
        Compute marginal log likelihood with model fit and complexity terms.

        :param hyper: tuple, amplitude, lengthscale, measurement error variance
        :return: tuple, log likelihood, model fit, complexity
        """
        # Throw exception if data not assigned
        if not self.data_assigned:
            raise Exception('Assign data first')

        # Determine kernel hyperparameters and measurement error
        ampl, leng, sn2 = hyper
        if sn2 is None:
            sn2 = self.sn2

        n_train = self.X.numel()  # compute training set size
        err_cov = sn2 * np.eye(n_train)  # compute measurement error covariance matrix

        K_tt = self.kern.K(X=self.X) + err_cov  # training set covariance matrix
        L = np.linalg.cholesky(K_tt)  # lower Cholesky factor
        A0 = np.linalg.solve(L, self.Y)
        A1 = np.linalg.solve(L.T, A0)

        modelfit = np.round(-0.5 * self.Y.T @ A1, 6).item()  # model fit term
        complexity = np.round(sum(np.log(np.diag(L))), 6)  # complexity penalty term
        loglik = np.round(modelfit - complexity - 0.5 * n_train * np.log(2 * np.pi), 6).item()  # log likelihood

        return loglik, modelfit, complexity
