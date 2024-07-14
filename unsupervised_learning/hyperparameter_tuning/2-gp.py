#!/usr/bin/env python3
"""def update(self, X_new, Y_new): that updates a
Gaussian Process"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix
        between two matrices using the RBF kernel"""
        sqdist = (np.sum(
            X1**2, 1).reshape(-1, 1) + np.sum(
                X2**2, 1) - 2 * np.dot(X1, X2.T))
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard
        deviation of points in a Gaussian process"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K + 1e-8 * np.eye(len(self.X)))

        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """Updates the Gaussian Process with
        a new sample point"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
