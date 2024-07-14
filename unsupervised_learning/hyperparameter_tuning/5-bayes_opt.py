#!/usr/bin/env python3
"""Public instance method def optimize(self, iterations=100): that
optimizes the black-box function"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """performs Bayesian optimization
        on a noiseless 1D Gaussian process"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample 
        location using Expected Improvement (EI)"""
        mu_s, sigma_s = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu_s - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu_s - best - self.xsi

        with np.errstate(divide='ignore'):
            z = imp / sigma_s
            EI = imp * norm.cdf(z) + sigma_s * norm.pdf(z)

        X_next = self.X_s[np.argmax(EI)]
        
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the acquisition function to
        obtain the next best sample point"""
        sampled_points = set()
        
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            
            if tuple(X_next) in sampled_points:
                break
            
            sampled_points.add(tuple(X_next))
            Y_next = self.f(X_next)
            self.update(X_next, Y_next)
        
        if self.minimize:
            optimal_idx = np.argmin(self.gp.Y)
        else:
            optimal_idx = np.argmax(self.gp.Y)
        
        X_opt = self.gp.X[optimal_idx]
        Y_opt = self.gp.Y[optimal_idx]
        
        return X_opt, Y_opt

    def update(self, X_new, Y_new):
        """Updates the Gaussian Proces"""
        self.gp.update(X_new, Y_new)
