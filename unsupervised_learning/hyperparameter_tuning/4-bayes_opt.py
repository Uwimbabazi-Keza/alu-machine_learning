#!/usr/bin/env python3
"""def acquisition(self): that calculates the next
best sample location:"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization
    on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor for Bayesian Optimization"""
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

        x = self.X_s[np.argmax(EI)]
        
        return x, EI
