#!/usr/bin/env python3
"""Performs PCA on a dataset and returns the transformed dataset with the specified
    number of dimensions."""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset and returns the transformed dataset with the specified
    number of dimensions."""
    X_centered = X - np.mean(X, axis=0)

    covariance_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, :ndim]

    T = np.dot(X_centered, W)

    return T
