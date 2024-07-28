#!/usr/bin/env python3
"""Perform PCA on the dataset X to retain the
specified fraction of variance."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on the dataset X and return the
    weights matrix that maintains the specified variance."""
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    cumulative_variance = np.cumsum(
        sorted_eigenvalues) / np.sum(sorted_eigenvalues)

    num_components = np.searchsorted(cumulative_variance, var) + 1

    W = sorted_eigenvectors[:, :num_components]

    return W
