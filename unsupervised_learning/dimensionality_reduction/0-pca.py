#!/usr/bin/env python3
"""Perform PCA on the dataset X to retain the
specified fraction of variance."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on the dataset X to retain the
    specified fraction of variance."""
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    num_components = np.argmax(cumulative_variance >= var) + 1

    W = eigenvectors[:, :num_components]
    
    return W
