#!/usr/bin/env python3
"""Performs PCA on a dataset and returns the transformed
dataset with the specified number of dimensions."""


import numpy as np


def pca(X, ndim):
    """Perform PCA on the dataset X to reduce 
    its dimensionality to ndim."""
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, :ndim]

    T = np.dot(X, W)

    return T
