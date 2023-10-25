#!/usr/bin/env python3
"""
def convolve_grayscale_valid(images, kernel):
that performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    def convolve_grayscale_valid(images, kernel):
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    oh, ow = h - kh + 1, w - kw + 1

    convolved_images = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            image_region = images[:, i:i + kh, j:j + kw]
            convolved_values = np.sum(image_region * kernel, axis=(1, 2))
            convolved_images[:, i, j] = convolved_values

    return convolved_images
