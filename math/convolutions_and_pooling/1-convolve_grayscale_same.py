#!/usr/bin/env python3
"""
def convolve_grayscale_same(images, kernel):
that performs a same convolution on grayscale images
"""
import numpy as np

def convolve_grayscale_same(images, kernel):
    """
    def convolve_grayscale_same(images, kernel):
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = max(0, (h - kh + 1) // 2)
    pad_w = max(0, (w - kw + 1) // 2)

    oh, ow = h, w

    convolved_images = np.zeros((m, oh, ow))

    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for i in range(oh):
        for j in range(ow):
            image_region = padded_images[:, i:i + kh, j:j + kw]

            convolved_values = np.sum(image_region * kernel, axis=(1, 2))

            convolved_images[:, i, j] = convolved_values

    return convolved_images
