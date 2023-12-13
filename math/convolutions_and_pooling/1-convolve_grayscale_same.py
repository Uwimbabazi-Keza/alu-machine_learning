#!/usr/bin/env python3
"""
def convolve_grayscale_same(images, kernel):
that performs a same convolution on grayscale images
"""
import numpy as np

def convolve_grayscale_same(images, kernel):
    """
    def convolve_grayscale_valid(images, kernel):
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    
    ph = kh // 2
    pw = kw // 2
    
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=0)
    
    oh = h
    ow = w
    
    result = np.zeros((m, oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            result[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    
    return result
