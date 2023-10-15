#!/usr/bin/env python3
"""
a function def add_arrays(arr1, arr2): that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """a function def add_arrays(arr1, arr2)
    """
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for i in range(len(arr1)):
        sum_array.append(arr1[i] + arr2[i])
    return sum_array
