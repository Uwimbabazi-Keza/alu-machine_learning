#!/usr/bin/env python3
"""calculates the positional encoding
for a transformer"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding
    for a transformer"""

    pos_enc = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)

    return pos_enc
