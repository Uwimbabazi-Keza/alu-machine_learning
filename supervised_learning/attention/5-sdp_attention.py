#!/usr/bin/env python3
"""calculates the scaled dot product
attention"""

import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product
    attention"""

    matmul_qk = np.matmul(Q, K.transpose(-2, -1))
    dk = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = np.exp(scaled_attention_logits)
    attention_weights_sum = np.sum(attention_weights, axis=-1, keepdims=True)
    attention_weights /= attention_weights_sum

    output = np.matmul(attention_weights, V)

    return output, attention_weights
