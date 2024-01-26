#!/usr/bin/env python3
"""Function def forward_prop(x, layer_sizes=[], activations=[])
"""


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for
    the neural network
    """

    prev_layer = x

    for i in range(len(layer_sizes)):
        n_nodes = layer_sizes[i]
        activation = activations[i] if i < len(activations) else None

        layer = create_layer(prev_layer, n_nodes, activation)

        prev_layer = layer

    return prev_layer
