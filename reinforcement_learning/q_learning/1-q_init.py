#!/usr/bin/env python3
"""loads the pre-made FrozenLakeEnv
evnironment from OpenAI’s gym"""

import numpy as np


def q_init(env):
    """Initialize the Q-table for
    the given environment."""
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    return q_table
