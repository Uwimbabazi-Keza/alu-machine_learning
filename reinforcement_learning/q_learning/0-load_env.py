#!/usr/bin/env python3
"""loads the pre-made FrozenLakeEnv
evnironment from OpenAI’s gym"""

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the pre-made FrozenLakeEnv
    evnironment from OpenAI’s gym"""
    if desc is not None:
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
    elif map_name is not None:
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    else:
        random_map = generate_random_map(size=8)
        env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=is_slippery)
    return env
