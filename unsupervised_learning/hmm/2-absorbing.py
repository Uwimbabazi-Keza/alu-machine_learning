#!/usr/bin/env python3
"""Determines if a Markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """Determines if a Markov chain is absorbing"""
    if not isinstance(P, np.ndarray):
        return False
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    
    n = P.shape[0]
    
    absorbing_states = np.diag(P) == 1
    
    if not np.any(absorbing_states):
        return False
    
    for i in range(n):
        if not absorbing_states[i]:
            visited = set()
            stack = [i]
            
            while stack:
                state = stack.pop()
                
                if state in visited:
                    continue
                visited.add(state)
                
                if absorbing_states[state]:
                    break
                
                for j in range(n):
                    if P[state, j] > 0 and j not in visited:
                        stack.append(j)
            else:
                return False
    
    return True
