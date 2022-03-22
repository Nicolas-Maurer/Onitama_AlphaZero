from copy import deepcopy
import numpy as np
from collections import defaultdict



def expand(state, possible_policy: list):
    """ Expand the node with all children with a positive probability, 
        the policy is obtained by the nn"""

    # Need to copy the state, or the children won't have the same state
    # Because we modify it on place
    current_state = deepcopy(state)
    
    for i, proba in enumerate(possible_policy):
        if proba != 0:
            
            # next_board = move(current_state, i)
            # state[-1].append(next_board)
            
            pass


def simulate():
    pass


def backpropagate():
    pass


def best_child(state: list, c_param = 0.8):
    
    children = state[-1]
    best_score = -np.inf
    best_move = None

    for child in children:
        # Change string by their index in the list
        score = - child["value"] + c_param * child["prior"] * np.sqrt(state["Number_of_visit"]) / child["Number_of_visit"]
    
        if score > best_score:
            best_score = score
            best_move = child

    return best_move
