import numpy as np
from collections import defaultdict



def expand(state: list, possible_policy: list):
    """ Expand the node with all children with a positive probability, 
        the policy is obtained by the nn"""

    # state will be list or np.array

    init_value = 0
    visit_number = 0
    board_state = np.array()

    # children = [child for child in possible_policy if child != 0]
    children = [(board_state, prior, init_value, visit_number)
                for prior in possible_policy if prior != 0]

    return state + children

    # for i, proba in enumerate(possible_policy):
    #     if proba != 0:
    #         # Create child
    #         pass
    #         # next_board = self.board.move(i)
    #         # self.children[i] = MonteCarloTreeSearchNode(
    #         #     model=self.model, board=next_board, prior=proba, parent=self)
    # pass


def simulate():
    pass


def backpropagate():
    pass


def best_child():
    pass
