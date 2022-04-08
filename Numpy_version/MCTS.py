import tensorflow as tf
import numpy as np
import random
from Numpy_version.Board import *
from Numpy_version.Deck import deck
from NNet_architecture import create_model


def init_game(deck: list) -> list:

    init_board_2D = np.array([[-1, -1, -2, -1, -1],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [1,  1,  2,  1,  1]])
    init_deck = random.sample(deck, 5)
    init_board_state = get_board_state(init_board_2D, init_deck, 1)
    # root = [board_state, action, prior, value, number_of_visit, [children]]
    root = [init_board_state, None, 0, 0, 0, []]

    return root


def expand(state: np.array, possible_policy: list):
    """ Expand the node with all children with a positive probability, 
        the policy is obtained by the nn"""

    board_state = state[0]

    # Softmax on positive policy
    possible_policy = [np.exp(proba - np.max(possible_policy))
                       if proba != 0 else 0 for proba in possible_policy]
    possible_policy = possible_policy / np.sum(possible_policy)

    state[-1] = [[move(board_state, i), i, proba, 0, 0, []]
                 for i, proba in enumerate(possible_policy) if proba != 0]


def get_best_child(state: list, c_param=0.25) -> list:

    best_score = -np.inf
    best_child = None

    for child in state[-1]:
        # 19652 and 1.25 come from UCB formula
        pb_c = np.log((state[4] + 19652 + 1) / 19652) + 1.25
        pb_c *= np.sqrt(state[4]) / (child[4] + 1)
        prior_score = pb_c * child[2]
        value_score = get_value(child)
        score = prior_score + value_score
        
        # # Prior is at index 2, number_of_visit at index 4
        # score = -get_value(child) + c_param * child[2] * \
        #     np.sqrt(state[4]) / (child[4] + 1)
        
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def get_value(state: list) -> int:
    """ Return the the value of a state

    Args:
        state (list): State of the game

    Returns:
        int: Value of the state
    """
    # value at index 3, number_of_visit at index 4
    if state[4]:
        return state[3] / state[4]
    else:
        return 0

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(state: list):

    priors = [child[2] for child in state[-1]]
    noise = np.random.gamma(0.3, 1, len(priors))
    frac = 0.25
    for i, (p, n) in enumerate(zip(priors, noise)):
        state[-1][i][2] = p * (1 - frac) + n * frac
        
    return state

def simulate(state: list, nb_simulation: int, model):

    if not state[-1]:
        policy, value = model.predict(state[0].reshape((1, 5, 5, 10)))
        possible_policy = get_legals_moves(
            state[0], policy[0]).flatten()
        expand(state, possible_policy)

    state = add_exploration_noise(state)

    for _ in range(nb_simulation):
        if (_+1) % 50 == 0:
            print(_+1)

        node_to_expand = state
        search_path = [node_to_expand]

        # select the node
        while node_to_expand[-1]:
            node_to_expand = get_best_child(node_to_expand)
            search_path.append(node_to_expand)

        board_state = node_to_expand[0]
        value = get_reward_for_player(board_state)

        if value is None:
            policy, value = model.predict(board_state.reshape((1, 5, 5, 10)))
            possible_policy = get_legals_moves(
                board_state, policy[0]).flatten()

            expand(node_to_expand, possible_policy)
        backpropagate(search_path, value, board_state[0, 0, 9])


def backpropagate(search_path: list, value: float, to_play: int):
    """At the end of a simulation, we propagate the evaluation all the way up

    Args:
        search_path (list): List of state to go through
        value (float): The value to backpropagate
        to_play (int): The player
    """
    for state in reversed(search_path):
        state[3] += value if state[0][0, 0, 9] == to_play else (1 - value)
        state[4] += 1

def pretty_print(state):

    print(get_board_2D(state[0]))
    print("Prior: {} Count: {} Value: {}".format(
        state[2], state[4], get_value(state)))

    for child in state[-1]:
        print(get_board_2D(child[0]))
        print("Prior: {} Count: {} Value: {}".format(
            child[2], child[4], get_value(child)))