import tensorflow as tf
import numpy as np
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

    state[-1] = [[move(board_state, i), i, proba, 0, 0, []]
                 for i, proba in enumerate(possible_policy) if proba != 0]


def get_best_child(state: list, c_param=0.8) -> list:

    best_score = -np.inf
    best_child = None

    for child in state[-1]:
        # Prior is at index 2, number_of_visit at index 4
        score = -get_value(child) + c_param * child[2] * \
            np.sqrt(state[4]) / (child[4] + 1)

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


def simulate(state: list, nb_simulation: int, model):

    for _ in range(nb_simulation):
        if _ % 50 == 0:
            print(_)

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
        state[3] += value if state[0][0, 0, 9] == to_play else -value
        state[4] += 1

def pretty_print(state):

    print(get_board_2D(state[0]))
    print("Prior: {} Count: {} Value: {}".format(
        state[2], state[4], get_value(state)))

    for child in state[-1]:
        print(get_board_2D(child[0]))
        print("Prior: {} Count: {} Value: {}".format(
            child[2], child[4], get_value(child)))




# root = init_game(deck)

# model = create_model()
# simulate(root, 100, model)

# pretty_print(root)

# Objectif à battre : 67 secondes pour 1000 coups
# Sans eager mode : 35 secondes pour 1000 coups

# https://stackoverflow.com/questions/62681257/tf-keras-model-predict-is-slower-than-straight-numpy
# # Disables eager execution
# tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly())


if __name__ == "__main__":
    from Numpy_version.MCTS import init_game
    root = init_game(deck)
    model = create_model()
    
    import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()
    
    simulate(root, 1000, model)

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(20)
