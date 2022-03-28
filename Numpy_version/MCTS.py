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
    # root = [board_state, prior, value, number_of_visit, [children]]
    root = [init_board_state, 0, 0, 0, []]

    return root


def expand(state: np.array, possible_policy: list):
    """ Expand the node with all children with a positive probability, 
        the policy is obtained by the nn"""
    # add prior to root
    board_state = state[0]

    state[-1] = [[move(board_state, i), 0, 0, 0, []]
                 for i, proba in enumerate(possible_policy) if proba != 0]


def get_best_child(state: list, c_param=0.8) -> list:

    children = state[-1]
    best_score = -np.inf
    best_child = None

    for child in children:
        # Prior is at index 1, value at index 2, number_of_visit at index 3
        score = - child[2] + c_param * child[1] * \
            np.sqrt(state[3]) / (child[3] + 1)

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def simulate(state: list, nb_simulation: int, model):
    
    for _ in range(nb_simulation):
        print(_)
        search_path = [state]
        
        # select the node
        while state[-1]:
            state = get_best_child(state)
            search_path.append(state)
        
        board_state = state[0]
        value = get_reward_for_player(board_state)
        
        if value is None:
            policy, value = model.predict(board_state.reshape((1, 5, 5, 10)))
            possible_policy = get_legals_moves(board_state, policy[0]).flatten()

            expand(state, possible_policy)

        backpropagate(search_path, value, board_state[0, 0, 9])
    


def backpropagate(search_path: list, value: float, to_play: int):
    """At the end of a simulation, we propagate the evaluation all the way up

    Args:
        search_path (list): List of state to go through
        value (float): The value to backpropagate
        to_play (int): The player
    """
    print("search_path")
    print(len(search_path))
    for state in reversed(search_path):
        state[2] += value if state[0][0, 0, 9] == to_play else -value
        state[3] += 1
        
        
root = init_game(deck)

model = create_model()
simulate(root, 10, model)



def pretty_print(state):
    
    print(get_board_2D(state[0]))
    print("Prior: {} Count: {} Value: {}".format(state[1], state[3], state[2]))

    for child in state[-1]:
        print(get_board_2D(child[0]))
        print("Prior: {} Count: {} Value: {}".format(child[1], child[3], child[2]))

pretty_print(root)