import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from copy import copy, deepcopy
from Numpy_version.Deck import deck
"""
Objects defining a state  :
- board
- visit number
- value

No need to keep track
- model (same will be used in all the game generated)
"""

board_2D = np.array([[-1, -1, -2, -1, -1],
                     [0,  0,  0,  0,  0],
                     [0,  0,  0,  0,  0],
                     [0,  0,  0,  0,  0],
                     [1,  1,  2,  1,  1]])

board_2D_2 = np.array([[-1, 0, -2, -1, -1],
                       [0,  -1,  0,  0,  0],
                       [0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0],
                       [1,  1,  2,  1,  1]])


def get_board_state(board_2D: np.array, cards: list, player: int) -> np.array:
    """Return the board state of size 5x5x10
    board_2D must be facing current player ! 

    Args:
        board_2D (np.array): Human readable board facing current player
        cards (list): Current list of cards (need to be given in the good order)
        player (int): Current player

    Returns:
        np.array: Board state of size 5x5x10
    """

    board_state = np.zeros((5, 5, 10))

    for i in range(5):
        for j in range(5):
            if board_2D[i][j] == 1:
                board_state[i, j, 0] = 1
            if board_2D[i][j] == 2:
                board_state[i, j, 1] = 1
            if board_2D[i][j] == -1:
                board_state[i, j, 2] = 1
            if board_2D[i][j] == -2:
                board_state[i, j, 3] = 1

    for c, card in enumerate(cards):
        board_state[:, :, c + 4] = card

    board_state[:, :, 9] = np.ones((5, 5)) * player

    return board_state


init_deck = random.sample(deck, 5)
board_state = get_board_state(board_2D, init_deck, 1)

# Board2D is useless ? Cards are needed ?
# root = [player, board2D, board_state, value, number_of_visit, [children]]
root = [1, board_2D, board_state]


for i in range(10):
    print("--------", i, "--------------")
    print(board_state[:, :, i])


def get_board_2D(board_state: np.array) -> np.array:

    # Two first plane always represent the current player's pawns
    # And the two next, the opponent player's pawns
    board_2D = board_state[:, :, 0] + board_state[:, :, 1] * 2 + \
              (board_state[:, :, 2] + board_state[:, :, 3] * 2) * -1

    return board_2D


def init_game():
    pass
# j1 cards, j2 cards, remaining card neccessary? they can be encoded in board_state ?


# player is 1 or -1
# board_2D is the current board
# board_state is the current board in 5x5x10
# j1 cards , j2 card and remaining cards are self explanatory
# value is the value of the current board obtained from the NN
# number of visit is the number of time the MCTS vsited the state
# chilren are all the possibles moves from the current board

# Order : get_player(), get_board_2D(), get_cards(), get_board_state(), get_value(), get_nb_visit(), get_children())


root = [1, board_2D, board_state, init_deck[0], init_deck[1],
        init_deck[2], init_deck[3], init_deck[4], None, 0, []]


# Can't use dictionnary, they are not supported by Numba
# def get_layer_codes(deck: list):

#     layer_code = {}
#     i = 0
#     for card in deck:
#         for move in card.moves:
#             layer_code[card, move] = i
#             i += 1
#     return layer_code

for i in range(5):
    for j in range(5):
        if deck[0][i][j] != 0:
            print(i, j)  # faire -(2, 2) pour avoir les mouvements


def move():
    pass


def get_value():
    pass


def get_children():
    pass


def get_legal_moves():
    pass
