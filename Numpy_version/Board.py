import numpy as np
import random
from copy import copy, deepcopy
from Numpy_version.Deck import deck
import tensorflow as tf
import numba as nb
import time


def get_board_state(board_2D: np.array, cards: list, player: int) -> np.array:
    """Return the board state of size 5x5x10

    ! board_2D must be facing current player ! 

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
            if board_2D[i][j] == 1 * player:
                board_state[i, j, 0] = 1
            if board_2D[i][j] == 2 * player:
                board_state[i, j, 1] = 1
            if board_2D[i][j] == -1 * player:
                board_state[i, j, 2] = 1
            if board_2D[i][j] == -2 * player:
                board_state[i, j, 3] = 1

    for c, card in enumerate(cards):
        board_state[:, :, c + 4] = card

    board_state[:, :, 9] = np.ones((5, 5)) * player

    return board_state


def get_board_2D(board_state: np.array) -> np.array:
    """ Return the human readable board from the board_state

    Args:
        board_state (np.array): 5x5x10 array of the current state

    Returns:
        np.array: Human readable board
    """

    player = board_state[:, :, 9][0][0]

    # Two first plane always represent the current player's pawns
    # And the two next, the opponent player's pawns
    board_2D = (board_state[:, :, 0] + board_state[:, :, 1] * 2 +
                (board_state[:, :, 2] + board_state[:, :, 3] * 2) * -1) * player

    return board_2D


def init_game(deck: list) -> list:

    init_board_2D = np.array([[-1, -1, -2, -1, -1],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [1,  1,  2,  1,  1]])
    init_deck = random.sample(deck, 5)
    init_board_state = get_board_state(init_board_2D, init_deck, 1)

    # Board2D and cards don't need to be encoded, as they are included in board_state ?
    # root = [board_state, value, number_of_visit, [children]]
    root = [init_board_state, 0, 0, []]

    return root


root = init_game(deck)


# Dictionnary need to be transformed in numba.type.Dict() to be supported by Numba
def get_layer_codes(deck: list, cards_name: list) -> dict:

    layer_code = {}
    count = 0
    for card, name in zip(deck, cards_name):
        for i in range(5):
            for j in range(5):
                if card[i][j] != 0:
                    # (i - 2, j - 2) if we want to be consistent with the last version.
                    layer_code[name, (i - 2, j - 2)] = count
                    count += 1
    return layer_code


deck
cards_name = ["tiger", "dragon", "frog", "rabbit", "crab", "elephant", "goose",
              "rooster", "monkey", "mantis", "horse", "ox", "crane", "boar", "eel", "cobra"]


layer_code = get_layer_codes(deck, cards_name)
layer_decode = {v: k for k, v in layer_code.items()}


def get_legals_moves(board_state: np.array, policy: np.array) -> np.array:

    # Returns all the legal moves for a policy obtained with the neural network

    player_board = board_state[:, :, 0] + board_state[:, :, 1]
    player_card_1 = board_state[:, :, 4]
    player_card_2 = board_state[:, :, 5]

    possibles_moves = []  # how to get card name ?
    # list(zip(cards_name, deck))

    for card in [player_card_1, player_card_2]:
        for i in range(5):
            for j in range(5):
                if card[i][j] != 0:
                    possibles_moves.append("card_name", i - 2, j - 2)

    possible_policy = np.zeros((5, 5, 52))
    # taille 5 x 5 x 52, ou 52 represente 1 mouvement d'une carte
    # Si je te dis 38 je dois pouvoir identifier qu'il s'agit du mouvement ('ox', (0, 1))
    # 5 x 5 est l'endroit ou la pièce arrive.

    # We can't land on our pieces
    for k in range(possible_policy.shape[2]):
        possible_policy[:, :, k] = possible_policy[:, :, k] * (1 - player_board)
        
    # Then we normalize to [0, 1]
    possible_policy = possible_policy / np.sum(possible_policy)
    return possible_policy



# move : get new board state : swap card and , swap plane of piece and move the piece, and change player value


def move():
    pass


def get_value():
    pass


def get_children():
    pass


def get_legal_moves():
    pass

