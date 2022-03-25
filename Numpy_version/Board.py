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


# Transform this 3 variables to nb.Typed.Dict() ?
layer_code = get_layer_codes(deck, cards_name)
layer_decode = {v: k for k, v in layer_code.items()}
list_cards = list(zip(deck, cards_name))

# This will help get_legals_moves() to run faster with dict


def get_legals_moves(board_state: np.array, policy: np.array) -> np.array:
    """ Returns all the legal moves of the policy from the state

    Args:
        board_state (np.array): State of the board : 5x5x10
        policy (np.array): Probability of next moves given by the NN : 5x5x52

    Returns:
        np.array: Filtered probability of the next moves : 5x5x52
    """

    # list_cards and layer_code are global variable

    player_board = board_state[:, :, 0] + board_state[:, :, 1]
    player_card_1 = board_state[:, :, 4]
    player_card_2 = board_state[:, :, 5]

    player_card_1_name = [
        ele[1] for ele in list_cards if np.array_equal(ele[0], player_card_1)][0]
    player_card_2_name = [
        ele[1] for ele in list_cards if np.array_equal(ele[0], player_card_2)][0]

    # All 'possible' moves
    possibles_moves = []
    for card, name in [(player_card_1, player_card_1_name), (player_card_2, player_card_2_name)]:
        for i in range(5):
            for j in range(5):
                if card[i][j] != 0:
                    possibles_moves.append((name, (i - 2, j - 2)))

    # Is there another, and better way to get the name of the cards?

    possible_policy = np.zeros((5, 5, 52))

    # All feasible moves
    for card, (line, column) in possibles_moves:
        for i in range(5):
            for j in range(5):
                if player_board[i][j] != 0:
                    if (0 <= i + line < 5) and (0 <= j + column < 5):
                        possible_policy[i + line, j + column, layer_code[card, (line, column)]] = \
                            policy[i + line, j + column,
                                   layer_code[card, (line, column)]]

    # We can't land on our pieces
    for k in range(possible_policy.shape[2]):
        possible_policy[:, :, k] = possible_policy[:, :, k] * \
            (1 - player_board)

    # Then we normalize to [0, 1]
    possible_policy = possible_policy / np.sum(possible_policy)
    return possible_policy


board_state = root[0]
policy = np.random.uniform(0, 1, size=(5, 5, 52))
possible_moves = get_legals_moves(board_state, policy)

print("Number of possible moves from this state :", len(
    [i for i in possible_moves.flatten() if i != 0]))

player_card_1 = board_state[:, :, 4]
player_card_2 = board_state[:, :, 5]
print(player_card_1)
print(player_card_2)


def move(state: np.array, action: int) -> np.array:

    # find where to land on the board
    plane = action % 52
    column = action // 52 % 5
    line = action // 52 // 5

    card, (x, y) = layer_decode[plane]
    # Layer_decode can be a list, and layer_decode[plane] return the nth element of it.
    
    # Find the position of the piece to move
    piece_x, piece_y = (line - x, column - y)
    
    # Create new state
    next_state = state.copy()

    # Swap the card played with the remaining.
    # How to find the played card ? by name with a dict ?
    # state[:, :, 8] is always the remaining card
    next_state[:, :, "carte jouée"] = state[:, :, 8]
    next_state[:, :, 8] = state[:, :, "carte jouée"]

    # Swap planes of the cards
    c1, c2 = next_state[:, :, 4], next_state[:, :, 5]
    next_state[:, :, 4] = next_state[:, :, 6]
    next_state[:, :, 5] = next_state[:, :, 7]
    next_state[:, :, 6], next_state[:, :, 7] = c1, c2

    # Find the piece (pawn or king)
    if next_state[piece_x, piece_y, 0] == 1:
        next_state[line, column , 0] = 1
        next_state[piece_x, piece_y, 0] = 1
        
    else:
        next_state[line, column , 1] = 1
        next_state[piece_x, piece_y, 1] = 1
        
    # Turn the board (4 first planes) to face the new player
    board_state[:, :, 0:5] = np.rot90(np.rot90(board_state[:, :, 0:5]))

    # Swap planes of the board
    b1, b2 = next_state[:, :, 0], next_state[:, :, 1]
    next_state[:, :, 0] = next_state[:, :, 2]
    next_state[:, :, 1] = next_state[:, :, 3]
    next_state[:, :, 2], next_state[:, :, 3] = b1, b2

    # Change player
    next_state[:, :, 9] *= -1

    return next_state


def is_game_over(state: np.array) -> bool:
    pass


def get_value():
    pass


action = np.argmax(possible_moves)
move(board_state, action)