import numpy as np
import random
from Numpy_version.Deck import deck
import numba as nb

def get_board_state(board_2D: np.array, cards: list, player: int) -> np.array:
    """Return the board state of size 5x5x10

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

    player = board_state[0, 0, 9]

    # Two first plane always represent the current player's pawns
    # And the two next, the opponent player's pawns
    board_2D = (board_state[:, :, 0] + board_state[:, :, 1] * 2 +
                (board_state[:, :, 2] + board_state[:, :, 3] * 2) * -1) * player

    return board_2D


# Dictionnary need to be transformed in numba.type.Dict() to be supported by Numba
def get_layer_code(deck: list, cards_name: list) -> dict:

    layer_code = {}
    counter = 0
    for card, name in zip(deck, cards_name):
        for i in range(5):
            for j in range(5):
                if card[i][j] != 0:
                    # (i - 2, j - 2) if we want to be consistent with the last version.
                    layer_code[name, (i - 2, j - 2)] = counter
                    counter += 1
    return layer_code


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

    try:
        player_card_1_name = [
            ele[1] for ele in list_cards if np.array_equal(ele[0], player_card_1)][0]
        player_card_2_name = [
            ele[1] for ele in list_cards if np.array_equal(ele[0], player_card_2)][0]
    except:
        print("player_card_1", player_card_1)
        print("player_card_2", player_card_2)
        
        print(list_cards)

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


def move(board_state: np.array, action: int) -> np.array:

    # find where to land on the board
    plane = action % 52
    column = action // 52 % 5
    line = action // 52 // 5

    # Card and move of the card played.
    card_name, (x, y) = layer_decode[plane]
    
    # Find the position of the piece to move
    piece_x, piece_y = (line - x, column - y)

    # Create new state
    next_state = board_state.copy()

    # Find the played piece (pawn or king)
    if next_state[piece_x, piece_y, 0] == 1:
        next_state[line, column, 0] = 1
        next_state[piece_x, piece_y, 0] = 0

    else:
        next_state[line, column, 1] = 1
        next_state[piece_x, piece_y, 1] = 0
    
    # Kill the enemy piece, if there is one on the place we land
    next_state[line, column, 3] = 0
    next_state[line, column, 4] = 0

    # Turn the board (4 first planes) to face the new player
    next_state[:, :, 0:4] = np.rot90(np.rot90(next_state[:, :, 0:4]))

    # Swap planes of the board  
    b1, b2 = next_state[:, :, 0].copy(), next_state[:, :, 1].copy()
    next_state[:, :, 0] = next_state[:, :, 2]
    next_state[:, :, 1] = next_state[:, :, 3]
    next_state[:, :, 2], next_state[:, :, 3] = b1, b2
    
    # Find the played card associated with the card name
    played_card = [card for card, name in list_cards if name == card_name][0]

    # played card is in J1 hand, so it's the plane 4 or 5
    if np.array_equal(board_state[:, :, 4], played_card):
        next_state[:, :, 4] = board_state[:, :, 8]
        next_state[:, :, 8] = board_state[:, :, 4]

    else:
        next_state[:, :, 5] = board_state[:, :, 8]
        next_state[:, :, 8] = board_state[:, :, 5]

    # Swap planes of the cards
    c1, c2 = next_state[:, :, 4].copy(), next_state[:, :, 5].copy()
    next_state[:, :, 4] = next_state[:, :, 6]
    next_state[:, :, 5] = next_state[:, :, 7]
    next_state[:, :, 6], next_state[:, :, 7] = c1, c2

    # Change player
    next_state[:, :, 9] *= -1

    return next_state


def is_game_over(board_state: np.array) -> bool:

    # if a king managed to go to the other side
    if board_state[0, 2, 1] == 1:
        return True
    
    # if the enemy king is alive return False
    for i in range(5):
        for j in range(5):
            if board_state[i, j, 3] == 1:
                return False
            
    return True


def get_reward_for_player(board_state):

    if is_game_over(board_state):
        return -1
    else:
        return None

cards_name = ["tiger", "dragon", "frog", "rabbit", "crab", "elephant", "goose",
                "rooster", "monkey", "mantis", "horse", "ox", "crane", "boar", "eel", "cobra"]

# Transform this 3 variables to nb.Typed.Dict() ?
layer_code = get_layer_code(deck, cards_name)
layer_decode = {v: k for k, v in layer_code.items()}
list_cards = list(zip(deck, cards_name))
