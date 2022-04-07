from copy import deepcopy
from tensorflow.keras.models import load_model
from Numpy_version.Board import *
from Numpy_version.MCTS import init_game, simulate, get_best_child
from Numpy_version.Deck import deck
import matplotlib.pyplot as plt


def static_init_game(deck: list) -> list:

    init_board_2D = np.array([[-1, -1, -2, -1, -1],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0],
                              [1,  1,  2,  1,  1]])
    init_deck = deck
    init_board_state = get_board_state(init_board_2D, init_deck, 1)
    # root = [board_state, action, prior, value, number_of_visit, [children]]
    root = [init_board_state, None, 0, 0, 0, []]

    return root



def arena(model1, model2, nb_simulations, max_number_of_move=150):
    
    nb_moves = 0
    
    root = static_init_game(deck[0:5])
        
    while not (is_game_over(root[0]) or nb_moves > max_number_of_move):
        
        player = root[0][0, 0, 9]
        nb_moves += 1
        
        if player == 1:
            
            root_player_1 = deepcopy(root)
            
            # If the root has not child 
            if not root_player_1[-1]:
                simulate(root_player_1, 1, model1)
            
            if nb_simulations:
                simulate(root_player_1, nb_simulations, model1)

            # Find the best child
            best_child = get_best_child(root_player_1)
            index = best_child[1]
            
            # Move to the next child
            if not root[-1]:
                simulate(root, 1, model1)
            
            # root = [child for child in root[-1] if child[1] == index][0]
            
            # Only used to play against random model
            root = root[-1][np.random.randint(len(root[-1]))]
        
        else:
            
            root_player_2 = deepcopy(root)
            
            # If the root has not child 
            if not root_player_2[-1]:
                simulate(root_player_2, 1, model2)
            
            if nb_simulations:
                simulate(root_player_2, nb_simulations, model2)

            # Find the best child
            best_child = get_best_child(root_player_2)
            index = best_child[1]
            
            # Move to the next child
            if not root[-1]:
                simulate(root, 1, model2)
            root = [child for child in root[-1] if child[1] == index][0]
        
        nb_moves += 1
        
        # plt.imshow(get_board_2D(root[0]))
        # plt.show()
        
    if nb_moves >= max_number_of_move:
        print("Tie ! Too much move have been played")
        return 0
    
    else:
        print(f"Player {player} won !")
        return player

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

model1 = load_model("models/short_model.h5")
model2 = load_model("models/short_model_2.h5")


arena(model1, model2, 1, 150)

victories = {0: 0,
             1: 0,
            -1: 0}

for i in range(100):
    victories[arena(model1, model2, 1, 150)] += 1
    

victories