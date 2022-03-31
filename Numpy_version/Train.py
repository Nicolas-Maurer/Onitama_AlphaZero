from NNet_architecture import create_model
from Numpy_version.MCTS import *
from Numpy_version.Deck import deck
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def load_data(model_name, nb_simulations):

    with open(f'self_games/{model_name}/{nb_simulations}_simu_board_states.pickle', 'rb') as handle:
        board_states = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_policies.pickle', 'rb') as handle:
        policies = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_values.pickle', 'rb') as handle:
        values = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_nb_games.pickle', 'rb') as handle:
        nb_games = pickle.load(handle)

    return board_states, policies, values, nb_games


def self_play(model, model_name, nb_simulations=200, nb_games=2, max_move_per_game=np.inf):
    
    games = [init_game(deck) for _ in range(nb_games)]
    game = 0
    
    while game < nb_games:
        board_states = []
        values = []
        policies = []

        node = games[game]
        i = 0
        while not is_game_over(node[0]) and i < max_move_per_game:
            simulate(node, nb_simulations, model)
            
            # Find the best move, i.e the most visited
            child_visits = [child[4] for child in node[-1]]
            index = np.argmax(child_visits)

            # MCTS policy
            policy = np.zeros(5 * 5 * 52)
            total_visits = sum(child_visits)
            for child in node[-1]:
                policy[child[1]] = child[4]/total_visits

            policies.append(policy)
            board_states.append(node[0])
            values.append(get_value(node))

            # Go to the next child
            node = node[-1][index]

            # plt.imshow(get_board_2D(node[0]))
            # plt.show()

            i += 1
            print("-----------", i)
        
        game += 1
        print("-----------------------------------------",
              game, "-----------------------------------------")
        
        
        if not os.path.exists(f"self_games"):
            os.mkdir(f"self_games")

        if not os.path.exists(f"self_games/{model_name}"):
            os.mkdir(f"self_games/{model_name}")

        old_nb_games = 0
        # Add the new generated game, policies, and values to old ones.
        if os.path.exists(f'self_games/{model_name}/{nb_simulations}_simu_board_states.pickle'):

            old_board_states, old_policies, old_values, old_nb_games = load_data(
                model_name, nb_simulations)

            board_states = old_board_states + board_states
            policies = old_policies + policies
            values = old_values + values

        game_played = old_nb_games + 1
        with open(f'self_games/{model_name}/{nb_simulations}_simu_board_states.pickle', 'wb') as handle:
            pickle.dump(board_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_policies.pickle', 'wb') as handle:
            pickle.dump(policies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_values.pickle', 'wb') as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_nb_games.pickle', 'wb') as handle:
            pickle.dump(game_played, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

model = create_model()
# model = load_model("models/model1.h5")

self_play(model, "test_numpy", 400, 10, max_move_per_game=150)


board_states, policies, values, nb_games = load_data("test_numpy", 400)


