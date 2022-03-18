from MCTS import MonteCarloTreeSearchNode
from NNet_architecture import create_model
from Deck import deck
from Board import Board

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


model = create_model()
# model = load_model("models/model1.h5")

new_board = Board(deck)
root = MonteCarloTreeSearchNode(model, new_board, prior=0)


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


def self_play(model, model_name, root, nb_simulations=200, nb_games=2, max_move_per_game=np.inf):

    games = [MonteCarloTreeSearchNode(model, Board(
        deck), prior=0) for i in range(nb_games)]
    game = 0
    i = 0

    while game < nb_games:

        board_states = []
        values = []
        policies = []

        node = games[game]
        # node = MonteCarloTreeSearchNode(model, Board(deck), prior=0)
        i = 0
        while not node.board.is_game_over() and i < max_move_per_game:
            node.simulate(nb_simulations)

            l = [(ind, node._number_of_visits)
                 for ind, node in node.children.items()]
            ind, action = max(l, key=lambda x: x[1])

            policy = np.zeros(5 * 5 * 52)
            total_visits = sum(
                [b._number_of_visits for b in node.children.values()])
            for a, b in node.children.items():
                policy[a] = b._number_of_visits/total_visits

            policies.append(policy)
            board_states.append(node.board.board_state)
            values.append(node.mean_value()[0][0])

            # nodes.append(node)
            node = node.children[ind]

            plt.imshow(node.board.board_2D)
            plt.show()
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


new_board = Board(deck)
root = MonteCarloTreeSearchNode(model, new_board, prior=0)

self_play(model, "model1", root, nb_simulations=200, nb_games=1)

model.save("models/model1.h5")


board_states, policies, values, old_nb_games = load_data("model1", 200)

board_states_2 = np.zeros((len(board_states), 5, 5, 10))
policies_2 = np.zeros((len(policies), 5, 5, 52))

for i, board in enumerate(board_states):

    board_states_2[i] = board
    policies_2[i] = policies[i].reshape((5, 5, 52))


model.fit(board_states_2, [policies_2, np.array(
    values)], batch_size=256, epochs=15)

pd.DataFrame(model.history.history).plot()