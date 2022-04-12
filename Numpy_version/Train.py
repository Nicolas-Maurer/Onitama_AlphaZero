from Numpy_version.MCTS import *
from Numpy_version.Deck import deck
import numpy as np
import pickle
import os
import pyautogui as p

def load_data(model_name, nb_simulations):

    with open(f'self_games/{model_name}/{nb_simulations}_simu_board_states.pickle', 'rb') as handle:
        board_states = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_policies.pickle', 'rb') as handle:
        policies = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_values.pickle', 'rb') as handle:
        values = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_terminal_values.pickle', 'rb') as handle:
        terminal_values = pickle.load(handle)
    with open(f'self_games/{model_name}/{nb_simulations}_simu_nb_games.pickle', 'rb') as handle:
        nb_games = pickle.load(handle)

    return board_states, policies, values, terminal_values, nb_games


def self_play(model, model_name, nb_simulations=200, nb_games=2, max_move_per_game=np.inf):
    """ Generate game and save them

    Args:
        model (_type_): _description_
        model_name (str): name of the model for saving games
        nb_simulations (int, optional): Number of simulations in MCTS Defaults to 200.
        nb_games (int, optional): Number of games to simulate
        max_move_per_game (_type_, optional): max number of move per game Defaults to np.inf.
    """

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

            # Select the move to play
            action_index = select_action(node, i)
            # child_visits = [child[4] for child in node[-1]]
            # action_index = np.argmax(child_visits)

            # MCTS policy
            policy = np.zeros(5 * 5 * 52)
            child_visits = [child[4] for child in node[-1]]
            total_visits = sum(child_visits)
            for child in node[-1]:
                policy[child[1]] = child[4]/total_visits

            policies.append(policy)
            board_states.append(node[0])
            values.append(get_value(node))

            # Go to the next child
            node = node[-1][action_index]

            # plt.imshow(get_board_2D(node[0]))
            # plt.show()

            i += 1
            print("-----------", i)
            # p.moveTo(500 + 400*(i%2), 500 + 400*(i%2), duration = 0.5)
            # p.press("esc")
            
        # Backpropagation of the terminal value  
        if i < max_move_per_game:
            player = node[0][0, 0, 9]
            terminal_values = [-1*player if _ % 2 == 0 else 1*player for _ in range(i)][::-1]
            print(terminal_values)
            print(player)
                
        else:
            terminal_values = [0 for _ in range(i)]

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

            old_board_states, old_policies, old_values, old_terminal_values, old_nb_games = load_data(
                model_name, nb_simulations)

            board_states = old_board_states + board_states
            policies = old_policies + policies
            values = old_values + values
            terminal_values = old_terminal_values + terminal_values

        game_played = old_nb_games + 1
        with open(f'self_games/{model_name}/{nb_simulations}_simu_board_states.pickle', 'wb') as handle:
            pickle.dump(board_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_policies.pickle', 'wb') as handle:
            pickle.dump(policies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_values.pickle', 'wb') as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_terminal_values.pickle', 'wb') as handle:
            pickle.dump(terminal_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'self_games/{model_name}/{nb_simulations}_simu_nb_games.pickle', 'wb') as handle:
            pickle.dump(game_played, handle, protocol=pickle.HIGHEST_PROTOCOL)


def select_action(state: list, count: int) -> list:
    
    visit_counts = [(child[4], child[1]) for child in state[-1]]
    
    if count <= 30:
        action_index = softmax_sample(visit_counts)
    else:
        action_index = visit_counts.index(max(visit_counts))
    return action_index


def softmax_sample(visit_counts :list[int, int]) -> int:
    
    visits = [x[0] for x in visit_counts]
    e_x = np.exp(visits - np.max(visits))
    e_x = np.cumsum(e_x / e_x.sum())
    p = np.random.uniform()
    
    for i, proba in enumerate(e_x):
        if proba >= p:
            return i