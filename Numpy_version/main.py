import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from Numpy_version.Train import self_play, load_data
from Numpy_version.Board import get_board_2D
tf.compat.v1.disable_eager_execution()

short_pseudo = load_model("models/short_pseudo.h5")

model_name = "short_pseudo"
nb_simulations = 50

self_play(short_pseudo, model_name, nb_simulations=nb_simulations, nb_games=200, max_move_per_game=150)

board_states, policies, values, terminal_values, nb_games = load_data(model_name, 50)
nb_games
len(board_states)
len(policies)
len(values)
len(terminal_values)

self_play(short_pseudo, model_name, nb_simulations=25, nb_games=1, max_move_per_game=150)
board_states, policies, values, terminal_values, nb_games = load_data(model_name, 25)

# Game visualization with cards
for i, (board, value, t) in enumerate(zip(board_states[-5:], values[-5:], terminal_values[-5:])):
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    subfigs.flat[0].subplots(1, 1).imshow(get_board_2D(board))
    axs = subfigs.flat[1].subplots(2, 1)
    axs[0].imshow(board[:, :, 4])
    axs[1].imshow(board[:, :, 5])
    plt.title((f"Move {i}", round(float(value), 3), t))
    plt.show()