import tensorflow as tf
from tensorflow.keras.models import load_model
from Numpy_version.Train import self_play, load_data
from Numpy_version.Board import get_board_2D
tf.compat.v1.disable_eager_execution()

short_pseudo = load_model("models/short_pseudo.h5")

self_play(short_pseudo, "short_pseudo", nb_simulations=25, nb_games=1, max_move_per_game=150)


board_states, policies, values, terminal_values, nb_games= load_data("short_pseudo", 25)
nb_games

for value in values[-150:]:
    print(value)
    
import matplotlib.pyplot as plt 

for board, value in zip(board_states[-50:], values[-50:]):
    fig = plt.figure()
    plt.title(value)
    plt.imshow(get_board_2D(board))
    plt.show()