# Source for the architecture
"""
https://arxiv.org/pdf/1712.01815.pdf

https://www.chessprogramming.org/AlphaZero#:~:text=AlphaZero%20evaluates%20positions%20using%20non,policy)%20and%20a%20position%20evaluation.

http://www.talkchess.com/forum3/viewtopic.php?f=2&t=69175&start=93
 
https://www.chessprogramming.org/Neural_Networks#Residual
"""

# # Input size : 5 x 5 x 10
# - board size : 5 x 5
# - P1 unique pieces : 2
# - P2 unique pieces : 2
# - P1 cards moves : 2
# - P2 cards moves : 2
# - Remaining card
# - Colour : 1

# P1 is the current player, P2 his opponent, the board is turned towards the current player.

# # Output size 5 x 5 x 52
# - board size : 5 x 5
# - 52 possibles moves, 1 for each directions for each cards

# The 52 planes can be seen like this ("Tiger",(-2,0))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def create_model():

    # input_block
    input_block = layers.Input(shape=(5, 5, 10))

    # convolutionnal_layer
    x = layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="linear")(input_block)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 19 Residual blocks with a skip connection
    for _ in range(19):
        y = layers.Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", strides=1, activation="linear")(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", strides=1, activation="linear")(y)
        y = layers.BatchNormalization()(y)

        x = layers.Add()([x, y])
        x = layers.LeakyReLU()(x)

    # policy_head with a final convolution of 52 filters
    policy_head = layers.Conv2D(filters=256, kernel_size=(
        1, 1), padding="same", activation="linear")(x)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.LeakyReLU()(policy_head)
    policy_head = layers.Conv2D(filters=52, kernel_size=(
        1, 1), padding="same", activation="linear")(policy_head)
    policy_head = layers.Softmax()(policy_head)

    # value_head
    value_head = layers.Conv2D(filters=1, kernel_size=(
        1, 1), padding="same", strides=1, activation="linear")(x)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.LeakyReLU()(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256, activation="linear")(value_head)
    value_head = layers.LeakyReLU()(value_head)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.Dense(1, activation="tanh",
                              name="value_head")(value_head)

    model = Model(inputs=[input_block], outputs=[policy_head, value_head])
    model.compile(loss=['categorical_crossentropy',
                  'mean_squared_error'], optimizer="Adam")
    return model
