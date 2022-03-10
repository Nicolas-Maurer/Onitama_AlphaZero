# Onitama_AlphaZero
Implementation of the AlphaZero algorithm for the game Onitama

![image](https://user-images.githubusercontent.com/62259863/151253760-b5abe21b-3f3d-45de-b0b0-4b97fea8cfbf.png)



For more details about this project, feel free to read my articles !   
- Part 1: https://medium.com/@nicolasmaurer/part-1-alphazero-implementation-for-the-game-onitama-370afb1259e6
- Part 2: Soon
- Part 3: Soon

# Onitama 
Onitama is a 2-player board game of size 5x5. 
Each player starts the game with 5 pieces: 4 pawn and a "Master pawn", and with 2 cards, another card is added near the board. 
The goal of the game is to kill the opponent's king, or to reach your opponent's king starting point.

The player can move their pieces according to the 2 cards in front of them, once a move from a card is played, the card is exchanged with the one near the board. 

# AlphaZero
My goal is to implement the AlphaZero algorithm for Onitama; following the amazing paper  
"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (silver et al. 2017)   
https://arxiv.org/pdf/1712.01815.pdf  


# Neural Network

The main goal of the neural network is to fit the value and policy obtained by the Monte Carlo Tree Search (MCTS).  
The neural network used by Alpha Zero is composed of an input block, some convolutionals blocs, 19 Residuals blocks, and 2 heads: one for the policy, one for the value.  
For Onitama, we can use the same global architecture, the only things we need to change are the input and output dimension.  

## Inputs
We will represent the board by a 5x5 matrix. 

In the input, we need to have the information of which piece is where, as well as the card in the player hands and the one on the side
The board is represented by a plane of size 5x5, and 10 planes, each plane are represented like this (in this order)
- Current player's pawns
- Current player's Master pawn
- Opponent's paws
- Opponent's Master pawn
- Moves on the current player's card 1
- Moves on the player's card 2
- Moves on the opponent's player's card 1
- Moves on the opponent's player's card 2
- Moves on the remaining card
- Matrix composed of 1 if the current player is the first player, else a matrix composed of -1

So, the input is represented by a 5 x 5 x 10 tensor. 

## Output
The output dimension correspond to all the possibles moves.  
To encode that we need a 5x5 matrix for all the case on the board, and plane for every move. 
There is 16 cards, with a certain number of possible moves in each, in total there is 52 differents moves.
Some move can be played by different cards but since we need to swap the card play with the one of the board we can't regroup them. 

So, the output is represented by a 5 x 5 x 52 tensor. 

## Loss
To be able to train a NN with two head we need a custom loss function. As mentionned in their paper, thye used a loss "that sums over mean-squared error and cross-entropy losses respectively"


# Monte Carlo Tree Search (MCTS)
In the paper, they 800 simulations for each MCTS, and the trees were kept for the whole game.

# Sources
[1] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (Silver et al. 2017)  
[2] Acquisition of Chess Knowledge in AlphaZero (McGrath et al. 2021)  
[3] Deep Residual Learning for Image Recognition (He et al. 2015)  
