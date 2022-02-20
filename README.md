# Onitama_AlphaZero
Implementation of the AlphaZero algorithm for the game Onitama

![image](https://user-images.githubusercontent.com/62259863/151253760-b5abe21b-3f3d-45de-b0b0-4b97fea8cfbf.png)

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

We will represent the board by a 5x5 matrix. 
In the input, we need to have the information of which piece is where, as well as the card in the player hands and the one on the side

The output dimension correspond to all the possibles moves. 
Input_dim = 5 x 5 x 10   
Output_dim = 5 x 5 x 52  



# Monte Carlo Tree Search (MCTS)
In the paper, they 800 simulations for each MCTS, and the trees were kept for the whole game.
