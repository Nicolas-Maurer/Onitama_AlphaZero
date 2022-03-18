# https://joshvarty.github.io/AlphaZero/

import numpy as np
from collections import defaultdict

class MonteCarloTreeSearchNode():
    
    def __init__(self, model, board, prior, parent=None):
        self.model = model
        self.board = board
        self.prior = prior
        self.parent = parent
              
        self.children = {}
        self._number_of_visits = 0
        self.value_sum = 0

        self._results = defaultdict(int)
        self._results[0] = 0
        self._results[1] = 0
        self._results[-1] = 0

            
    def expand(self, possible_policy):
        """ Expand the node with all children with a positive probability, the policy is obtained by the nn"""
                
        for i, proba in enumerate(possible_policy):
            if proba != 0:
                
                next_board = self.board.move(i) 
                self.children[i] = MonteCarloTreeSearchNode(model=self.model, board=next_board, prior=proba, parent=self)
                

    def simulate(self, nb_simulation, model=None):
        """ Simulate i path"""
        
        if model == None:
            model = self.model
        
        for _ in range(nb_simulation):
            if _%50 == 0:
                print(_)
                   
            node_to_expand = self
                                   
            search_path = [node_to_expand]
            
            # select the node
            while node_to_expand.children:  
                node_to_expand = node_to_expand.best_child()
                search_path.append(node_to_expand)
            
            value = node_to_expand.board.get_reward_for_player() 

            if value is None:
                # if the game has not ended we expand 

                # policy, value = node_to_expand.model.predict(node_to_expand.board.board_state.reshape((1, 5, 5, 10)))
                policy, value = model.predict(node_to_expand.board.board_state.reshape((1, 5, 5, 10)))
                possible_policy = node_to_expand.board.get_legal_moves(policy[0]).flatten()
                
                node_to_expand.expand(possible_policy)
            
            self.backpropagate(search_path, value, node_to_expand.board.player)
                
        return self
    
    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.board.player == to_play else -value
    
            node._number_of_visits += 1


    def best_child(self, c_param=0.8):
        """return child that maximize UCB"""

        # C_param is the exploration rate it's supposed to grow slowly with search time
        # Mean action_value + C_param * Prior * sqrt(parent visit count) / (1 + visit count)
        # The value of the child is from the perspective of the opposing player

        best_score = -np.inf
        best_action = None
                
        for action, child in self.children.items():
            score = -child.mean_value() + c_param * child.prior * np.sqrt(child.parent._number_of_visits) / (child._number_of_visits + 1)

            if score > best_score:
                best_score = score
                best_action = action
                
        return self.children[best_action]
    
    
    def mean_value(self):
        if self._number_of_visits == 0:
            return 0
        return self.value_sum / self._number_of_visits


    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.board.__str__(), prior, self._number_of_visits, self.mean_value())