import matplotlib.pyplot as plt
import numpy as np

class Card():
    
    def __init__(self, name, moves):
        self.name = name
        self.moves = moves
        self.map = self.maps()
        
    def maps(self):
        maps = np.zeros((5,5))
        # maps[2,2] = 2

        for move in self.moves:
            maps[2 + move[0]][2 + move[1]] += 1  
        return maps
        
    def plot(self):
        plt.imshow(self.map)
        plt.title(f"{self.name}")
        plt.show()

tiger = Card("tiger", [(-2, 0), (1, 0)])
dragon = Card("dragon", [(1, 1), (1, -1), (-1, 2), (-1, -2)])
frog = Card("frog", [(0, -2), (-1, -1), (1, 1)])
rabbit = Card("rabbit", [(1, -1), (-1, 1), (0, 2)])
crab = Card("crab", [(0, 2), (0, -2), (-1, 0)])
elephant = Card("elephant", [(0, 1), (-1, 1), (0, -1), (-1, -1)])
goose = Card("goose", [(0, -1), (-1, -1), (0, 1), (1, 1)])
rooster = Card("rooster",[(1, -1), (0, -1), (0, 1), (-1, 1)])
monkey = Card("monkey", [(-1, 1), (1, -1), (-1, -1), (1, 1)])
mantis = Card("mantis", [(1, 0), (-1, -1), (-1, 1)])
horse = Card("horse", [(1, 0), (0, -1), (-1, 0)])
ox = Card("ox", [(1, 0), (0, 1), (-1, 0)])
crane = Card("crane", [(1, -1), (1, 1), (-1, 0)])
boar = Card("boar", [(0, -1), (-1, 0), (0, 1)])
eel = Card("eel", [(1, -1), (-1, -1), (0, 1)])
cobra = Card("cobra", [(0, -1), (1, 1), (-1, 1)])

deck = [tiger, dragon, frog, rabbit, crab, elephant, goose,
    rooster, monkey, mantis, horse, ox, crane, boar, eel, cobra]


