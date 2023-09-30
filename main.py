from GameSimulator import GameSimulator
import Environment
import numpy as np

np.random.seed(2)
env = Environment.RandomMG(S=100, A=6, B=6, gamma=0.95)
#env = Environment.MatchingPennies()
game = GameSimulator(env=env, T=1000)
game.simulate_full_info()
