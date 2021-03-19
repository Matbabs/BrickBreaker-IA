import gym
import neural_network as nn
import trainer, player, darwin
import sys

# ENVIRONEMENT
ENV = gym.make("Breakout-ram-v0")
ENV.reset()
RAM_Y_BALL = 101
RAM_X_BALL = 99
RAM_X_AGENT = 72
RAM_OBS = [RAM_Y_BALL, RAM_X_BALL, RAM_X_AGENT]

# TRAIN
TRAIN_GAMES = 200
SCORE_REQUIREMENTS = 100

# PLAY
PLAY_GAMES = 50

# DARWIN
POPULATION = 6
CREATE_GENERATIONS = 20
PLAY_GAMES_GEN = 2
MUTATE_MAGNITUDE = 1e-06
MUTATE_PROBABILITY = 3
CROSSOVER_PROBABILITY = 3

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        model = nn.NeuralNetwork()
        if sys.argv[1] == "train":
            values, targets = trainer.get_train_data(ENV, TRAIN_GAMES, SCORE_REQUIREMENTS, RAM_OBS)
            model.train(values, targets)
            model.save(sys.argv[2])
        elif sys.argv[1] == "play":
            model.load(sys.argv[2])
            player.play(model, ENV, PLAY_GAMES, RAM_OBS, False)
        elif sys.argv[1] == "darwin":
            darwin.natural_selection(POPULATION, sys.argv[2], ENV, CREATE_GENERATIONS, PLAY_GAMES_GEN, RAM_OBS, MUTATE_MAGNITUDE, MUTATE_PROBABILITY, CROSSOVER_PROBABILITY)
    else:
        print("[USAGE] python3 breaker.py <train|play> <model>")