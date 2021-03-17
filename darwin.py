import neural_network, player
import time
import random

def natural_selection(models, env, generations, games, ram_obs):

    models[0].randomize_weights(100000)

    for generation in range(generations):
        models_scores_mean = []
        for model in range(len(models)):
            for _ in range(games):
                models_scores_mean.append(player.play(models[model], env, games, ram_obs, False, False, True))
                print(f"--> Generation: {generation}, Model: {model}, Score: {models_scores_mean[-1]}")
