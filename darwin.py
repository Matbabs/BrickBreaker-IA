import neural_network, player
import neural_network as nn
import numpy as np
import time
import random
from enum import Enum

class GeneticAdmixture(Enum):
    MUTATION = 1,
    CROSSOVER = 2

def natural_selection(population, path, env, generations, games, ram_obs, mutate_man, mutate_prob, crossover_prob):
    # GENERATE INITIAL POPULATION (ADD SOME VARIATIONS)
    models = []
    for _ in range(population):
        model = nn.NeuralNetwork()
        model.load(path)
        model.genetic_weights_admixture(GeneticAdmixture.MUTATION, mutate_prob, magnitude = mutate_man)
        models.append(model)
    # FOR EACH GENERATION
    for generation in range(generations):
        # PLAY
        best_score = 0
        models_scores_mean = []
        for model in range(population):
            models_scores_mean.append(player.play(models[model], env, games, ram_obs, False, False, True))
            print(f"* Generation: {generation}, Model: {model}, Score: {models_scores_mean[-1]}")
        best_score = max(models_scores_mean)
        # NEW POPULATION
        new_models = []
        # SELECT THE 2 BEST
        for _ in range(2):
            best = np.argmax(models_scores_mean)
            models_scores_mean[best] = 0
            new_models.append(models[best])
        # CROSSOVER BETWEEN THE 2 BEST FOR THE FOLLOWING 2
        for two_next in range(2):
            other = np.argmax(models_scores_mean)
            models_scores_mean[other] = 0
            model = models[other]
            model.copy_dna_weights(new_models[two_next].model)
            model.copy_dna_bias(new_models[two_next].model)
            model.genetic_weights_admixture(GeneticAdmixture.CROSSOVER, crossover_prob, model = new_models[1 if two_next == 0 else 0].model)
            new_models.append(model)
        # REPRODUCE BEST
        for _ in range(2):
            new_models.append(new_models[0])
        # SAVE BEST MODEL
        new_path = path+"_"+"darwin"+"_"+str(generation)+"_"+str(best_score)
        print(f"--> Save model: {new_path}")
        new_models[0].save(new_path)
        # MUTATE POPULATION
        for candidate in range(2,len(new_models)):
            new_models[candidate].genetic_weights_admixture(GeneticAdmixture.MUTATION, mutate_prob, magnitude=mutate_man)
        # SAVE NEW POPULATION
        models = new_models