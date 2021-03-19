import numpy as np
import tensorflow as tf
import neural_network as nn

def get_train_data(env, games, score_requirement, ram_obs):
    values = []
    targets = []
    for game in range(games):
        score = 0
        prev_observation = []
        memory = []
        done = False
        observation = []
        while not done:
            action = 1
            if len(observation) > 0:
                if observation[ram_obs[1]] < observation[ram_obs[2]] : action = 3
                elif observation[ram_obs[1]] > observation[ram_obs[2]] + 16: action = 2
            observation, reward, done, _ = env.step(action)
            if len(prev_observation) > 0 :
                memory.append([[prev_observation[i] for i in ram_obs], action])
            prev_observation = observation
            score += reward
            if done: break
        if score >= score_requirement:
            print(f"--> Game: {game}/{games}, Score: {score} ")
            for mem in memory:
                prev_action = []
                for k in range(4):
                    prev_action.append(1 if k == mem[1] else 0)
                values.append(mem[0])
                targets.append(prev_action)
        env.reset()
    print(f"--> Data: {len(values)}")
    np_values = np.array(values)
    np_targets = np.array(targets)
    return np_values, np_targets