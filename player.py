import numpy as np
import tensorflow as tf
from statistics import mean
import time

def play(model, env, games, ram_obs):
    try:
        scores_mean = 0
        scores = []
        observation = env.reset()
        action_total = 0
        for game in range(games):
            env.step(1)
            score = 0
            while True:
                action = np.argmax(model.predict(np.array([[observation[i] for i in ram_obs]])))
                if action_total > 0 and action_total % 100 == 0: 
                    action = 1
                    action_total = 0
                observation, reward , done, _ = env.step(action)
                env.render()
                action_total += 1
                score += reward
                # time.sleep(.01)
                if done: break
            env.reset()
            print(f"--> Game: {game}/{games}, S: {score} ")
            scores.append(score)
        env.close()
        scores_mean = mean(scores)
    except KeyboardInterrupt:
        print("[STOPED]")
    print(f"Games scores mean: {scores_mean}")
    return scores_mean