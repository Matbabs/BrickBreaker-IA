import numpy as np
import tensorflow as tf
from statistics import mean
import time

def play(model, env, games, ram_obs, rendering = True, printer = True, darwin = False):
    scores_mean = 0
    scores = []
    for game in range(games):
        observation = env.reset()
        env.step(1)
        score = 0
        ball_y = None; prev_ball_y = None
        while True:
            action = np.argmax(model.predict(np.array([[observation[i] for i in ram_obs]])))
            observation, reward , done, info = env.step(action)
            ball_y = observation[ram_obs[0]]
            if ball_y == 0 and prev_ball_y == ball_y: env.step(1)
            prev_ball_y = ball_y
            score += reward
            if rendering: env.render()
            if done or (darwin and info['ale.lives'] == 4): break
        env.reset()
        if printer: print(f"--> Game: {game+1}/{games}, Score: {score} ")
        scores.append(score)
    env.close()
    scores_mean = mean(scores)
    if printer: print(f"--> Games scores mean: {scores_mean}")
    return scores_mean