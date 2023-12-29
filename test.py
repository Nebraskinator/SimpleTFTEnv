# -*- coding: utf-8 -*-
from simpletft.tft import SimpleTFT
import numpy as np


if __name__ == "__main__":

    config = {'reward_structure' : 'power',
              'debug' : False}
    log_file_dir = "./"
    env = SimpleTFT(config)
    num_games = 100
    for i in range(num_games):
        log_file_path = log_file_dir + f"{i}.txt" if log_file_dir else log_file_dir
        obs, taking_actions, action_masks = env.reset(log_file_path)
        dones = {p: False for p in env.live_agents}
        while not all(dones.values()):
            action = {p: np.random.choice(np.flatnonzero(mask)) for p, mask in action_masks.items()}
            obs, rewards, taking_actions, dones, action_masks = env.step(action)

    print(f"successfully completed {num_games} games with random actions")
        
