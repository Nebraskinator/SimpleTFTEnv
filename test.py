# -*- coding: utf-8 -*-
from simpletft.tft import SimpleTFT
import numpy as np


if __name__ == "__main__":


    env = SimpleTFT()
    
    for i in range(1000):
        obs, taking_actions, action_masks = env.reset()
        dones = {p: False for p in env.live_agents}
        while not all(dones.values()):
            action = {p: np.random.choice(np.flatnonzero(mask)) for p, mask in action_masks.items()}
            obs, rewards, taking_actions, dones, action_masks = env.step(action)

    print("successfully completed 1000 games with random actions")
        
