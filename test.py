# -*- coding: utf-8 -*-
from simpletft.tft import SimpleTFT
import numpy as np

env = SimpleTFT('./log.txt')

for i in range(10000):
    obs, taking_actions, dones, action_masks = env.reset()
    while not all(dones.values()):
        action = {p: np.random.choice(np.flatnonzero(mask)) for p, mask in action_masks.items()}
        obs, rewards, taking_actions, dones, action_masks = env.step(action)
        
