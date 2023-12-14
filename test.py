# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:01:33 2023

@author: ruhe
"""

from env import SimpleTFT
import numpy as np

env = SimpleTFT()

obs, taking_actions, dones, action_masks = env.reset()
while not all(dones.values()):
    action = {p: np.random.choice(np.flatnonzero(mask)) for p, mask in action_masks.items()}
    obs, rewards, taking_actions, dones, action_masks = env.step(action)