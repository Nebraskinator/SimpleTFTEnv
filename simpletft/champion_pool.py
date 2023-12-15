# -*- coding: utf-8 -*-
from .champion import SimpleTFTChampion
import numpy as np

class SimpleTFTChampionPool(object):
    def __init__(self, 
                 champ_copies: int, 
                 num_teams: int, 
                 num_positions: int):
        self.champions = []
        for team in range(num_teams):
            for pos in range(num_positions):
                for _ in range(champ_copies):
                    self.champions.append(SimpleTFTChampion(pos, team, 0))
    def sample(self, num: int=1) -> list:
        if num > len(self.champions):
            raise Exception("Cannot sample {} champions from a pool of size {}".format(num, len(self.champions)))
        sample = []
        for _ in range(num):
            champ = np.random.choice(self.champions)
            sample.append(champ)
            self.champions.remove(champ)
        return sample
            
    def add(self, champ: SimpleTFTChampion):
        if not champ.level:
            self.champions.append(champ)
        else:
            for i in range(2**champ.level):
                self.champions.append(SimpleTFTChampion(champ.preferred_position, champ.team))

