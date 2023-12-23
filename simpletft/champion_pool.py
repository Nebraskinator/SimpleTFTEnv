# -*- coding: utf-8 -*-
from .champion import SimpleTFTChampion
import numpy as np

class SimpleTFTChampionPool(object):
    def __init__(self, 
                 champ_copies: int, 
                 num_teams: int, 
                 num_positions: int):
        """
        Initialize the Champion Pool with a specified number of copies, teams, and positions.

        :param champ_copies: Number of copies for each champion.
        :param num_teams: Number of teams in the pool.
        :param num_positions: Number of different positions in the pool.
        """
        self.__champions = [SimpleTFTChampion(pos, team, 0) 
                            for team in range(num_teams) 
                            for pos in range(num_positions) 
                            for _ in range(champ_copies)]

    def sample(self, num: int = 1) -> list:
        """
        Sample a specified number of champions from the pool.

        :param num: Number of champions to sample.
        :return: A list of sampled SimpleTFTChampion instances.
        """
        if num > len(self.__champions):
            raise Exception(f"Cannot sample {num} champions from a pool of size {len(self.__champions)}")

        indices = np.random.choice(len(self.__champions), size=num, replace=False)
        sample = [self.__champions[idx] for idx in indices]
        for idx in sorted(indices, reverse=True):
            del self.__champions[idx]

        return sample

    def add(self, champ: SimpleTFTChampion):
        """
        Add a champion back to the pool. If the champion is leveled up, 
        decompose it into base level champions.

        :param champ: The SimpleTFTChampion instance to be added.
        """
        if not champ.level:
            self.__champions.append(champ)
        else:
            self.__champions.extend(SimpleTFTChampion(champ.preferred_position, champ.team, 0)
                                    for _ in range(2 ** champ.level))

