# -*- coding: utf-8 -*-

class SimpleTFTChampion(object):
    def __init__(self, preferred_position: int, team: int, level: int = 0):
        """
        Initialize a SimpleTFT Champion with preferred position, team, and level.

        :param preferred_position: An integer representing the champion's preferred position.
        :param team: An integer representing the team the champion belongs to.
        :param level: An integer representing the champion's level, default is 0.
        """
        if not all(isinstance(x, int) and x >= 0 for x in [preferred_position, team, level]):
            raise ValueError("preferred_position, team, and level must be non-negative integers")
            
        self.__preferred_position = preferred_position
        self.__team = team
        self.__level = level

    @property
    def preferred_position(self):
        return self.__preferred_position

    @property
    def team(self):
        return self.__team

    @property
    def level(self):
        return self.__level

    def set_preferred_position(self, position: int):
        if isinstance(position, int) and position >= 0:
            self.__preferred_position = position
        else:
            raise ValueError("Preferred position must be a non-negative integer")

    def set_team(self, team: int):
        if isinstance(team, int) and team >= 0:
            self.__team = team
        else:
            raise ValueError("Team must be a non-negative integer")

    def level_up(self):
        """
        Increment the champion's level by 1.
        """
        self.__level += 1

    def match(self, other):
        """
        Check if another champion is identical in terms of preferred position, team, and level.

        :param other: Another instance of SimpleTFTChampion to compare with.
        :return: Boolean, True if all attributes match, False otherwise.
        """
        if isinstance(other, self.__class__):
            return all((self.__preferred_position == other.__preferred_position,
                        self.__team == other.__team,
                        self.__level == other.__level))
        return False
