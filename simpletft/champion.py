# -*- coding: utf-8 -*-

class SimpleTFTChampion(object):
    def __init__(self, 
                 preferred_position: int, 
                 team: int, 
                 level: int=0):
        self.preferred_position = preferred_position
        self.team = team
        self.level = level
        
    def match(self, other):
        if isinstance(other, self.__class__):
            return all([self.preferred_position == other.preferred_position,
                        self.team == other.team,
                        self.level == other.level])
        return False
    
    def level_up(self):
        self.level += 1