# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:55:50 2023

@author: ruhe
"""

import numpy as np
from collections import defaultdict

class SimpleTFTChampion(object):
    def __init__(self, 
                 preferred_position: int, 
                 team: int, 
                 level: int=0):
        self.preferred_position = preferred_position
        self.team = team
        self.level = level
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.preferred_position == other.preferred_position,
                        self.team == other.team,
                        self.level == other.level])
        return False
    
    def level_up(self):
        self.level += 1

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
        
class SimpleTFTPlayer(object):
    def __init__(self, 
                 champion_pool_ptr: SimpleTFTChampionPool, 
                 board_size: int, 
                 bench_size: int, 
                 shop_size: int):
        self.champion_pool_ptr = champion_pool_ptr
        self.board = [None for _ in range(board_size)]
        self.bench = [None for _ in range(bench_size)]
        self.shop = [None for _ in range(shop_size)]
        self.action_positions = board_size + bench_size + shop_size + 1
        self.gold = 0
        self.hp = 10
        self.idle_action = self.action_positions * (self.action_positions - 1) + 1
        
    def take_action(self, action: int):
        if self.is_alive():
            action_from = action // self.action_positions
            action_to = action % self.action_positions
            if action_from < len(self.board):
                if action_to < len(self.board):
                    self.move_board_to_board(action_from, action_to)
                elif action_to < len(self.board) + len(self.bench):
                    bench_dest = action_to - len(self.board)
                    self.move_board_to_bench(action_from, bench_dest)
                elif action_to < action_from < len(self.board) + len(self.bench) + len(self.shop):
                    self.sell_from_board(action_from)
            elif action_from < len(self.board) + len(self.bench):
                bench_from = action_from - len(self.board)
                if action_to < len(self.board):
                    self.move_bench_to_board(bench_from, action_to)
                elif action_to < len(self.board) + len(self.bench):
                    bench_dest = action_to - len(self.board)
                    self.move_bench_to_bench(bench_from, bench_dest)
                elif action_to < action_from < len(self.board) + len(self.bench) + len(self.shop):
                    self.sell_from_bench(bench_from)
            elif action_from < len(self.board) + len(self.bench) + len(self.shop):
                shop_from = action_from - len(self.board) - len(self.bench)
                self.purchase_from_shop(shop_from)
            else:
                if action_to == 0:
                    self.refresh_shop
                if action_to == 1:
                    pass
        
    def make_action_mask(self) -> np.array:
        mask = np.zeros(self.idle_action + 1)
        if self.is_alive():
            action_from = 0
            for i_board, pos in enumerate(self.board):
                if pos:
                    mask[action_from:action_from + len(self.board)] = 1
                    mask[action_from + i_board] = 0
                    if not self.bench_full():
                        for i_bench, bench_pos in enumerate(self.bench):
                            mask[action_from + len(self.board) + i_bench] = 1
                    mask[action_from + len(self.board) + len(self.bench) + 1] = 1
                action_from += self.action_positions
            for i_bench, pos in enumerate(self.bench):
                if pos:
                    mask[action_from:action_from + len(self.board)] = 1
                    mask[action_from + len(self.board) + len(self.bench) + 1] = 1
                action_from += self.action_positions
            for i_shop, pos in enumerate(self.shop):
                if pos and self.gold > 0:
                    if not self.bench_full():
                        mask[action_from] = 1
                    else:
                        for board_pos in self.board:
                            if board_pos and board_pos == pos:
                                mask[action_from] = 1
                                break
                        for bench_pos in self.bench:
                            if bench_pos and bench_pos == pos:
                                mask[action_from] = 1
                                break
                action_from += self.action_positions
            if self.gold > 0:
                mask[action_from] = 1
        mask[self.idle_action] = 1
        return mask
        
    def move_board_to_board(self, board_from: int, board_to: int):
        if board_from < len(self.board) and board_to < len(self.board):
            from_champ = self.board[board_from]
            to_champ = self.board[board_to]
            self.board[board_from] = to_champ
            self.board[board_to] = from_champ

    def move_board_to_bench(self, board_from: int, bench_to: int):
        if board_from < len(self.board) and bench_to < len(self.bench):
            from_champ = self.board[board_from]
            to_champ = self.bench[bench_to]
            self.board[board_from] = to_champ
            self.bench[bench_to] = from_champ
            
    def move_bench_to_board(self, bench_from: int, board_to: int):
        if bench_from < len(self.bench) and board_to < len(self.board):
            from_champ = self.bench[bench_from]
            to_champ = self.board[board_to]
            self.bench[bench_from] = to_champ
            self.board[board_to] = from_champ
            
    def move_bench_to_bench(self, bench_from: int, bench_to: int):
        if bench_from < len(self.bench) and bench_to < len(self.bench):
            from_champ = self.bench[bench_from]
            to_champ = self.bench[bench_to]
            self.bench[bench_from] = to_champ
            self.bench[bench_to] = from_champ
        
    def purchase_from_shop(self, shop_from: int):
        if self.gold > 0 and self.shop[shop_from]:
            if self.add_champion(self.shop[shop_from]):
                self.gold -= 1
                self.shop[shop_from] = None
        
    def add_champion(self, champ: SimpleTFTChampion) -> bool:
            for pos in self.board:
                if pos and pos == champ:
                    pos.level_up()
                    return True
            for pos in self.bench:
                if pos and pos == champ:
                    pos.level_up()
                    return  True 
            for i, pos in enumerate(self.bench):
                if not pos:
                    self.bench[i] = champ
                    return True   
            return False
        
    def refresh_shop(self):
        if self.gold > 0:
            for i in range(len(self.shop)):
                if self.shop[i]:
                    self.champion_pool_ptr.add(self.shop[i])
                    self.shop[i] = None
            self.shop = self.champion_pool_ptr.sample(len(self.shop))
            self.gold -= 1
    
    def sell_from_board(self, board_from):
        self.champion_pool_ptr.add(self.board[board_from])
        self.board[board_from] = None

    def sell_from_bench(self, bench_from):
        self.champion_pool_ptr.add(self.bench[bench_from])
        self.bench[bench_from] = None
        
    def calculate_board_power(self) -> int:
        pwr = 0
        teams = defaultdict(list)
        for i, champ in enumerate(self.board):
            if champ:
                pwr += champ.level + 1
                if not champ.preferred_position in teams[champ.team]:
                    teams[champ.team].append(champ.preferred_position)
                if i == champ.preferred_position:
                    pwr += 1
        for team, members in teams.items():
            pwr += max(0, len(members) - 1)
        return pwr          
    
    def add_gold(self, amount: int):
        self.gold += amount
        
    def take_damage(self, amount: int=1):
        self.hp -= amount
        
    def is_alive(self) -> bool:
        return self.hp > 0
    
    def board_full(self) -> bool:
        for pos in self.board:
            if not pos:
                return False
        return True
    
    def bench_full(self) -> bool:
        for pos in self.bench:
            if not pos:
                return False
        return True
    
    def add_to_bench(self, champ: SimpleTFTChampion) -> bool:
        for i, pos in enumerate(self.bench):
            if not pos:
                self.bench[i] = champ
                return True
        return False
        
                
class SimpleTFT(object):
    def __init__(self):
        self.num_players = 2
        self.board_size = 3
        self.bench_size = 2
        self.shop_size = 2
        self.num_teams = 3
        self.team_size = 3
        self.champ_copies = 5
        self.actions_per_round = 5
        self.gold_per_round = 3
        self.interest_increment = 5
        
        self.observation_shape = (self.num_players, 
                                  self.board_size 
                                      + self.bench_size 
                                      + self.shop_size
                                      + 1,
                                  self.num_teams + self.board_size + int(np.log2(self.champ_copies))
                                  )
        
        self.champion_pool = None
        self.live_agents = ['player_{}'.format(i) for i in range(self.num_players)]
        self.players = {}
        self.actions_until_combat = 0
        
    def step(self, action: dict) -> (dict, dict, dict, dict, dict):
        for p, a in action.items():
            self.players[p].take_action(a)
        if not self.actions_until_combat:
            rewards = self.combat()
            self.actions_until_combat = self.actions_per_round
        else:
            rewards = {p: 0 for p in self.players}
        self.actions_until_combat -= 1
        
        return self.make_player_observations(), rewards, self.make_acting_player_dict(), self.make_dones(), self.make_action_masks()
                
    def reset(self) -> (dict, dict, dict):
        self.champion_pool = SimpleTFTChampionPool(self.champ_copies,
                                                   self.num_teams,
                                                   self.board_size)
        self.live_agents = ['player_{}'.format(i) for i in range(self.num_players)]
        self.players = {p: SimpleTFTPlayer(self.champion_pool, 
                                           self.board_size,
                                           self.bench_size,
                                           self.shop_size)
                        for p in self.live_agents}
        self.actions_until_combat = self.actions_per_round - 1
        for p, player in self.players.items():
            if not player.add_champion(self.champion_pool.sample(1)[0]):
                player.add_gold(1)
            player.add_gold(3)
            player.refresh_shop()
            
        return self.make_player_observations(), self.make_acting_player_dict(), self.make_dones(), self.make_action_masks()
        
    def combat(self) -> dict:
        self.live_agents = [p for p, player in self.players.items() if player.is_alive()]
        reward = {p: 0 for p in self.players}
        if len(self.live_agents) > 1:
            shuffle = np.random.choice(self.live_agents, len(self.live_agents), replace=False).tolist()
            prev = None
            for p in shuffle:
                if prev:
                    prev_pwr = prev.calculate_board_power()
                    pwr = self.players[p].calculate_board_power()
                    if prev_pwr == pwr:
                        prev.take_damage()
                        self.players[p].take_damage()
                    elif prev_pwr > pwr:
                        self.players[p].take_damage()
                    else:
                        prev.take_damage()
                    prev = None
                else:
                    prev = self.players[p]
            if prev:
                prev_pwr = prev.calculate_board_power()
                pwr = self.players[shuffle[0]].calculate_board_power()
                if prev_pwr == pwr:
                    prev.take_damage()
                elif pwr > prev_pwr:
                    prev.take_damage()
            
            new_live_agents = [p for p, player in self.players.items() if player.is_alive()]
            for p in self.live_agents:
                if p not in new_live_agents:
                    reward[p] -= 1
            self.live_agents = new_live_agents
            if len(self.live_agents) < 2:
                for p in self.live_agents:
                    reward[p] += 1
        return reward
            
      
    
    def make_dones(self) -> dict:
        if len(self.live_agents) > 1:
            return {p: not player.is_alive() for p, player in self.players.items()}
        else:
            return {p: True for p in self.players}
    
    def make_action_masks(self) -> dict:
        return {p: player.make_action_mask() for p, player in self.players.items()}
    
    def make_acting_player_dict(self) -> dict:
        return {p: p in self.live_agents for p in self.players}
    
    def make_player_observations(self) -> dict:
        obs = {}
        public_observations = self.make_public_observations()
        for p, player in self.players.items():
            p_obs = np.zeros(self.observation_shape)
            p_obs[0, :, :] = self.observe_player(player, False)
            ax1 = 1
            for op, _ in self.players.items():
                if op != p:
                    p_obs[ax1, :, :] = public_observations[op]
                    ax1 += 1
            obs[p] = p_obs
        return obs
        
    def make_public_observations(self) -> dict:
        return {p: self.observe_player(player) for p, player in self.players.items()}
        
    def observe_player(self, player: SimpleTFTPlayer, public=True) -> np.array:
        obs = np.zeros(self.observation_shape[1:])
        if player.is_alive():
            ax1 = 0
            for pos in player.board:
                if pos:
                    obs[ax1, :] = self.observe_champion(pos)
                ax1 += 1
            for pos in player.bench:
                if pos:
                    obs[ax1, :] = self.observe_champion(pos)
                ax1 += 1
            for pos in player.shop:
                if not public and pos:
                    obs[ax1, :] = self.observe_champion(pos)
                ax1 += 1      
            if not public:              
                obs[ax1, 0] = player.gold / 30
            obs[ax1, 1] = player.hp / 10
            obs[ax1, 2] = self.actions_until_combat / (self.actions_per_round - 1)
        return obs
        
    def observe_champion(self, champ: SimpleTFTChampion) -> np.array:
        obs = np.zeros(self.observation_shape[-1])
        if champ:
            obs[champ.team] = 1
            ax = self.num_teams
            obs[ax + champ.preferred_position] = 1
            ax += self.board_size
            obs[ax + champ.level] = 1
        return obs
        
        



        
        

        
        
        
        

        