# -*- coding: utf-8 -*-
from .champion import SimpleTFTChampion
from .champion_pool import SimpleTFTChampionPool
from .player import SimpleTFTPlayer
import numpy as np
import os

class SimpleTFT(object):
    def __init__(self, log_file_path: str=""):
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
        self.log_file_path = ""
        self.action_space = self.board_size + self.bench_size + self.shop_size + 1
        self.action_space = self.action_space * self.action_space
        
        if log_file_path and os.path.exists(os.path.dirname(log_file_path)):
            self.log_file_path = log_file_path
        
        self.observation_shape = (self.num_players, 
                                  self.board_size 
                                      + self.bench_size 
                                      + self.shop_size
                                      + 1,
                                  self.num_teams + self.board_size + int(np.log2(self.champ_copies)) + 1
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
            self.post_combat()
        else:
            rewards = {p: 0 for p in self.players}
        self.actions_until_combat -= 1
        
        return self.make_player_observations(), rewards, self.make_acting_player_dict(), self.make_dones(), self.make_action_masks()
                
    def reset(self, log_file_path: str="") -> (dict, dict, dict):
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
            champ = self.champion_pool.sample(1)[0]
            if not player.add_champion(champ):
                player.add_gold(1)
                self.champion_pool.add(champ)
            player.add_gold(self.gold_per_round + 1)
            player.refresh_shop()
        
        if log_file_path:
            self.log_file_path = log_file_path
            
        return self.make_player_observations(), self.make_acting_player_dict(), self.make_action_masks()
        
    def post_combat(self):
        for p, player in self.players.items():
            if not player.is_alive():
                player.death_cleanup()
        for p, player in self.players.items():
            if player.is_alive():
                player.add_gold(self.gold_per_round + min(player.gold // self.interest_increment, 5) + 1)
                player.refresh_shop()
                   
    def combat(self) -> dict:
        self.live_agents = [p for p, player in self.players.items() if player.is_alive()]
        reward = {p: 0 for p in self.players}
        if len(self.live_agents) > 1:
            shuffle = np.random.choice(self.live_agents, len(self.live_agents), replace=False).tolist()
            prev = None
            for p in shuffle:
                if prev:
                    prev_pwr = self.players[prev].calculate_board_power()
                    pwr = self.players[p].calculate_board_power()
                    if self.log_file_path:
                        self.log_matchup(prev, prev_pwr, self.players[prev],
                                         p, pwr, self.players[p])
                    if prev_pwr == pwr:
                        self.players[prev].take_damage()
                        self.players[p].take_damage()
                    elif prev_pwr > pwr:
                        self.players[p].take_damage()
                    else:
                        self.players[prev].take_damage()
                    prev = None
                else:
                    prev = p
            if prev:
                prev_pwr = self.players[prev].calculate_board_power()
                pwr = self.players[shuffle[0]].calculate_board_power()
                if self.log_file_path:
                    self.log_matchup(prev, prev_pwr, self.players[prev],
                                     shuffle[0], pwr, self.players[shuffle[0]])
                if prev_pwr == pwr:
                    self.players[prev].take_damage()
                elif pwr > prev_pwr:
                    self.players[prev].take_damage()
            
            new_live_agents = [p for p, player in self.players.items() if player.is_alive()]
            for p in self.live_agents:
                if p not in new_live_agents:
                    reward[p] -= 1
            self.live_agents = new_live_agents
            if len(self.live_agents) < 2:
                for p in self.live_agents:
                    reward[p] += 1
        return reward
            
    def action_space_size(self):
        return self.action_space
    
    def log_matchup(self, player1_name: str, 
                    player1_power: int,
                    player1: SimpleTFTPlayer, 
                    player2_name: str, 
                    player2_power: int,
                    player2: SimpleTFTPlayer):
                 
        header = f"\t{player1_name} - Power: {player1_power} - HP: {player1.hp} - Gold: {player1.gold}\t\t\t{player2_name} - Power: {player2_power} - HP: {player2.hp} - Gold: {player2.gold}\n"
        p1_teams = [c.team for c in player1.bench if c]
        p1_pos = [c.preferred_position for c in player1.bench if c]
        p1_level = [c.level for c in player1.bench if c]
        p2_teams = [c.team for c in player2.bench if c]
        p2_pos = [c.preferred_position for c in player2.bench if c]
        p2_level = [c.level for c in player2.bench if c]
        p1 = zip(p1_teams, p1_pos, p1_level)
        p2 = zip(p2_teams, p2_pos, p2_level)
        p1 = [i for i in p1]
        p2 = [i for i in p2]
        ts = 6 - int(1.4 * len(p1))
        bench_header = f"\t Bench: {p1}" + "\t" * ts + f" Bench: {p2}\n"
        sub_header = f"{'Position':^10} | {'Team':^10} | {'Preferred Pos':^15} | {'Level':^10}\t|||\t"
        sub_header += f"{'Position':^10} | {'Team':^10} | {'Preferred Pos':^15} | {'Level':^10}\n"
        divider = '-' * (10 + 1 + 10 + 1 + 15 + 1 + 10 + 1) * 2 + "\n"
    
        log_entry = header + bench_header + sub_header + divider
    
        for position, champ in enumerate(player1.board):
            team, pref_pos, level = "None", "None", "None"
            if champ:
                team = champ.team
                pref_pos = champ.preferred_position
                level = champ.level
            log_entry += f"{position:^10} | {team:^10} | {pref_pos:^15} | {level:^10}\t|||\t"
            team, pref_pos, level = "None", "None", "None"
            champ = player2.board[position]
            if champ:
                team = champ.team
                pref_pos = champ.preferred_position
                level = champ.level
            log_entry += f"{position:^10} | {team:^10} | {pref_pos:^15} | {level:^10}\n"
           
        if self.log_file_path:
            with open(self.log_file_path, 'a') as file:
                file.write(log_entry + "\n")
    
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

