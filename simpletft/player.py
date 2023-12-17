# -*- coding: utf-8 -*-
from .champion import SimpleTFTChampion
from .champion_pool import SimpleTFTChampionPool
import numpy as np
from collections import defaultdict

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
        self.killed = False
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
                    self.refresh_shop()
                if action_to == 1:
                    pass
        
    def make_action_mask(self) -> np.array:
        mask = np.zeros(self.action_positions * self.action_positions)
        if self.is_alive():
            action_from = 0
            for i_board, pos in enumerate(self.board):
                if pos:
                    mask[action_from:action_from + len(self.board)] = 1
                    mask[action_from + i_board] = 0
                    if not self.bench_full():
                        for i_bench, bench_pos in enumerate(self.bench):
                            mask[action_from + len(self.board) + i_bench] = 1
                    mask[action_from + len(self.board) + len(self.bench)] = 1
                action_from += self.action_positions
            for i_bench, pos in enumerate(self.bench):
                if pos:
                    mask[action_from:action_from + len(self.board)] = 1
                    mask[action_from + len(self.board) + len(self.bench)] = 1
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
        
    def find_matches(self):
        for board_i, pos in enumerate(self.board):
            if pos:
                for match_i, match_pos in enumerate(self.board[board_i + 1:]):
                    if pos.match(match_pos):
                        pos.level_up()
                        self.board[match_i + board_i + 1] = None
                        self.find_matches()
                        return
                for match_i, match_pos in enumerate(self.bench):
                    if pos.match(match_pos):
                        pos.level_up()
                        self.bench[match_i + board_i + 1] = None
                        self.find_matches()
                        return                    
        for bench_i, pos in enumerate(self.bench):
            if pos:
                for match_i, match_pos in enumerate(self.bench[bench_i + 1:]):
                    if match_pos and pos.match(match_pos):
                        pos.level_up()
                        self.bench[match_i + bench_i + 1] = None
                        self.find_matches()
                        return          
        
    def add_champion(self, champ: SimpleTFTChampion) -> bool:
            for pos in self.board:
                if pos and pos.match(champ):
                    pos.level_up()
                    self.find_matches()
                    return True
            for pos in self.bench:
                if pos and pos.match(champ):
                    pos.level_up()
                    self.find_matches()
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
        if self.board[board_from]:
            self.champion_pool_ptr.add(self.board[board_from])
            self.gold += 2**self.board[board_from].level
            self.board[board_from] = None

    def sell_from_bench(self, bench_from):
        if self.bench[bench_from]:
            self.champion_pool_ptr.add(self.bench[bench_from])
            self.gold += 2**self.bench[bench_from].level
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
    
    def death_cleanup(self):
        if not self.killed:
            for i, c in enumerate(self.board):
                if c:
                    self.champion_pool_ptr.add(c)
                    self.board[i] = None
            for i, c in enumerate(self.bench):
                if c:
                    self.champion_pool_ptr.add(c)
                    self.bench[i] = None
            for i, c in enumerate(self.shop):
                if c:
                    self.champion_pool_ptr.add(c)
                    self.shop[i] = None
            self.killed = True
    
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