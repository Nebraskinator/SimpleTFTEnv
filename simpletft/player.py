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
            """
            Initialize a SimpleTFT player with a reference to a champion pool, and sizes for board, bench, and shop.
    
            :param champion_pool_ptr: Reference to the SimpleTFTChampionPool.
            :param board_size: The size of the player's board.
            :param bench_size: The size of the player's bench.
            :param shop_size: The size of the player's shop.
            :raises ValueError: If any size values are non-positive integers.
            """
            if not all(isinstance(x, int) and x > 0 for x in [board_size, bench_size, shop_size]):
                raise ValueError("board_size, bench_size, and shop_size must be positive integers")
    
            self.__champion_pool_ptr = champion_pool_ptr
            self.__board = [None for _ in range(board_size)]
            self.__bench = [None for _ in range(bench_size)]
            self.__shop = [None for _ in range(shop_size)]
            self.__action_positions = board_size + bench_size + shop_size + 1
            self.__gold = 0
            self.__hp = 10
            self.__killed = False
            self.__idle_action = self.__action_positions * (self.__action_positions - 1) + 1
        
    @property
    def gold(self):
        return self.__gold

    @property
    def hp(self):
        return self.__hp
    
    @property
    def board(self):
        return self.__board.copy()
    
    @property
    def bench(self):
        return self.__bench.copy()
    
    @property
    def shop(self):
        return self.__shop.copy()
        
    def take_action(self, action: int):
        """
        Execute an action based on the given action code.

        :param action: An integer representing the specific action to be taken.
        :raises ValueError: If the action is not an integer or is out of the valid range.
        """
        if not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer")
        if not 0 <= action < self.__action_positions * self.__action_positions:
            raise ValueError(f"Action must be within the range 0 to {self.__action_positions * self.__action_positions - 1}")

        if self.is_alive():
            action_from = action // self.__action_positions
            action_to = action % self.__action_positions

            if action_from < len(self.__board):
                self._process_board_actions(action_from, action_to)
            elif action_from < len(self.__board) + len(self.__bench):
                self._process_bench_actions(action_from, action_to)
            elif action_from < self.__action_positions - 1:
                self._process_shop_actions(action_from, action_to)
            else:
                self._process_other_actions(action_to)

    def _process_board_actions(self, action_from: int, action_to: int):
        """
        Process actions originating from the board.

        :param action_from: The board position from which the action originates.
        :param action_to: The target position of the action.
        """
        if action_to < len(self.__board):
            self.move_board_to_board(action_from, action_to)
        elif action_to < len(self.__board) + len(self.__bench):
            bench_dest = action_to - len(self.__board)
            self.move_board_to_bench(action_from, bench_dest)
        elif action_to == len(self.__board) + len(self.__bench):
            self.sell_from_board(action_from)

    def _process_bench_actions(self, action_from: int, action_to: int):
        """
        Process actions originating from the bench.

        :param action_from: The bench position from which the action originates.
        :param action_to: The target position of the action.
        """
        bench_from = action_from - len(self.__board)
        if action_to < len(self.__board):
            self.move_bench_to_board(bench_from, action_to)
        elif action_to < len(self.__board) + len(self.__bench):
            bench_dest = action_to - len(self.__board)
            self.move_bench_to_bench(bench_from, bench_dest)
        elif action_to == len(self.__board) + len(self.__bench):
            self.sell_from_bench(bench_from)

    def _process_shop_actions(self, action_from: int, action_to: int):
        """
        Process actions originating from the shop.

        :param action_from: The shop position from which the action originates.
        :param action_to: The target position of the action.
        """
        shop_from = action_from - len(self.__board) - len(self.__bench)
        self.purchase_from_shop(shop_from)

    def _process_other_actions(self, action_to: int):
        """
        Process other types of actions.

        :param action_to: The target position of the action.
        """
        if action_to == 0:
            self.refresh_shop()
        # 'pass' for action_to == 1 is already implicit
        
    def make_action_mask(self) -> np.array:
        """
        Create an action mask representing the valid actions the player can take.

        :return: A numpy array representing the action mask.
        """
        mask = np.zeros(self.__action_positions * self.__action_positions)
        if self.is_alive():
            action_from = 0
            for i_board, pos in enumerate(self.__board):
                if pos:
                    self._update_mask_for_board(mask, action_from, i_board)
                action_from += self.__action_positions

            for i_bench, pos in enumerate(self.__bench):
                if pos:
                    self._update_mask_for_bench(mask, action_from)
                action_from += self.__action_positions

            for i_shop, pos in enumerate(self.__shop):
                self._update_mask_for_shop(mask, action_from, pos, i_shop)
                action_from += self.__action_positions

            if self.__gold > 0:
                mask[action_from] = 1  # Refresh shop action

        mask[self.__idle_action] = 1  # Always allow idle action
        return mask

    def _update_mask_for_board(self, mask, action_from, i_board):
        """
        Update the action mask for board positions.

        :param mask: The action mask to update.
        :param action_from: The starting action position.
        :param i_board: The index of the board position being updated.
        """
        mask[action_from:action_from + len(self.__board) + len(self.__bench) + 1] = 1
        mask[action_from + i_board] = 0  # Can't move to the same position
        mask[action_from + len(self.__board) + len(self.__bench)] = 1  # Sell action

    def _update_mask_for_bench(self, mask, action_from):
        """
        Update the action mask for bench positions.

        :param mask: The action mask to update.
        :param action_from: The starting action position.
        """
        mask[action_from:action_from + len(self.__board)] = 1
        mask[action_from + len(self.__board) + len(self.__bench)] = 1  # Sell action

    def _update_mask_for_shop(self, mask, action_from, pos, i_shop):
        """
        Update the action mask for shop positions.

        :param mask: The action mask to update.
        :param action_from: The starting action position.
        :param pos: The champion position in the shop.
        :param i_shop: The index of the shop position being updated.
        """
        if pos and self.__gold > 0:
            # Check if the bench is not full or there's a matching champion on the board or bench
            if not self.bench_full() or self._has_matching_champion(pos):
                mask[action_from] = 1

    def _has_matching_champion(self, champion: SimpleTFTChampion) -> bool:
        """
        Check if there is a matching champion on the board or bench.

        :param champion: The SimpleTFTChampion instance to match against.
        :return: True if a matching champion is found, False otherwise.
        """
        # Check both the board and the bench for a match
        return any(champ.match(champion) for champ in self.__board if champ) or \
               any(champ.match(champion) for champ in self.__bench if champ)

    def move_board_to_board(self, board_from: int, board_to: int):
        """
        Move a champion from one position to another on the board.

        :param board_from: The starting position of the champion on the board.
        :param board_to: The target position of the champion on the board.
        :raises IndexError: If either board_from or board_to are out of bounds.
        """
        if not 0 <= board_from < len(self.__board) or not 0 <= board_to < len(self.__board):
            raise IndexError("Board positions are out of bounds")

        from_champ = self.__board[board_from]
        to_champ = self.__board[board_to]
        self.__board[board_from] = to_champ
        self.__board[board_to] = from_champ

    def move_board_to_bench(self, board_from: int, bench_to: int):
        """
        Move a champion from the board to the bench.

        :param board_from: The position of the champion on the board.
        :param bench_to: The target position on the bench.
        :raises IndexError: If board_from or bench_to are out of bounds.
        """
        if not 0 <= board_from < len(self.__board) or not 0 <= bench_to < len(self.__bench):
            raise IndexError("Board or bench positions are out of bounds")

        from_champ = self.__board[board_from]
        to_champ = self.__bench[bench_to]
        self.__board[board_from] = to_champ
        self.__bench[bench_to] = from_champ

    def move_bench_to_board(self, bench_from: int, board_to: int):
        """
        Move a champion from the bench to the board.

        :param bench_from: The position of the champion on the bench.
        :param board_to: The target position on the board.
        :raises IndexError: If bench_from or board_to are out of bounds.
        """
        if not 0 <= bench_from < len(self.__bench) or not 0 <= board_to < len(self.__board):
            raise IndexError("Bench or board positions are out of bounds")

        from_champ = self.__bench[bench_from]
        to_champ = self.__board[board_to]
        self.__bench[bench_from] = to_champ
        self.__board[board_to] = from_champ

    def move_bench_to_bench(self, bench_from: int, bench_to: int):
        """
        Move a champion from one position to another on the bench.

        :param bench_from: The starting position of the champion on the bench.
        :param bench_to: The target position of the champion on the bench.
        :raises IndexError: If either bench_from or bench_to are out of bounds.
        """
        if not 0 <= bench_from < len(self.__bench) or not 0 <= bench_to < len(self.__bench):
            raise IndexError("Bench positions are out of bounds")

        from_champ = self.__bench[bench_from]
        to_champ = self.__bench[bench_to]
        self.__bench[bench_from] = to_champ
        self.__bench[bench_to] = from_champ

    def purchase_from_shop(self, shop_from: int):
        """
        Purchase a champion from the shop.

        :param shop_from: The position of the champion in the shop.
        :raises IndexError: If shop_from is out of bounds.
        :raises ValueError: If there is insufficient gold or the shop position is empty.
        """
        if not 0 <= shop_from < len(self.__shop):
            raise IndexError("Shop position is out of bounds")
        if self.__gold <= 0:
            raise ValueError("Insufficient gold to make a purchase")
        if not self.__shop[shop_from]:
            raise ValueError("No champion at the specified shop position")

        if self.add_champion(self.__shop[shop_from]):
            self.__gold -= 1
            self.__shop[shop_from] = None

    def find_matches(self):
        """
        Find and process matching champions on the board and bench to level them up.
        """
        # Searching for matches on the board
        for board_i, pos in enumerate(self.__board):
            if pos:
                if self._process_match_in_list(self.__board, pos, start_index=board_i + 1):
                    return  # Restart search after finding a match
                if self._process_match_in_list(self.__bench, pos):
                    return  # Restart search after finding a match
                
        # Searching for matches on the bench
        for bench_i, pos in enumerate(self.__bench):
            if pos:
                if self._process_match_in_list(self.__bench, pos, start_index=bench_i + 1):
                    return  # Restart search after finding a match

    def _process_match_in_list(self, lst, champion, start_index=0):
        """
        Process matching champions in a given list (board or bench).

        :param lst: The list to search in (either board or bench).
        :param champion: The champion to match against.
        :param start_index: The index to start searching from.
        :return: True if a match was found and processed, False otherwise.
        """
        for i, match_pos in enumerate(lst[start_index:], start=start_index):
            if match_pos and champion.match(match_pos):
                champion.level_up()
                lst[i] = None
                self.find_matches()  # Recursively search for new matches
                return True
        return False

    def add_champion(self, champ: SimpleTFTChampion) -> bool:
        """
        Attempt to add a champion to the board or bench.

        :param champ: The SimpleTFTChampion instance to be added.
        :return: True if the champion was successfully added or matched, False otherwise.
        """
        for pos in self.__board:
            if pos and pos.match(champ):
                pos.level_up()
                self.find_matches()
                return True

        for pos in self.__bench:
            if pos and pos.match(champ):
                pos.level_up()
                self.find_matches()
                return True

        for i, pos in enumerate(self.__bench):
            if not pos:
                self.__bench[i] = champ
                return True

        return False  # No space or match found
        
    def refresh_shop(self):
        """
        Refresh the shop's champions if the player has enough gold. 
        Existing champions in the shop are returned to the champion pool.
        """
        if self.__gold <= 0:
            raise ValueError("Insufficient gold to refresh shop")

        for i in range(len(self.__shop)):
            if self.__shop[i]:
                self.__champion_pool_ptr.add(self.__shop[i])
                self.__shop[i] = None

        self.__shop = self.__champion_pool_ptr.sample(len(self.__shop))
        self.__gold -= 1

    def sell_from_board(self, board_from: int):
        """
        Sell a champion from the board, adding its value to the player's gold.

        :param board_from: The board position of the champion to be sold.
        :raises IndexError: If board_from is out of bounds.
        """
        if not 0 <= board_from < len(self.__board):
            raise IndexError("Board position is out of bounds")

        if self.__board[board_from]:
            self.__champion_pool_ptr.add(self.__board[board_from])
            self.__gold += 2 ** self.__board[board_from].level
            self.__board[board_from] = None

    def sell_from_bench(self, bench_from: int):
        """
        Sell a champion from the bench, adding its value to the player's gold.

        :param bench_from: The bench position of the champion to be sold.
        :raises IndexError: If bench_from is out of bounds.
        """
        if not 0 <= bench_from < len(self.__bench):
            raise IndexError("Bench position is out of bounds")

        if self.__bench[bench_from]:
            self.__champion_pool_ptr.add(self.__bench[bench_from])
            self.__gold += 2 ** self.__bench[bench_from].level
            self.__bench[bench_from] = None
        
    def calculate_board_power(self) -> int:
        """
        Calculate the total power of the board based on the levels of the champions and their positions.

        :return: An integer representing the total power of the board.
        """
        pwr = 0
        teams = defaultdict(list)
        for i, champ in enumerate(self.__board):
            if champ:
                pwr += champ.level + 1  # Base power from champion level
                if champ.preferred_position not in teams[champ.team]:
                    teams[champ.team].append(champ.preferred_position)
                if i == champ.preferred_position:
                    pwr += 1  # Bonus for being in the preferred position

        # Team bonus calculation
        for team, members in teams.items():
            pwr += max(0, len(members) - 1)
        return pwr

    def add_gold(self, amount: int):
        """
        Add a specified amount of gold to the player.

        :param amount: The amount of gold to be added.
        :raises ValueError: If the amount is negative.
        """
        if amount < 0:
            raise ValueError("Cannot add a negative amount of gold")
        self.__gold += amount

    def take_damage(self, amount: int = 1):
        """
        Inflict damage to the player, reducing their health points.

        :param amount: The amount of damage to be inflicted.
        :raises ValueError: If the amount is negative.
        """
        if amount < 0:
            raise ValueError("Cannot inflict negative damage")
        self.__hp -= amount

    def is_alive(self) -> bool:
        """
        Check if the player is still alive.

        :return: True if the player's health points are greater than 0, False otherwise.
        """
        return self.__hp > 0
    
    def death_cleanup(self):
        """
        Perform cleanup actions upon the death of the player. 
        This includes returning all champions from board, bench, and shop to the champion pool.
        """
        if not self.__killed:
            self._return_champions_to_pool(self.__board)
            self._return_champions_to_pool(self.__bench)
            self._return_champions_to_pool(self.__shop)
            self.__killed = True

    def _return_champions_to_pool(self, lst):
        """
        Helper method to return champions from a list (board, bench, or shop) to the champion pool.

        :param lst: The list of champions (board, bench, or shop).
        """
        for i, champ in enumerate(lst):
            if champ:
                self.__champion_pool_ptr.add(champ)
                lst[i] = None

    def board_full(self) -> bool:
        """
        Check if the board is full.

        :return: True if the board is full, False otherwise.
        """
        return all(pos is not None for pos in self.__board)

    def bench_full(self) -> bool:
        """
        Check if the bench is full.

        :return: True if the bench is full, False otherwise.
        """
        return all(pos is not None for pos in self.__bench)

    def add_to_bench(self, champ: SimpleTFTChampion) -> bool:
        """
        Attempt to add a champion to the bench.

        :param champ: The SimpleTFTChampion instance to be added.
        :return: True if the champion was successfully added, False otherwise.
        :raises TypeError: If champ is not an instance of SimpleTFTChampion.
        """
        if not isinstance(champ, SimpleTFTChampion):
            raise TypeError("champ must be an instance of SimpleTFTChampion")

        for i, pos in enumerate(self.__bench):
            if not pos:
                self.__bench[i] = champ
                return True
        return False  # Bench is full