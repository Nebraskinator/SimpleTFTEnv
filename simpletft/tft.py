# -*- coding: utf-8 -*-
from .champion import SimpleTFTChampion
from .champion_pool import SimpleTFTChampionPool
from .player import SimpleTFTPlayer
import numpy as np
import os

class SimpleTFT(object):
    def __init__(self, config: dict = {}):
        """
        Initialize the SimpleTFT game with configurable settings.

        :param config: A dictionary containing game configuration settings.
        :raises ValueError: If any configuration values are invalid.
        """
        # Default values can be set here or obtained from the config
        self.__num_players = config.get('num_players', 2)
        self.__board_size = config.get('board_size', 3)
        self.__bench_size = config.get('bench_size', 2)
        self.__shop_size = config.get('shop_size', 2)
        self.__num_teams = config.get('num_teams', 3)
        self.__team_size = config.get('team_size', 3)
        self.__champ_copies = config.get('champ_copies', 5)
        self.__actions_per_round = config.get('actions_per_round', 5)
        self.__gold_per_round = config.get('gold_per_round', 3)
        self.__interest_increment = config.get('interest_increment', 5)
        self.__debug = config.get('debug', False)
        
        valid_reward_structures = ['game_placement', 'damage', 'mixed']
        self.__reward_structure = config.get('reward_structure', 'game_placement')  # Default to 'game_placement'
        if self.__reward_structure not in valid_reward_structures:
            raise ValueError(f"Invalid reward structure. Must be one of {valid_reward_structures}")
            
        # Validate configuration
        if not all(isinstance(value, int) and value > 0 for value in [
            self.__num_players, self.__board_size, self.__bench_size, 
            self.__shop_size, self.__num_teams, self.__team_size, 
            self.__champ_copies, self.__actions_per_round, 
            self.__gold_per_round, self.__interest_increment]):
            raise ValueError("All configuration values must be positive integers")

        # Calculate the maximum attainable champion level
        max_champ_level = int(np.log2(self.__champ_copies))

        # Calculate the total number of positions among all players
        total_positions = self.__num_players * (self.__board_size + self.__bench_size)

        # Calculate the minimum pool size required
        min_pool_size = total_positions * (2 ** max_champ_level) + self.__shop_size * self.__num_players

        # Calculate the total number of champions available in the pool
        total_champions_available = self.__champ_copies * self.__num_teams * self.__board_size

        # Validate if the champion pool is sufficient
        if min_pool_size > total_champions_available:
            raise ValueError("Insufficient champions in the pool based on the configuration. "
                             "Consider increasing champ_copies, num_teams, or reducing board_size, bench_size, or num_players.")


        self.__action_space_size = self.__board_size + self.__bench_size + self.__shop_size + 1
        self.__action_space_size *= self.__action_space_size
        
        self.__observation_shape = (self.__num_players, 
                                    self.__board_size 
                                        + self.__bench_size 
                                        + self.__shop_size
                                        + 1,
                                    self.__num_teams + self.__board_size + int(np.log2(self.__champ_copies)) + 1
                                    )
        
        self.__champion_pool = None
        self.__live_agents = ['player_{}'.format(i) for i in range(self.__num_players)]
        self.__players = {}
        self.__actions_until_combat = 0
        self.__log = []
        self.__log_file_path = ""
        
    @property
    def live_agents(self):
        """
        Get a copy of the current live agents in the game.

        :return: A copy of the list of live agent identifiers.
        """
        return self.__live_agents.copy()
    
    @property
    def observation_shape(self):
        """
        Get the shape of the observation space for the game.
    
        :return: A tuple representing the shape of the observation space.
        """
        return self.__observation_shape
        
    @property
    def num_players(self):
        """
        Get the number of players in the game.
    
        :return: int 
        """
        return self.__num_players
    
    def step(self, action: dict) -> (dict, dict, dict, dict, dict):
        """
        Process a game step given the actions of each player.

        :param action: A dictionary mapping player identifiers to their actions.
        :return: Tuple containing player observations, rewards, acting players, game state, and action masks.
        """
        for p, a in action.items():
            if p not in self.__players:
                raise ValueError(f"Player {p} is not part of the game.")
            self.__players[p].take_action(a)

        if self.__debug:
            self._log_player_states()
            self._dump_logs()

        if not self.__actions_until_combat:
            if self.__debug:
                self.__log.append("combat round")
            rewards = self.combat()
            self.__actions_until_combat = self.__actions_per_round
            self.post_combat()
        else:
            rewards = {p: 0 for p in self.__players}
            self.__actions_until_combat -= 1

        return (self.make_player_observations(), rewards, self.make_acting_player_dict(), 
                self.make_dones(), self.make_action_masks())
                
    def reset(self, log_file_path: str = "") -> (dict, dict, dict):
        """
        Reset the game to its initial state.

        :param log_file_path: Optional path for a log file.
        :return: Tuple containing initial player observations, acting players, and action masks.
        """
        self.__champion_pool = SimpleTFTChampionPool(self.__champ_copies,
                                                     self.__num_teams,
                                                     self.__board_size,
                                                     debug=self.__debug)
        self.__live_agents = ['player_{}'.format(i) for i in range(self.__num_players)]
        self.__players = {p: SimpleTFTPlayer(self.__champion_pool, 
                                             self.__board_size,
                                             self.__bench_size,
                                             self.__shop_size,
                                             debug=self.__debug)
                          for p in self.__live_agents}
        self.__actions_until_combat = self.__actions_per_round - 1

        for p, player in self.__players.items():
            champ = self.__champion_pool.sample(1)[0]
            if not player.add_champion(champ):
                player.add_gold(1)
                self.__champion_pool.add(champ)
            player.add_gold(self.__gold_per_round + 1)
            player.refresh_shop()
            
        if self.__debug:
            if self.__log:
                self._dump_logs()
            self._log_player_states()

        if log_file_path:
            if os.path.exists(os.path.dirname(log_file_path)) or os.path.isdir(os.path.dirname(log_file_path)):
                self.__log_file_path = log_file_path
            else:
                raise ValueError(f"Provided log_file_path is not a valid directory: {log_file_path}")

        return self.make_player_observations(), self.make_acting_player_dict(), self.make_action_masks()
        
    def post_combat(self):
        """
        Perform post-combat actions for each player, including cleanup and adding gold.
        """
        for p, player in self.__players.items():
            if not player.is_alive():
                player.death_cleanup()
            elif player.is_alive():
                gold_addition = self.__gold_per_round + min(player.gold // self.__interest_increment, 5) + 1
                player.add_gold(gold_addition)
                player.refresh_shop()
                   
    def combat(self) -> dict:
        """
        Conduct combat between players and assign rewards.

        :return: A dictionary containing the rewards for each player.
        """
        self.__live_agents = [p for p, player in self.__players.items() if player.is_alive()]
        rewards = {p: 0 for p in self.__players}
        combat_results = {}

        if len(self.__live_agents) > 1:
            shuffle = np.random.choice(self.__live_agents, len(self.__live_agents), replace=False).tolist() if len(self.__live_agents) > 2 else self.__live_agents.copy()
            prev = None

            for p in shuffle:
                if prev:
                    self._resolve_combat(prev, p, combat_results)
                    prev = None
                else:
                    prev = p

            # Resolve combat for the last agent if odd number of agents
            if prev:
                self._resolve_last_combat(prev, shuffle[0], combat_results)

            self._update_live_agents()
            self._assign_rewards_based_on_structure(rewards, combat_results)

        if self.__debug:
            for p, reward in rewards.items():
                self.__log.append(f"{p}: received {reward} reward")

        return rewards

    def _resolve_last_combat(self, last_player, first_player, combat_results):
        """
        Resolve combat for the last player in case of an odd number of players.

        :param last_player: Identifier for the last player.
        :param first_player: Identifier for the first player in the shuffle.
        :param rewards: The dictionary to update with combat results.
        """
        last_player_power = self.__players[last_player].calculate_board_power()
        first_player_power = self.__players[first_player].calculate_board_power()

        if self.__log_file_path:
            self.log_matchup(last_player, last_player_power, self.__players[last_player],
                             first_player, first_player_power, self.__players[first_player])
            
        # Only the last player takes damage if they lose or draw
        if last_player_power <= first_player_power:
            self.__players[last_player].take_damage()
            combat_results[last_player] = -1
        else:
            combat_results[last_player] = +1

    def _resolve_combat(self, player1, player2, combat_results):
        """
        Resolve combat between two players and update rewards.

        :param player1: Identifier for the first player.
        :param player2: Identifier for the second player.
        :param rewards: The dictionary to update with combat results.
        """
        player1_power = self.__players[player1].calculate_board_power()
        player2_power = self.__players[player2].calculate_board_power()

        if self.__log_file_path:
            self.log_matchup(player1, player1_power, self.__players[player1],
                             player2, player2_power, self.__players[player2])

        # Resolve combat outcome
        if player1_power == player2_power:
            self.__players[player1].take_damage()
            combat_results[player1] = -1
            self.__players[player2].take_damage()
            combat_results[player2] = -1
        elif player1_power > player2_power:
            self.__players[player2].take_damage()
            combat_results[player2] = -1
            combat_results[player1] = +1
        else:
            self.__players[player1].take_damage()
            combat_results[player1] = -1
            combat_results[player2] = +1

    def _update_live_agents(self):
        """
        Update the list of live agents after combat.
        """
        self.__live_agents = [p for p, player in self.__players.items() if player.is_alive()]

    def _assign_rewards_based_on_structure(self, rewards, combat_results):
            """
            Assign rewards to players based on the selected reward structure.
    
            :param rewards: The rewards dictionary to be updated.
            :param combat_results: The combat results for reward calculation.
            """
            if self.__reward_structure in ('damage', 'mixed'):
                for p, result in combat_results.items():
                    rewards[p] += result
            if self.__reward_structure in ('game_placement', 'mixed'):
                loss_penalty = (len(self.__live_agents) >= self.__num_players // 2) * -1
                for p in combat_results.keys():
                    if p not in self.__live_agents:
                        rewards[p] += loss_penalty
                    elif len(self.__live_agents) == 1:
                        rewards[p] += 1
            
    def action_space_size(self):
        return self.__action_space_size
    
    def log_matchup(self, player1_name: str, player1_power: int, player1: SimpleTFTPlayer, 
                    player2_name: str, player2_power: int, player2: SimpleTFTPlayer):
        """
        Log the matchup details between two players.

        :param player1_name: Name of the first player.
        :param player1_power: Combat power of the first player.
        :param player1: First player object.
        :param player2_name: Name of the second player.
        :param player2_power: Combat power of the second player.
        :param player2: Second player object.
        """
        # Ensure the log file path is set
        if not self.__log_file_path:
            print("Log file path is not set. Cannot log matchup.")
            return

        # Building the log entry
        log_entry = self._build_log_entry(player1_name, player1_power, player1, player2_name, player2_power, player2)

        # Writing to the log file
        try:
            with open(self.__log_file_path, 'a') as file:
                file.write(log_entry + "\n")
        except IOError as e:
            print(f"Failed to write to log file: {e}")

    def _build_log_entry(self, player1_name, player1_power, player1, player2_name, player2_power, player2):                 
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
           
        return log_entry
    
    def make_dones(self) -> dict:
        """
        Create a dictionary indicating whether each player is done with the game.

        :return: A dictionary mapping player identifiers to their done status.
        """

        return {p: (len(self.__live_agents) <= 1) or not player.is_alive() for p, player in self.__players.items()}
    
    def make_action_masks(self) -> dict:
        """
        Generate action masks for each player.

        :return: A dictionary mapping player identifiers to their action masks.
        """
        return {p: player.make_action_mask() for p, player in self.__players.items()}
    
    def make_acting_player_dict(self) -> dict:
        """
        Create a dictionary indicating the active (live) players in the game.

        :return: A dictionary mapping player identifiers to a boolean indicating if they are active.
        """
        return {p: (p in self.__live_agents) for p in self.__players}
    
    def make_player_observations(self) -> dict:
        """
        Generate observations for each player, including both their own and others' publicly visible states.

        :return: A dictionary of observations for each player.
        """
        observations = {}
        public_observations = self.make_public_observations()
        for player_id, player in self.__players.items():
            player_obs = np.zeros(self.__observation_shape)
            player_obs[0, :, :] = self.observe_player(player, False)

            ax1 = 1
            for other_player_id, _ in self.__players.items():
                if other_player_id != player_id:
                    player_obs[ax1, :, :] = public_observations[other_player_id]
                    ax1 += 1

            observations[player_id] = player_obs
        return observations
        
    def make_public_observations(self) -> dict:
        """
        Generate public observations for each player.

        :return: A dictionary of public observations for each player.
        """
        return {player_id: self.observe_player(player) for player_id, player in self.__players.items()}
        
    def observe_player(self, player: SimpleTFTPlayer, public=True) -> np.array:
        """
        Observe the state of a player.

        :param player: The player to observe.
        :param public: Flag to determine if the observation is public or private.
        :return: An array representing the player's state.
        """
        observation = np.zeros(self.__observation_shape[1:])
        if player.is_alive():
            observation = self._build_player_observation(player, public)
        return observation


    def _build_player_observation(self, player: SimpleTFTPlayer, public=True) -> np.array:
        observation = np.zeros(self.observation_shape[1:])
        if player.is_alive():
            ax1 = 0
            for i, pos in enumerate(player.board):
                if pos:
                    observation[ax1, :] = self.observe_champion(pos)
                ax1 += 1
            for i, pos in enumerate(player.bench):
                if pos:
                    observation[ax1, :] = self.observe_champion(pos)
                ax1 += 1
            for i, pos in enumerate(player.shop):
                if not public and pos:
                    observation[ax1, :] = self.observe_champion(pos)
                ax1 += 1      
            if not public:              
                observation[ax1, 0] = np.clip(player.gold / 30, 0, 1)
            observation[ax1, 1] = player.hp / 10
            observation[ax1, 2] = self.__actions_until_combat / (self.__actions_per_round - 1)
        return observation 
        
    def observe_champion(self, champ: SimpleTFTChampion) -> np.array:
        """
        Observe the state of a champion.
        
        :param champ: The champion to observe.
        :return: An array representing the champion's state.
        """
        observation = np.zeros(self.observation_shape[-1])
        if champ:
            observation[champ.team] = 1
            ax = self.__num_teams
            observation[ax + champ.preferred_position] = 1
            ax += self.__board_size
            observation[ax + champ.level] = 1
        return observation 
    
    def _log_player_states(self):  
        """
        Log the states of all players.
 
        This method goes through each player, collects their state logs, and appends them to the game's log.
        """
        for p, player in self.__players.items():
            try:
                player_dump = player.dump_log()  # Assuming 'dump_log' is a method in SimpleTFTPlayer class
                self.__log.extend([f"{p}: " + l for l in player_dump])
            except AttributeError as e:
                print(f"Error while dumping log for player {p}: {e}")
    
    def _dump_logs(self):
        # Ensure the log file path is set
        if not self.__log_file_path:
            print("Log file path is not set. Cannot log matchup.")
            return

        # Writing to the log file
        try:
            with open(self.__log_file_path, 'a') as file:
                for line in self.__log:
                    file.write(line + "\n")
        except IOError as e:
            print(f"Failed to write to log file: {e}")
        finally:
            self.__log = [] # Reset the log regardless of success or failure

