# Simple TFT Environment for Reinforcement Learning

## Introduction

This repository contains a reinforcement learning environment modeled after popular autobattler games. The environment is multi-agent (adversarial) with stochastic elements and incomplete information.

## Environment Description

### Game Rules

The simple TFT game encapsulates the following key mechanics:

- **Board Setup**: Each player has a board with 3 positions to place champions. 
- **Champions**: Players can purchase champions to place on their board. Champions each have a preferred position and teammates that boost their power.
- **Battles**: Players face off against opponents after brief preparation rounds. The outcomes of these matches are determined based on the champions' levels, strategic placement on the board, and the synergies with their teammates.
- **Bench**: Players have a small bench area to hold and manage their collection of champions.
- **Shop**: There is a shop that presents a random selection of champions, drawn from a shared pool, for players to purchase each round.

### Winning Condition

The winner of a combat is based on the number and level of champions on the board, the number of champions sharing a team, and the level of each champion.

## Interface

The environment is interacted with through a dictionary-based interface:

- **Actions**: Actions are passed to the environment's `step` function in the format `{player_name: action}`.
- **Observations**: The state of the environment is returned as a dictionary, offering a comprehensive view of the current game state for each agent.
- **Rewards**: Rewards are provided to guide the agents' learning process, structured in a dictionary format similar to observations.
- **Agent States**: The environment tracks each agent's terminal state (whether they are still active in the game or not) and returns this information as part of the state dictionary.
- **Action Masks**: To facilitate the learning process, the environment also provides action masks indicating valid actions for each agent at any given point in the game.

## Getting Started

### Installation

Provide steps for installing and setting up the environment. This might include cloning the repository, installing dependencies, etc.

```bash
# Example installation steps
git clone https://github.com/Nebraskinator/SimpleTFTEnv
cd SimpleTFTEnv
pip install -r requirements.txt
python test.py
