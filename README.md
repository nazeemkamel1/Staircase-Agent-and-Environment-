# Staircase

## The environment
Action Space: Discrete(2)
Observation Space: Discrete(26)
import: gymnasium.make("cs272/Staircase-v0")
Staircase involves climbing a set of stairs starting from the bottom with the goal of reaching the top in the fewest number of rounds. The player may not always move the intended number of steps each round due to "traps" that can randomly boost the player forward or knock them backwards. 

## Description
The player is climbing a 25-stair staircase. The starting location [0,0] is the bottom of the staircase and the goal is to reach stair 25, [0,25]. Three stairs—6, 13, and 20—are traps that can give the player a lucky boost or knock them backwards. The player makes moves until they reach the top of the staircase, or run out of rounds. 

## Action Space
The action shape is (1,) in the range {0, 1} indicating how many steps the player makes.
The action space is `Discrete(2)`: action 0 moves up one stair, and action 1 moves up two, skipping a step. Movement is limited to within the 25 steps, anything over 25 places you on the 25th step and anything below places you on the first step.
0: Moves up one stair
1: Moves up two stairs, skipping a step

## Observation Space
The observation is one integer, being the player's current stair. `Discrete(26)` represents positions 0 through 25, inclusive. The number of possible observations is 26, the number of stairs in our staircase (25) and our starting location at the bottom. 

## Starting State
The episode starts with the player in state [0] (location [0,0])

## Rewards
Each step gives a reward of -1. Reaching stair 25 adds 100, so the final step to the goal gives 99.
Reach goal: +100
Each step: -1

## Episode End
Reaching the goal ends the episode (`terminated=True`). If the player has not finished after 50 steps, Gymnasium's time-limit wrapper ends the episode as truncated. 
Termination: The player reaches the top of the staircase.
Truncation (using the time_limit wrapper): The length of the episode is 50

## Information
On landing on a trap, the player has a 70% chance of moving forward 3 stairs and a 30% chance of moving back 4. Trap effects can trigger another trap if they land on one. Random outcomes use Gymnasium's seeded `self.np_random` generator.

## Arguments
The constructor accepts `render_mode`; use `"ansi"` to get a text rendering, or leave it as `None`. The registered environment ID is `cs272/Staircase-v0`, created with `gym.make("cs272/Staircase-v0")`.
