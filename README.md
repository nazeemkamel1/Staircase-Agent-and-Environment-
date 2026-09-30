# Staircase

## The environment

The agent is climbing a 25-stair staircase. It starts at the bottom and wants to reach stair 25. Three stairs—6, 13, and 20—are traps that can give the agent a lucky boost or knock it backward.

The observation is one integer, being the agent's current stair. `Discrete(26)` represents positions 0 through 25, inclusive. The action space is `Discrete(2)`: action 0 moves up one stair, and action 1 moves up two, skipping a step. Movement is limited to within the 25 steps, anything over 25 places you on the 25th step and anything below places you on the first step.

Each step gives a reward of -1. Reaching stair 25 adds 100, so the final step to the goal gives 99. Reaching the goal ends the episode (`terminated=True`). If the agent has not finished after 50 steps, Gymnasium's time-limit wrapper ends the episode as truncated. On landing on a trap, the agent has a 70% chance of moving forward 3 stairs and a 30% chance of moving back 4. Trap effects can trigger another trap if they land on one. Random outcomes use Gymnasium's seeded `self.np_random` generator.

The constructor accepts `render_mode`; use `"ansi"` to get a text rendering, or leave it as `None`. The registered environment ID is `cs272/Staircase-v0`, created with `gym.make("cs272/Staircase-v0")`.