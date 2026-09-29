import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class StaircaseEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi"]
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.num_stairs = 25
        self.step_penalty = -1
        self.goal_reward = 100

        # State = current stair number
        # Positions are 0 through num_stairs, inclusive.
        self.observation_space = spaces.Discrete(self.num_stairs + 1)

        # move 1 stair or move 2 stairs
        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode

        self.position = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0

        observation = self.position
        info = {}

        return observation, info


    def step(self, action):
        # TODO:
        # 1. Apply action
        # 2. Update position
        # 3. Later: apply trap behavior
        # 4. Calculate reward
        # 5. Check termination
        if action == 0:
            self.position += 1
        elif action == 1:
            self.position += 2

        self.position = min(self.position, self.num_stairs)

        observation = self.position
        # penalty for each step taken
        reward = self.step_penalty

        # Check if the agent has reached the goal
        reached_goal = self.position == self.num_stairs

        # Reaching the goal is termination
        terminated = reached_goal

        # If the agent has reached the goal, add the goal reward
        if reached_goal:
            reward += self.goal_reward

        # Gymnasium's TimeLimit wrapper sets this to True at the step limit.
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode != "ansi":
            return None

        lines = ["Staircase (A = agent):"]
        for stair in range(self.num_stairs, -1, -1):
            marker = "A" if self.position == stair else " "
            lines.append(f"{'  ' * stair}{marker}[{stair}]")

        return "\n".join(lines)


    def close(self):
        pass


register(
    id="cs272/Staircase-v0",
    entry_point="environment:StaircaseEnv",
    max_episode_steps=50,
)