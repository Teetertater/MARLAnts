import numpy as np

import gym
from gym import spaces


# CONSTANTS
NUM_AGENTS = 2

FPS = 2
DISPLAY_SIZE = 512
CELL_SIZE = 32
NUM_CELLS = DISPLAY_SIZE//CELL_SIZE
TARGET_TOWER_HEIGHT = 5
MAX_AGENT_BLOCKS = 2


action_list = ["walk", "place"]


def xy_to_coord(x, y):
    return x*CELL_SIZE, DISPLAY_SIZE - (y + 1)*CELL_SIZE


def adjacent_tower_size(x, dir, block_grid):
    return int(sum(block_grid[x+1].T)) if dir else int(sum(block_grid[x-1].T))


class Agent:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.direction = 1  # 0 = Left, 1 = right
        self.holding_blocks = 2  # TARGET_TOWER_HEIGHT - 1

    def moveRight(self):
        self.x += 1

    def moveLeft(self):
        self.x -= 1

    def fall(self):
        self.y -= 1


class TowerEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'console'], 'fps': FPS}

    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects
        self.num_agents = NUM_AGENTS
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete((TARGET_TOWER_HEIGHT+1) * MAX_AGENT_BLOCKS+1)
        self.observation_space = spaces.Dict({"adjacent_tower": spaces.Discrete(TARGET_TOWER_HEIGHT+1),
                                              "holding_blocks": spaces.Discrete(MAX_AGENT_BLOCKS+1)})

        self.windowWidth, self.windowHeight = DISPLAY_SIZE, DISPLAY_SIZE

        self.reset()

    def _get_game_state(self):
        state = [{}]*NUM_AGENTS
        for i, agent in enumerate(self.agents):
            tower_size = adjacent_tower_size(agent.x, agent.direction, self.block_grid)
            state[i]["adjacent_tower"] = tower_size
            state[i]["holding_blocks"] = agent.holding_blocks
        return state

    def reset(self):
        # Reset the state of the environment to an initial state
        # Resets blocks and builds foundation for one tower
        self.block_grid = np.zeros((NUM_CELLS + 2, NUM_CELLS))  # +2 to simplify adjacency code (like padding)
        self.block_grid[np.random.randint(0, NUM_CELLS)][0] = 1

        self.agents = [Agent(i*2) for i in range(self.num_agents)]

        self.screen = None
        return self._get_game_state()

    def render(self, fps, mode='human', close=False):
        # Render the environment to the screen
        if mode == 'console':
            print(self._get_game_state)
        elif mode == "human":
            # TODO : Move imports to init() and abstract mode
            import pygame
            from pygame import freetype

            pygame.font.init()
            pygame.freetype.init()
            printfont = pygame.freetype.Font(None, 18)

            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    self.screen = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
                clock = pygame.time.Clock()
                self.screen.fill((0, 0, 0))

            id = self.block_grid.shape
            for i in range(id[0]):
                for j in range(id[1]):
                    if self.block_grid[i][j]:
                        pygame.draw.rect(self.screen, (0, 0, 255), (*xy_to_coord(i, j), CELL_SIZE, CELL_SIZE))
            for agent in self.agents:
                rx, ry = xy_to_coord(agent.x, agent.y)
                pygame.draw.rect(self.screen, (0, 255, 255), (rx, ry, CELL_SIZE, CELL_SIZE))
                printfont.render_to(self.screen, (rx+CELL_SIZE//2, ry+CELL_SIZE//2),
                                    str(agent.holding_blocks), (0, 10, 10))
                pygame.display.flip()
            tick = fps if fps != None else self.metadata["fps"]
            clock.tick(tick)

    def step(self, actions):
        # Execute one time step within the environment
        reward = 0
        tower_heights_after = [0]*self.num_agents
        for i, agent in enumerate(self.agents):
            action = actions[i]
            hit_left_wall = (agent.x + 1 == DISPLAY_SIZE/CELL_SIZE and agent.direction)
            hit_right_wall = (agent.x == 0 and not agent.direction)

            if hit_left_wall or hit_right_wall:
                agent.direction ^= 1

            if action not in [0, 1]:
                raise Exception("Invalid action: {}".format(action))

            tower_height_before = adjacent_tower_size(agent.x, agent.direction, self.block_grid)

            if action_list[action] == "walk":
                if agent.direction:
                    agent.moveRight()
                else:
                    agent.moveLeft()

            if action_list[action] == "place" and agent.holding_blocks > 0:
                agent.holding_blocks -= 1
                if agent.direction:
                    block_ind = int(sum(self.block_grid[agent.x+1]))
                    self.block_grid[agent.x+1][block_ind] = 1
                else:
                    block_ind = int(sum(self.block_grid[agent.x-1]))
                    self.block_grid[agent.x-1][block_ind] = 1

            tower_heights_after[i] = adjacent_tower_size(agent.x, agent.direction, self.block_grid)
            reward += self._get_reward(action, agent.holding_blocks, tower_height_before, tower_heights_after[i])

            hit_left_wall = (agent.x + 1 == DISPLAY_SIZE/CELL_SIZE and agent.direction)
            hit_right_wall = (agent.x == 0 and not agent.direction)
            if hit_left_wall or hit_right_wall:
                agent.direction ^= 1

        result, done = self._is_over()

        return self._get_game_state(), reward, done, result

    def _get_reward(self, holding_blocks, action, t1, t2):
        if action == 0:
            return -1/self.num_agents
        if t1 == 0:
            return -20/self.num_agents
        if t2 == TARGET_TOWER_HEIGHT:
            return 50/self.num_agents
        if t2 > t1:
            return 5/self.num_agents
        if holding_blocks < 1:
            return -20/self.num_agents
        if t2 == t1:
            raise Exception
        else:
            return -1/self.num_agents

    def _is_over(self):
        tower_built = max(sum(self.block_grid.T)) == TARGET_TOWER_HEIGHT
        if tower_built:
            return "win", True
        elif all([agent.holding_blocks == 0 for agent in self.agents]):
            return "lose", True
        else:
            return "", False
