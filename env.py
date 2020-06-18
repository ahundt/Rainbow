# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

import gym
from gym_minigrid.minigrid import Grid
import numpy as np

from custom_envs import *

class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()

class MinigridEnv():
  def __init__(self, args):
    self.env_name = args.env

    if args.progress_reward:
      self.env = LavaCrossingSpotRewardEnv(args.action_reward_penalty)
      self.progress_reward = True
    else:
      self.env = gym.make(args.env).unwrapped
      self.progress_reward = False

    self.img_size = 84

    self.resize = T.Compose([T.ToPILImage(), T.Resize(self.img_size, interpolation=Image.CUBIC),
        T.Grayscale(), T.ToTensor()])

    self.actions = len(self.env.actions) - 1 # don't allow done action
    self.device = args.device
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = False

    self.optimal_steps = self._get_optimal_steps()

  def _get_obs_rgb(self, obs):
    obs = obs.transpose((2, 0, 1))
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = torch.from_numpy(obs)
    obs = self.resize(obs).to(self.device)
    return obs

  def _get_state(self):
    state = self._get_obs_rgb(self.env.render(mode=None))
    return state

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(self.img_size, self.img_size, device=self.device))

  def get_allowed_mask(self):
    allowed = np.zeros(self.actions)
    # TODO modify this initialization depending on the selected task
    # set left, right, forward to allowed, all others not allowed for now
    allowed[:3] = 1

    # check if forward is ok
    grid = self.env.grid
    next_pos = self.env.front_pos
    
    next_pos_obj = grid.get(next_pos[0], next_pos[1])
    if next_pos_obj is not None:
      # don't allow moving forward into lava or wall
      if next_pos_obj.type == 'lava' or next_pos_obj.type == 'wall':
        allowed[2] = 0

    return allowed

  def reset(self):
    self._reset_buffer()
    self.env.reset()
    obs = self._get_state()
    self.state_buffer.append(obs[0])
    self.optimal_steps = self._get_optimal_steps()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action, agent_pos=None):
    # figure out actions
    if self.progress_reward:
      _, reward, done = self.env.step(action, self.agent_pos())
    else:
      _, reward, done, _ = self.env.step(action)

    if not self.training:
      # assign action efficiency reward if we successfully reached the goal, otherwise, reward is 0
      if done and reward != 0:
        reward = self.optimal_steps / self.env.step_count
      else:
        reward = 0

    obs = self._get_state()
    self.state_buffer.append(obs[0])

    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  def train(self):
    self.training = True
    if self.progress_reward:
      self.env.train()

  def eval(self):
    self.training = False
    if self.progress_reward:
      self.env.eval()

  def action_space(self):
    return self.actions

  def agent_pos(self):
    return self.env.agent_pos

  def _get_optimal_steps(self):
    # for now, raise an error if the env isn't 9x9 lava crossing with 1 gap
    if self.env_name != 'MiniGrid-LavaCrossingS9N1-v0':
      raise ValueError("Action Efficiency only implemented for LavaCrossings9N1")

    # iterate through grid and find crossing position
    cross_pos = (0, 0)
    grid = np.array(self.env.grid.grid.copy())

    # 0 for vertical, 1 for horizontal
    lava_orientation = None
    lava_ref_spot = None
    for ind, i in enumerate(grid):
      if i is None:
        grid[ind] = 0
        r = ind // self.env.grid.height
        c = ind - (r * self.env.grid.height)
        # vertical lava strip and same column as lava
        if lava_orientation == 0 and lava_ref_spot[1] == c:
          cross_pos = (r, c)
        # horizontal lava strip and same row as lava
        elif lava_orientation == 1 and lava_ref_spot[0] == r:
          cross_pos = (r, c)

      elif i.type == 'lava':
        grid[ind] = 1
        if lava_orientation is None:
          # figure out the orientation and set the reference point (to calculate crossing point)
          lava_orientation = (grid[ind + 1] is not None and grid[ind + 1].type == 'lava') or \
              (grid[ind + 2] is not None and grid[ind + 2].type == 'lava')
          lava_ref_spot = (ind // self.env.grid.height, ind - (r * self.env.grid.height))
      elif i.type == 'wall':
        grid[ind] = 1
      elif i.type == 'goal':
        # we are at the goal here, assign it a value of 2
        grid[ind] = 2
        # calculate row and column of goal, store this info
        r = ind // self.env.grid.height
        c = ind - (r * self.env.grid.height)

    # now calculate optimal path length
    base_len = 14 # 6 steps, turn, 6 more steps
    agent_dir = self.env.dir_vec
    grid_np = grid.reshape([self.env.grid.height, self.env.grid.width])
    
    # if the agent is pointing at the crossing, no extra turns
    if agent_dir[1] == 1 and cross_pos[1] == 1:
      return base_len
    if agent_dir[0] == 1 and cross_pos[0] == 1:
      return base_len

    # otherwise, if the agent is pointing at lava or at a wall, add 2 extra turn(s)
    if grid_np[1 + agent_dir[::-1][0], 1 + agent_dir[::-1][1]] == 1:
      # if we need 2 turns to even move forward, add 1 extra turn here
      # check each possible position attained after a single turn
      if grid_np[1 + agent_dir[0], 1 + agent_dir[1]] and grid_np[1 - agent_dir[0], 1 - agent_dir[1]] == 1:
        return base_len + 3

      return base_len + 2

    # otherwise, we can move forward from the start, and add 1 extra step for the crossing turn
    return base_len + 1

  def render(self):
    state = self._get_state().cpu().numpy()[0]
    cv2.imshow('screen', state)
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
