# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
import numpy as np
import torch
import copy

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal', 'allowed_actions'))
# TODO fix this to be general, right now allow turning left/right or moving forward by default
default_allowed = np.array([1, 1, 1, 0, 0, 0])
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False, default_allowed)

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size  # The actual overall capacity, total number of spots available for data storage over all time.
    self.full = False  # Used to track if we have reached the actual capacity
    # The midpoint of the tree is where the leaf data starts, which corresponds with the the first entry of self.data
    # Recall that your basic array based binary tree data structure has 2*size-1 elements, 
    # so the first half is part of the tree and the second half is a contiguous array of leaf values.
    self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
    self.data = np.array([None] * size)  # Wrap-around cyclic buffer of data
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    while parent != 0:
      left, right = 2 * parent + 1, 2 * parent + 2
      self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
      parent = (parent - 1) // 2
    # fill out final top level entry
    self.sum_tree[0] = self.sum_tree[1] + self.sum_tree[2]

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # number of spots actually utilized in the sum tree, which stores priorities. This will always be at least half full based on the size of storage.
  def _sum_tree_leaves_end(self):
     # TODO(ahundt) check for off by 1 error, particularly where this is called
    #  return len(self.sum_tree) if self.full else self.index + self.size - 1
     return len(self.sum_tree) if self.full else self.index + (self.size - 1)
  
  def _tree_indices_to_data_indices(self, tree_indices):
    return tree_indices - (self.size - 1)
  
  def _data_indices_to_tree_indices(self, data_indices):
    return data_indices + (self.size - 1)
  
  def _sum_tree_leaves_start_idx(self):
    return self._data_indices_to_tree_indices(0)

  # number of spots actually utilized in the cyclic data buffer
  def _data_occupancy(self):
    return self.size if self.full else self.index

  # Searches for the location of a value in sum tree
  def _retrieve(self, indices, values):
    left, right = 2 * indices + 1, 2 * indices + 2
    tree_len = self._sum_tree_leaves_end()
    left_val = self.sum_tree[left]
    while np.min(left) < tree_len:
      indices[values <= left_val] = left
      right_is_less = right < tree_len
      indices[right_is_less] = right
      values[right_is_less] -= left_val
      left, right = 2 * indices + 1, 2 * indices + 2
    return indices

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, values):
    tree_indices = self._retrieve(np.zeros_like(values), values)  # Search for index of item from root
    data_indices = self._data_indices_to_tree_indices(tree_indices)
    return (self.sum_tree[tree_indices], data_indices, tree_indices)  # Return value, data index, tree index
  
  def sample(self, batch_size, history_length, num_future_steps):
    data_occupancy = self._data_occupancy()
    all_data_idxs = np.arange(data_occupancy)
    # get the sum tree values which are actually valid, and 
    # TODO(ahundt) check for off by 1 error when extracting all probabilities
    all_probs = self.sum_tree[self._sum_tree_leaves_start_idx():self._sum_tree_leaves_end()]
    # only sample from entries where we can actually get useful data
    invalid_indices = np.logical_not(np.logical_and((all_data_idxs - self.index) % self.size >= history_length, (self.index - all_data_idxs) % self.size > num_future_steps))
    all_probs[invalid_indices] = 0.0
    # normalize all probs to create sampling probabilities
    # all_probs /= self.total()
    all_probs /= np.sum(all_probs)
    # print("prob sum: " + str(sum(all_probs)))
    # The sum of all probabilities must be 1 to sample
    # NOTE: if np.random.choice throws a ValueError there is a data structure bug!
    data_indices = np.random.choice(data_occupancy, batch_size, p=all_probs)
    tree_indices = self._data_indices_to_tree_indices(data_indices)
    tree_values = self.sum_tree[tree_indices]
    return (tree_values, data_indices, tree_indices)  # Return value, data index, tree index


  # Returns data given a data index
  def get(self, data_index):
    data_idxs = data_index % self.size
    return self.data[data_idxs]

  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight # Initial importance sampling weight β, annealed to 1 over course of training
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
    self.progress_reward = args.progress_reward
    # TODO change names
    self.spot_trial_reward = args.trial_reward

  # Adds state, allowed actions, and selected action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal, allowed_actions):
    state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    self.transitions.append(Transition(self.t, state, action, reward, not terminal, allowed_actions), self.transitions.max)  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  # Returns a transition with blank states where appropriate
  def _get_transition(self, idxs):
    batch_size =  len(idxs)
    transition_batch = np.array([[None] * (self.history + self.n)] * batch_size)
    base_transition_batch = self.transitions.get(idxs)
    # transition_batch[:, self.history - 1] = 
    # TODO(ahundt) vectorize
    for i in range(batch_size):
      transition_batch[i, self.history - 1] = base_transition_batch[i]
      for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
        if transition_batch[i, t + 1].timestep == 0:
          transition_batch[i,t] = blank_trans  # If future frame has timestep 0
        else:
          transition_batch[i,t] = self.transitions.get(idxs[i] - self.history + 1 + t)
      for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
        if transition_batch[i,t - 1].nonterminal:
          transition_batch[i,t] = self.transitions.get(idxs[i] - self.history + 1 + t)
        else:
          transition_batch[i,t] = blank_trans  # If prev (next) frame is terminal
    return transition_batch

  # Returns a valid partially prepared sample batch from a segment
  def _get_samples_from_transitions_segment_tree(self, batch_size):
    normalized_probs, transition_data_idxs, priority_tree_idxs = self.transitions.sample(batch_size, self.history, self.n)
    # Retrieve all required transition data (from t - h to t + n)
    transition_batch = self._get_transition(transition_data_idxs)
    # Create un-discretised state and nth next state
    batch_elements = np.arange(batch_size)
    states = [torch.stack([trans.state for trans in transition_batch[i, :self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255) for i in batch_elements]
    next_states = [torch.stack([trans.state for trans in transition_batch[i, self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255) for i in batch_elements]
    # allowed actions
    allowed_actions = [transition_batch[i,self.history - 1].allowed_actions for i in batch_elements]
    # Discrete action to be used as index
    actions = [torch.tensor([transition_batch[i, self.history - 1].action], dtype=torch.int64, device=self.device) for i in batch_elements]
    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    # If you want to modify the reward function add a config argument and change both this line and agent.py Agent.learn() where Tz is set
    R = [torch.tensor([sum(self.discount ** n * transition_batch[i, self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device) for i in batch_elements]
    # Mask for non-terminal nth next states
    nonterminals = [torch.tensor([transition_batch[i, self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device) for i in batch_elements]

    states, next_states = torch.stack(states), torch.stack(next_states)
    # TODO(adit98) check cat vs stack here
    actions, R, nonterminals = torch.cat(actions), torch.cat(R), torch.stack(nonterminals)
    allowed_actions = np.vstack(allowed_actions)
    return normalized_probs, transition_data_idxs, priority_tree_idxs, states, actions, R, next_states, nonterminals, allowed_actions 

  # Returns a valid completely prepared sample batch from memory
  def sample(self, batch_size):
  #   p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
  #   segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    normalized_probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, allowed_actions = self._get_samples_from_transitions_segment_tree(batch_size)  # Get batch of valid samples
    # probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * normalized_probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, allowed_actions, weights

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    # Create stack of states
    state_stack = [None] * self.history
    state_stack[-1] = self.transitions.data[self.current_idx].state
    allowed_actions = self.transitions.data[self.current_idx].allowed_actions
    prev_timestep = self.transitions.data[self.current_idx].timestep
    for t in reversed(range(self.history - 1)):
      if prev_timestep == 0:
        state_stack[t] = blank_trans.state  # If future frame has timestep 0
      else:
        state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
        prev_timestep -= 1

    state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
    self.current_idx += 1
    return state, allowed_actions

  next = __next__  # Alias __next__ for Python 2 compatibility
