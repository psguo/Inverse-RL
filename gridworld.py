#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# Gridworld class:
# Fashioned as a simplified version of OpenAI gym frozen lake environment

import numpy as np
import environment as env

MAPS = {
  "4x4": [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
  ],
  "8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
  ],
}

def generate_default_rewards(desc):
  # Default rewards given a description:
  # 1 for G, -1 for H and 0 for everything else
  desc = np.asarray(desc, dtype='c')
  nrow, ncol = desc.shape
  R = np.zeros(nrow * ncol)

  gstate = np.array(desc == b'G').astype('float64').ravel().nonzero()[0]
  R[gstate] = 1
  hstate = np.array(desc == b'H').astype('float64').ravel().nonzero()[0]
  R[hstate] = -1

  return R

class GridWorld(env.DiscreteEnvironment):

  def __init__(self, desc=None, map_name="8x8", R=None):
    # desc refers to grid structure, like the values in MAPS above.
    # map_name is either "4x4" or "8x8"
    # R is a reward array with a reward for each state.
    # If None, default rewards are generated.

    if desc is None and map_name is None:
      raise ValueError("Must provide either desc or map_name")
    elif desc is None:
      desc = MAPS[map_name]

    self.desc = desc = np.asarray(desc, dtype='c')
    self.nrow, self.ncol = nrow, ncol = desc.shape

    nA = 4
    nS = nrow * ncol

    # Assuming only one init state
    init_state = np.array(
        desc == b'S').astype('float64').ravel().nonzero()[0][0]

    if R is None:
      R = generate_default_rewards(desc)
    self.R = R

    # Creating the transitions.
    P = {s : {a : [] for a in range(nA)} for s in range(nS)}

    # Unwrap r, c indices
    def to_s(row, col):
      return row * ncol + col

    # Account for action at edge of grid
    def inc(row, col, a):
      if a == 0: # left
        col = max(col - 1,0)
      elif a == 1: # down
        row = min(row + 1, nrow - 1)
      elif a == 2: # right
        col = min(col + 1, ncol - 1)
      elif a == 3: # up
        row = max(row - 1, 0)
      return (row, col)

    for row in range(nrow):
      for col in range(ncol):
        s = to_s(row, col)
        for a in range(4):
          transitions = P[s][a]
          stype = desc[row, col]
          # Check if terminal state (goal).
          # Once you are in this state, you cannot move out.
          # So all actions lead back to this same state and give no reward.
          if stype in b'G':
            transitions.append((1.0, s, 0, True))
          else:
            # Transition is uncertain
            for b in [(a-1)%4, a, (a+1)%4]:
              new_row, new_col = inc(row, col, b)
              new_state = to_s(new_row, new_col)
              new_stype = desc[new_row, new_col]
              done = bytes(new_stype) in b'G'
              # You have 50% chance of success in the action and 25% chance of
              # going either right or left of the action.
              p = 1.0/2.0 if b == a else 1.0/4.0
              transitions.append((p, new_state, R[new_state], done))

        super(GridWorld, self).__init__(nS, nA, P, init_state)