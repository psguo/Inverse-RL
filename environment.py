#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# Environment class:
# Fashioned as a simplified version of OpenAI gym discrete environment

import numpy as np

def discrete_sample(probs):
  # Samples from discrete probability set.
  probs = np.array(probs)
  # CDF from cumulative sum
  cprobs = np.cumsum(probs)

  # Return index of random number in the cumulative distribution.
  # Ignoring last index of cumulative prob since it's always 1.
  return np.searchsorted(cprobs[:-1], np.random.uniform())

class DiscreteEnvironment(object):
  # Abstract class for environment

  def __init__(self, nS, nA, P, initState):
    # Number of states
    self.nS = nS
    self._state_space = range(nS)

    # Number of actions
    self.nA = nA
    self._action_space = range(nA)

    # Specifies, for each state, the transition given the action.
    # P[state][action] = (prob, state, reward, done)
    self.P = P

    # Initial state. Resets to this state.
    self.initState = initState

    self._state = initState
    self._last_action = None

  def reset(self):
    # Reset state
    self._state = self.initState
    self._last_action = None
    return self._state

  def step(self, action):
    # Samples next state given action.
    # Returns new state, reward, flag for done and probability
    # Follows the same structure as OpenAI gym
    transitions = self.P[self._state][action]
    probs = [t[0] for t in transitions]
    idx = discrete_sample(probs)

    p, s, r, d = transitions[idx]

    self._state = s
    self._last_action = action

    return (s, r, d, {"prob" : p})

  def generateTransitionMatrices(self):
    # Generates an nS x nA x nS matrix T where T[s][a] is a probability
    # distribution over states of transitioning from state s using action a.
    T = np.zeros((self.nS, self.nA, self.nS))
    for s in self._state_space:
      for a in self._action_space:
        for tr in self.P[s][a]:
          T[s][a][tr[1]] += tr[0]

    return T