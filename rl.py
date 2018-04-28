#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np
import gridworld


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.1
  This implements value iteration for learning a policy given an environment.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """

  value_func = np.zeros(env.nS)
  itr = 0
  delta = tol
  while delta >= tol and itr < max_iterations:
    delta = 0
    Vs = np.ones(env.nS) * (-np.inf)
    for s in range(env.nS):
      v = value_func[s]
      for a in range(env.nA):
        action_value = 0
        for next_state in env.P[s][a]:
          action_value += next_state[0] * (next_state[2] + gamma * value_func[next_state[1]])
        if Vs[s] < action_value:
          Vs[s] = action_value
      delta = max(delta, abs(v - Vs[s]))
    value_func = Vs.copy()
    itr += 1

  return value_func, itr


def policy_from_value_function(env, value_function, gamma):
  """
  Q3.2.1/Q3.2.2
  This generates a policy given a value function.
  Useful for generating a policy given an optimal value function from value
  iteration.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    value_function: numpy.ndarray
      Optimal value function array of length nS
    gamma: float
      Discount factor, must be in range [0, 1)

  Output:
    numpy.ndarray
    policy: Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
  """
  policy = np.zeros(env.nS, dtype='int')
  action_values = np.zeros(env.nA)
  for s in range(env.nS):
    for a in range(env.nA):
      next_values = [possible[0] * (possible[2] + gamma * value_function[possible[1]]) for possible in
                     env.P[s][a]]
      action_values[a] = np.sum(next_values)
    policy[s] = np.argmax(action_values)
  return policy


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
  """Performs policy evaluation.

  Evaluates the value of a given policy.

  Parameters
  ----------
  env: gym.core.Environment
    The environment to compute value iteration for. Must have nS,
    nA, and P as attributes.
  gamma: float
    Discount factor, must be in range [0, 1)
  policy: np.array
    The policy to evaluate. Maps states to actions.
  max_iterations: int
    The maximum number of iterations to run before stopping.
  tol: float
    Determines when value function has converged.

  Returns
  -------
  np.ndarray, int
    The value for the given policy and the number of iterations till
    the value function converged.
  """
  value_func = np.zeros(env.nS)
  itr = 0
  delta = tol
  while delta >= tol and itr < max_iterations:
    delta = 0
    Vs = np.zeros(env.nS)
    for s in range(env.nS):
      v = value_func[s]
      a = policy[s]
      next_possibles = env.P[s][a]
      Vs[s] = 0
      for next in next_possibles:
        Vs[s] += next[0] * (next[2] + gamma * value_func[next[1]])
      delta = max(delta, abs(v - Vs[s]))
    value_func = Vs.copy()
    itr += 1
  return value_func, itr


def improve_policy(env, gamma, value_func, policy):
  """Performs policy improvement.

  Given a policy and value function, improves the policy.

  Parameters
  ----------
  env: gym.core.Environment
    The environment to compute value iteration for. Must have nS,
    nA, and P as attributes.
  gamma: float
    Discount factor, must be in range [0, 1)
  value_func: np.ndarray
    Value function for the given policy.
  policy: dict or np.array
    The policy to improve. Maps states to actions.

  Returns
  -------
  bool, np.ndarray
    Returns true if policy changed. Also returns the new policy.
  """
  policy_changed = False
  for s in range(env.nS):
    old_action = policy[s]

    # action_values = np.zeros(env.nA)
    # for a in range(env.nA):
    #     next_values = [possible[0] * (possible[2] + gamma * value_func[possible[1]]) for possible in
    #                    env.P[s][a]]
    #     action_values[a] = np.sum(next_values)
    # policy[s] = np.argmax(action_values)

    best_action = 0
    best_action_value = -np.inf
    for a in range(env.nA):
      action_value = 0
      for next_state in env.P[s][a]:
        action_value += next_state[0] * (next_state[2] + gamma * value_func[next_state[1]])
      if best_action_value < action_value:
        best_action = a
        best_action_value = action_value
    policy[s] = best_action

    if old_action != policy[s]:
      policy_changed = True
  return policy_changed, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.2: BONUS
  This implements policy iteration for learning a policy given an environment.

  You should potentially implement two functions "evaluate_policy" and
  "improve_policy" which are called as subroutines for this.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """
  policy = np.zeros(env.nS, dtype='int')
  value_func = np.zeros(env.nS)
  itr = 0
  eval_itr_total = 0
  for itr in range(max_iterations):
    value_func, eval_itr_cur = evaluate_policy(env, gamma, policy, max_iterations, tol)
    eval_itr_total += eval_itr_cur
    isPolicyChanged, policy = improve_policy(env, gamma, value_func, policy)
    if not isPolicyChanged:
      break
  return policy, value_func, itr + 1, eval_itr_total


def td_zero(env, gamma, policy, alpha):
  """
  Q3.2.2
  This implements TD(0) for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """

  value_func = np.zeros(env.nS)
  max_iterations = 10000
  itr = 0
  while itr < max_iterations:
    s = env.reset()
    done = False
    while not done:
      action = policy[s]
      s_, r, done, info = env.step(action)
      value_func[s] = value_func[s] + alpha * (r + gamma * value_func[s_] - value_func[s])
      s = s_
    itr += 1
  return value_func


def n_step_td(env, gamma, policy, alpha, n):
  """
  Q3.2.4: BONUS
  This implements n-step TD for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    n: int
      Number of future steps for calculating the return from a state.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """

  value_func = np.zeros(env.nS)
  max_iterations = 10000
  itr = 0
  while itr < max_iterations:
    s = env.reset()
    done = False
    rewards = []
    states = []
    while not done:
      action = policy[s]
      s_, r, done, info = env.step(action)
      states.append(s)
      rewards.append(r)
      s = s_

    T = len(states)
    G = np.zeros(T)
    for t in range(T):
      if t + n < T:
        s = states[t]
        V_end = value_func[s]
      else:
        V_end = 0
      discounted_rewards = np.zeros((n,))
      for k in range(n):
        if t + k < T:
          discounted_rewards[k] = np.power(gamma, k) * rewards[t + k]

      G[t] = (gamma ** n) * V_end + np.sum(discounted_rewards)

    for t in range(T):
      s = states[t]
      value_func[s] = value_func[s] + alpha * (G[t] - value_func[s])

    itr += 1
  return value_func


def print_values(values, shape):
  values = values.reshape(shape)
  for i in range(shape[0]):
    if values[i][0] > 0:
      row_str = ' '
    else:
      row_str = ''
    for j in range(shape[1]):
      if values[i][j] >= 0:
        row_str += "{} "
      row_str += "%.2f" % values[i][j] + ' '
    print('    ' + row_str + '\\\\')


def print_policy(policy, shape, action_names):
  policy = policy.reshape(shape)
  for i in range(shape[0]):
    row_str = ''
    for j in range(shape[1]):
      row_str += (action_names[policy[i][j]]) + ' '
    print('    ' + row_str + '\\\\')


if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')

  # Play around with these values if you want!
  gamma = 0.9
  alpha = 0.05
  n = 10

  shape = (8, 8)
  action_names = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}

  # Q3.2.1
  V_vi, n_iter = value_iteration(env, gamma)
  policy = policy_from_value_function(env, V_vi, gamma)
  # print("Itr: ", n_iter)
  # print_policy(policy, shape, action_names)
  # print_values(V_vi, shape)

  # Q3.2.2: BONUS
  # policy, V_pi, n_iter, eval_itr_total = policy_iteration(env, gamma)
  # print("Itr: ", n_iter)
  # print_policy(policy, shape, action_names)
  # print_values(V_pi, shape)

  # Q3.2.3
  # V_td = td_zero(env, gamma, policy, alpha)
  # print_values(V_td, shape)

  # Q3.2.4: BONUS
  V_ntd = n_step_td(env, gamma, policy, alpha, n)
  print_values(V_ntd, shape)
