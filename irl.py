#!/usr/bin/python
# 16-831 Spring 2018
# Project 4
# IRL questions:
# Fill in the various functions in this file for Q3.3 on the project.

import numpy as np
import cvxopt as cvx

import gridworld
import rl

def irl_lp(policy, T_probs, gamma, R_max, l1):
  """
  Solves the linear program formulation for finite discrete state IRL.

  Inputs:
    policy: np.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    T_probs: np.ndarray
      nS x nA x nS matrix where:
      T_probs[s][a] is a probability distribution over states of transitioning
      from state s using action a.
      Can be generated using env.generateTransitionMatrices.
    gamma: float
      Discount factor, must be in range [0, 1)
    R_max: float
      Maximum reward allowed.
    l1: float
      L1 regularization penalty.

  Output:
    np.ndarray
    R: Array of rewards for each state.
  """

  T_probs = np.asarray(T_probs)
  nS, nA, _ = T_probs.shape
  nS = int(nS)
  nA = int(nA)

  c = np.zeros([3 * nS])
  c[nS:2 * nS] = l1
  c[2 * nS:] = -1

  G_1 = np.hstack((np.eye(nS), np.zeros((nS,nS)), np.zeros((nS,nS))))
  G_2 = np.hstack((-np.eye(nS), np.zeros((nS,nS)), np.zeros((nS,nS))))
  G_3 = np.hstack((np.eye(nS), -np.eye(nS), np.zeros((nS, nS))))
  G_4 = np.hstack((-np.eye(nS), -np.eye(nS), np.zeros((nS, nS))))

  G_temp = []

  P_a_star = []
  for s in range(nS):
    P_a_star.append(T_probs[s, policy[s], :])
  P_a_star = np.array(P_a_star)
  inv_temp = np.linalg.inv(np.eye(nS) - gamma * P_a_star)

  for s in range(nS):
    a_star = int(policy[s])
    G_s = []
    for a in range(nA):
      if a != a_star:
        G_s.append(-np.matmul(T_probs[s,a_star,:]-T_probs[s,a,:],inv_temp))
    G_temp += G_s

  G_5 = np.hstack((np.array(G_temp), np.zeros((nS * (nA - 1), nS)), np.zeros((nS * (nA - 1), nS))))

  G_B = np.zeros((nS*(nA-1), nS))
  for s in range(nS):
    G_B[(nA-1)*s:(nA-1)*(s+1),s] = 1

  G_6 = np.hstack((np.array(G_temp), np.zeros((nS * (nA - 1), nS)), G_B))

  G = np.vstack((G_1,G_2,G_3,G_4,G_5,G_6))

  h = np.zeros(G.shape[0])
  h[:2*nS] = R_max
  # Create c, G and h in the standard form for cvxopt.
  # Look at the documentation of cvxopt.solvers.lp for further details

  # Don't do this all at once. Create portions of the vectors and matrices for
  # different parts of the objective and constraints and concatenate them
  # together using something like np.r_, np.c_, np.vstack and np.hstack.

  # You shouldn't need to touch this part.
  c = cvx.matrix(c)
  G = cvx.matrix(G)
  h = cvx.matrix(h)
  sol = cvx.solvers.lp(c, G, h)

  R = np.asarray(sol["x"][:nS]).squeeze()

  return R

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')
  shape = (8, 8)
  action_names = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}

  # Generate policy from Q3.2.1
  gamma = 0.9
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)
  # rl.print_policy(policy, shape, action_names)

  T = env.generateTransitionMatrices()

  # Q3.3.5
  # Set R_max and l1 as you want.
  R_max = 1
  l1 = 0.2
  R = irl_lp(policy, T, gamma, R_max, l1)
  print("R: ")
  rl.print_values(R, shape)

  # You can test out your R by re-running VI with your new rewards as follows:
  env_irl = gridworld.GridWorld(map_name='8x8', R=R)
  Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
  policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)

  print("Values: ")
  rl.print_values(Vs_irl, shape)
  rl.print_policy(policy_irl, shape, action_names)