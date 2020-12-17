# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import numpy as np

from env import Env, MinigridEnv


# Test DQN
def test(args, T, dqn, val_mem, metrics, results_dir, evaluate=False):
  if args.minigrid:
    env = MinigridEnv(args)
    env.seed(args.test_seed)
  else:
    env = Env(args)

  env.eval()
  metrics['steps'].append(T)
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  comp_percentage = 0
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      if args.spot_action_mask:
        if not args.minigrid:
          raise NotImplementedError("Action mask only implemented for minigrid")
        allowed_actions = env.get_allowed_mask()
      
      else:
        allowed_actions = np.ones(env.action_space())

      action = dqn.act_e_greedy(state, allowed_actions)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        if reward_sum != 0: comp_percentage += 1
        T_rewards.append(reward_sum)
        break
  env.close()

  # normalize trial completion percentage
  comp_percentage /= args.evaluation_episodes

  # Test Q-values over validation memory
  for state, allowed_actions in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state, allowed_actions))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    if comp_percentage == 1:
      if metrics['trial_completed'] == 0:
        metrics['trial_completed'] = 1
        dqn.save(results_dir, 'first_completed_model.pth')

      # Save model parameters if improved
      if avg_reward > metrics['best_avg_reward']:
        metrics['best_avg_reward'] = avg_reward
        dqn.save(results_dir, 'best_efficiency_model.pth')

    # Append to results and save metrics
    metrics['rewards'].append(T_rewards)
    metrics['Qs'].append(T_Qs)
    metrics['trial_completion'].append(comp_percentage)
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Plot
    _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
    _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

  else:
    # Append to results and save metrics
    metrics['rewards'] = T_rewards
    metrics['Qs'] = T_Qs
    metrics['trial_completion'] = comp_percentage
    print('wrote results')
    torch.save(metrics, os.path.join(results_dir, 'best_metrics.pth'))
    
  # Return average reward and Q-value
  return avg_reward, avg_Q, comp_percentage

# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
