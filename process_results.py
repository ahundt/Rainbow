import pandas as pd
import numpy as np
import torch
import os

for exp in os.listdir('results'):
  if os.path.isfile(exp): continue
  processed_path = os.path.join('results', exp, 'results.csv')
  if os.path.exists(processed_path): continue
  if not os.path.exists(os.path.join('results', exp, 'best_metrics.pth')): continue

  # put into dataframe to process
  #results = pd.DataFrame.from_dict(torch.load(os.path.join('results', exp, 'best_metrics.pth')))
  results = pd.DataFrame(torch.load(os.path.join('results', exp, 'best_metrics.pth'))['rewards'], columns=['rewards'])

  #results = results.drop(columns=['best_avg_reward', 'Qs'])

  # get min, max, mean, std of reward, trial success rate (0 reward means failure)
  results['min_reward'] = results['rewards'].apply(lambda x: np.min(x))
  results['max_reward'] = results['rewards'].apply(lambda x: np.max(x))
  results['mean_reward'] = results['rewards'].apply(lambda x: np.mean(x))
  results['reward_std'] = results['rewards'].apply(lambda x: np.std(x))
  results['trial_success_rate'] = results['rewards'].apply(lambda x: np.mean(np.array(x) != 0))
  results = results.drop(columns='rewards')
  results.to_csv(processed_path)
