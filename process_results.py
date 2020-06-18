import pandas as pd
import numpy as np
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Generating Metrics')
parser.add_argument('-r', '--reprocess', action='store_true', default=False,
    help='regenerate results csvs if they already exist')
args = parser.parse_args()

for exp in os.listdir('results'):
  if os.path.isfile(exp): continue
  # results.csv is the processed version of training log
  processed_path = os.path.join('results', exp, 'results.csv')
  # reevaluation_results.csv is the processed version of best model re-evaluated on 30 envs
  eval_processed_path = os.path.join('results', exp, 'reevaluation_results.csv')

  # process metrics.pth
  if args.reprocess or not os.path.exists(processed_path):
    if not os.path.exists(os.path.join('results', exp, 'metrics.pth')): continue
    results = pd.DataFrame.from_dict(torch.load(os.path.join('results', exp, 'metrics.pth')))
    # get min, max, mean, std of reward, trial success rate (0 reward means failure)
    results = results.drop(columns=['best_avg_reward', 'Qs'])
    results['min_reward'] = results['rewards'].apply(lambda x: np.min(x))
    results['max_reward'] = results['rewards'].apply(lambda x: np.max(x))
    results['mean_reward'] = results['rewards'].apply(lambda x: np.mean(x))
    results['reward_std'] = results['rewards'].apply(lambda x: np.std(x))
    # TODO hacky, trial is successful if we took less than max steps (100 steps)
    results['trial_success_rate'] = results['rewards'].apply(lambda x: np.mean(np.array(x) > 0.16))
    results = results.drop(columns='rewards')
    results.to_csv(processed_path)
    print("Wrote to", processed_path)

  if args.reprocess or not os.path.exists(eval_processed_path):
    if not os.path.exists(os.path.join('results', exp, 'best_metrics.pth')): continue
    collated_results = pd.DataFrame(columns=['min_reward', 'max_reward', 'mean_reward',
      'reward_std', 'trial_success'])
    results = torch.load(os.path.join('results', exp, 'best_metrics.pth'))
    rewards = results['rewards']
    collated_results['min_reward'] = np.min(rewards)
    collated_results['max_reward'] = np.max(rewards)
    collated_results['mean_reward'] = np.mean(rewards)
    collated_results['reward_std'] = np.std(rewards)
    if 'trial_completion' in results:
      collated_results['trial_success'] = results['trial_completion']
    else:
      # TODO hacky
      collated_results['trial_success'] = np.mean((np.array(rewards) > 0.16).astype(int))

    collated_results.to_csv(eval_processed_path)
    print("Wrote to", eval_processed_path)
