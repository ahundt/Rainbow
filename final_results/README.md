## Run 1: `action\_mask\_reward.html`
## Command: `python main.py --minigrid --target-update 2000 --T-max 1000000 \
--learn-start 20000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 20 \
--architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
--evaluation-interval 5000 --env MiniGrid-LavaCrossingS9N1-v0 --batch-size 128 \
--action-mask --id lava-large-env-action-mask`

## Run 2: `all\_terms\_support\_reward.html`
## Command: `python main.py --minigrid --target-update 2000 --T-max 1000000 \
--learn-start 20000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 1 \
--architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
--evaluation-interval 5000 --env MiniGrid-LavaCrossingS9N1-v0 --batch-size 128 \
--action-mask --spot-q --progress-reward --id lava-large-env-progress-reward-all-terms-support`

## Code Modifications: Modify line 108 of `agent.py` to 
`Tz = returns.unsqueeze(1) + self.discount * self.support.unsqueeze(0)`

## Run 3: `discounted\_support\_reward.html`
## Command: `python main.py --minigrid --target-update 2000 --T-max 1000000 \
--learn-start 20000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 1 \
--architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
--evaluation-interval 5000 --env MiniGrid-LavaCrossingS9N1-v0 --batch-size 128 \
--action-mask --spot-q --progress-reward --id lava-large-env-progress-reward-discounted-support`

## Code Modifications: Modify line 108 of `agent.py` to 
`Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)`

## Run 4: `double\_goal\_reward.html`
## Command: `python main.py --minigrid --target-update 2000 --T-max 1000000 \
--learn-start 20000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 1 \
--architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
--evaluation-interval 5000 --env MiniGrid-LavaCrossingS9N1-v0 --batch-size 128 \
--action-mask --spot-q --progress-reward --id lava-large-env-progress-reward-double-goal`

## Code Modifications: Modify line 99 of `custom\_envs.py` to `return 2 * self.reward\_grid[c, r]`

## Run 5: `empty\_support\_reward.html`
## Command: `python main.py --minigrid --target-update 2000 --T-max 1000000 \
--learn-start 20000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 1 \
--architecture data-efficient --hidden-size 256 --learning-rate 0.0001 \
--evaluation-interval 5000 --env MiniGrid-LavaCrossingS9N1-v0 --batch-size 128 \
--action-mask --spot-q --progress-reward --id lava-large-env-progress-reward-no-support`

## Code Modifications: Modify line 108 of `agent.py` to 
`Tz = returns.unsqueeze(1) + torch.zeros_like(self.support.unsqueeze(0))`
