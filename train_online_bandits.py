import argparse
from policy import get_policy
from logger import HistoryLogger, get_logger
from bandits import get_bandit
from utils import get_config_for_dataset

policies = ['greedy']
parser = argparse.ArgumentParser(description='Train CMAB policy on selected dataset')
parser.add_argument('dataset', type=str, choices=['mushroom', 'ecoli', 'mnist', 'shuttle'], help='Dataset name')
args = parser.parse_args()

for p in policies:
    config = get_config_for_dataset(p, args.dataset)
    bandit = get_bandit(args.dataset)
    policy = get_policy(p, bandit.arms_count(), bandit.context_size(), config) 
    logger = get_logger("wandb", policy.get_name(), args.dataset + '|bandits' )
    config['policy'] = policy.get_name()
    logger.log_config(config)

    history = HistoryLogger(policy.get_name(), args.dataset)    
    regret_sum = 0
    for _ in range(max(50000, len(bandit))):
        context = bandit.get_context(1)
        action = policy.get_action(context)
        reward, regret = bandit.pull_arm(action)
        regret_sum += regret
        policy.update(context, action, reward)
        logger.log({ "reward": reward, "regret": regret_sum })
        history.log(context, action, reward)
    del logger
