import argparse
from policy import get_policy
from logger import get_logger
from bandits import get_bandit
from offline_bandits import ReplyOfflineBandit 
from utils import get_config_for_dataset, get_history_logs_for_dataset

policies = ['neural_ucb']
parser = argparse.ArgumentParser(description='Train CMAB policy on selected dataset')
parser.add_argument('dataset', type=str, choices=['mushroom', 'ecoli', 'mnist'], help='Dataset name')
args = parser.parse_args()

history_logs = get_history_logs_for_dataset(args.dataset)
for p in policies:
    i = 0
    for log in history_logs:
        i+=1
        if i < 27:
            continue
        reply_bandit = ReplyOfflineBandit(log['filename'])
        logger = get_logger("wandb", p + " H:" + log['offline_policy'], project_name=args.dataset + '|test')
        config = get_config_for_dataset(p, args.dataset)
        policy = get_policy(p, reply_bandit.arms_count(), reply_bandit.context_size(), config)
        
        config['policy'] = policy.get_name()
        config['offline_policy'] = log['offline_policy']
        logger.log_config(config)
        
        history = reply_bandit.get_dataset()
        policy.pretrain(history, logger)
        del logger

        policy.reset()
        bandit = get_bandit(args.dataset)
        logger = get_logger("wandb", policy.get_name(), args.dataset + '|compare_pretrained')
        logger.log_config(config)
        regret_sum = 0
        for _ in range(50000):
            context = bandit.get_context(1)
            action = policy.get_action(context)
            reward, regret = bandit.pull_arm(action)
            regret_sum += regret
            policy.update(context, action, reward)
            logger.log({ "regret": regret_sum })
        del logger