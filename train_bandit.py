import argparse
import yaml
from policy import get_policy
from bandits import get_bandit
from logger import HistoryLogger, get_logger
from offline_bandits import get_offline_bandit 

parser = argparse.ArgumentParser(description='Train CMAB policy on selected dataset')
parser.add_argument('dataset', type=str, choices=['mushroom', 'ecoli'], help='Dataset name')
parser.add_argument('policy', type=str, 
                    choices=['random', 'epsilon_greedy', 'lin_ucb', 'neural_ucb', 'bayes_by_backprob'], 
                    help='Policy')
parser.add_argument('config', type=str, help='Policy config')
parser.add_argument('--offline', type=str, default="")
parser.add_argument('--offline_method', type=str, choices=['dm', 'ips', 'dr'], default='dm')
args = parser.parse_args()

with open(args.config, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

bandit = get_bandit(args.dataset)
policy = get_policy(args.policy, bandit.arms_count(), bandit.context_size(), config) 

if not args.offline:
    logger = get_logger("wandb", policy.get_name())
    config['policy'] = args.policy
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
else:
    offline_bandit = get_offline_bandit(args.offline_method, args.offline)
    logger = get_logger("wandb", args.policy + " " + args.offline, project_name='offline_bandits')
    config['policy'] = args.policy
    config['offline_method'] = args.offline_method
    logger.log_config(config)

    for step in range(len(offline_bandit)):
        context = offline_bandit.get_context(1)
        (context)
        action = policy.get_action(context)
        reward = offline_bandit.pull_arm(action)
        policy.update(context, action, reward)
        logger.log({ "reward": reward })

        if step % 250 == 0:
            eval_regret = 0
            for _ in range(len(bandit)):
                context = bandit.get_context(1)
                action = policy.get_action(context)
                reward, regret = bandit.pull_arm(action)
                eval_regret += regret
        
            logger.log({ "eval/regret": eval_regret })
