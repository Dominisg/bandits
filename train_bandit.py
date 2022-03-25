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
parser.add_argument('offline', type=str, optional=True)
parser.add_argument('offline_method', type=str, choices=['dm', 'ips', 'dr'], default='dm', optional=True)
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
    history = HistoryLogger(policy.get_name())
    
    regret_sum = 0
    for _ in range(50000):
        context = bandit.get_context(1)
        action = policy.get_action(context)
        reward, regret = bandit.pull_arm(action)
        regret_sum += regret
        policy.update(context, action, reward)
        logger.log({ "reward": reward, "regret": regret_sum })
        history.log(context, action, reward)
else:
    offline_bandit = get_offline_bandit(args.offline)
    logger = get_logger("dummy", args.policy + " " + args.offline, project_name='offline_bandits')

    for step in range(len(offline_bandit)):
        context = offline_bandit.get_context(1)
        action = policy.get_action(context)
        reward = offline_bandit.pull_arm(action)
        policy.update(context, action, reward)
        logger.log({ "reward": reward })

        if step % 1000 == 0:
            eval_regret = 0
            for _ in range(len(bandit)):
                context = bandit.get_context(1)
                action = policy.get_action(context)
                reward, regret = bandit.pull_arm(action)
                eval_regret += regret
        
            logger.log({ "eval/regret": eval_regret })