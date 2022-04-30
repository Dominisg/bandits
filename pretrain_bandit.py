import argparse
from policy import get_policy
from logger import get_logger
from bandits import get_bandit
from offline_bandits import ReplyOfflineBandit 
from utils import get_config_for_dataset, get_history_logs_for_dataset
from multiprocessing import Process

def pretrain_policy(p, history_logs, dataset, pretrain_ucb):
    for log in history_logs:
        reply_bandit = ReplyOfflineBandit(log['filename'])
        logger = get_logger("wandb", p + " H:" + log['offline_policy'], project_name=dataset + '|pretrain_early_stop')
        config = get_config_for_dataset(p, dataset)
        config['device'] = 'cuda'
        policy = get_policy(p, reply_bandit.arms_count(), reply_bandit.context_size(), config)
        
        config['policy'] = policy.get_name()
        config['offline_policy'] = log['offline_policy']
        logger.log_config(config)
        
        history = reply_bandit.get_dataset()
        if pretrain_ucb and 'neural_ucb' in config['policy']:
            config['policy'] += ' ucb_pretrained'
            policy.pretrain(history, logger, True)
        else:
            policy.pretrain(history, logger)
        del logger

        bandit = get_bandit(dataset, True)
        logger = get_logger("wandb", policy.get_name(), dataset + '|eval_pretrained')
        logger.log_config(config)
        regret_sum = 0
        for _ in range(50000):
            context = bandit.get_context(1)
            action = policy.get_action(context)
            reward, regret = bandit.pull_arm(action)
            regret_sum += regret
            policy.update(context, action, reward)
            logger.log({ "regret": regret_sum })

if __name__ == '__main__':
    policies = ['greedy', 'bayes_by_backprob', 'neural_ucb', 'neural_ucb_pretrained']
    parser = argparse.ArgumentParser(description='Train CMAB policy on selected dataset')
    parser.add_argument('dataset', type=str, choices=['mushroom', 'ecoli', 'mnist', 'shuttle'], help='Dataset name')
    args = parser.parse_args()

    history_logs = get_history_logs_for_dataset(args.dataset)
    for p in policies:
        pretrain_ucb = False
        if p == 'neural_ucb_pretrained':
            p = 'neural_ucb'
            pretrain_ucb = True
        p = Process(target=pretrain_policy, args=(p, history_logs, args.dataset, pretrain_ucb))
        p.start()
        p.join() # this blocks until the process terminates
