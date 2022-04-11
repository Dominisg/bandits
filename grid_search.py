from policy import get_policy
from bandits import get_bandit
from logger import get_logger

policies = ['neural_ucb']
learning_rate = [0.0001]
lr_gamma = [0.98, 0.90]
weight_decay = [0.0001]
kl_weight = [0.0001, 0.00001]
alpha = [0.1]
lin_alpha = [0.1, 0.3, 1, 10]
trials = 1

for p in policies:
    configs = []
    for _ in range(trials):
        if p == 'bayes_by_backprob':
                for lr in learning_rate:
                    for wd in weight_decay:
                        for kw in kl_weight:
                            for lg in lr_gamma:
                                configs.append({  
                                    'epsilon': 0.00,
                                    'learning_rate': lr,
                                    'batch_size': 64,
                                    'weight_decay': wd,
                                    'hidden_size': 100,
                                    'bayes': True,
                                    'kl_weight': kw,
                                    'lr_gamma':lg
                                })
        elif p == 'lin_ucb':
            for a in lin_alpha: 
                configs.append({
                    'alpha' : a,
                })
        elif p == 'neural_ucb':
            for lr in learning_rate:
                for wd in weight_decay:
                    for a in alpha:
                        for lg in lr_gamma:
                            configs.append({
                                'alpha': a,
                                'learning_rate': lr,
                                'batch_size': 64,
                                'weight_decay': wd,
                                'hidden_size': 100,
                                'lr_gamma': lg
                            })
        else:
            for lr in learning_rate:
                for wd in weight_decay:
                    for lg in lr_gamma:
                            configs.append({  
                                'epsilon': 0.0,
                                'learning_rate': lr,
                                'batch_size': 64,
                                'weight_decay': wd,
                                'hidden_size': 100,
                                'lr_gamma': lg
                            })
                            
    for config in configs:
        bandit = get_bandit('mnist')
        policy = get_policy(p, bandit.arms_count(), bandit.context_size(), config) 
        logger = get_logger("wandb", policy.get_name(), 'mnist_grid_search')
        config['policy'] = p
        logger.log_config(config)

        regret_sum = 0
        for _ in range(20000):
            if (regret_sum < 10000):
                context = bandit.get_context(1)
                action = policy.get_action(context)
                reward, regret = bandit.pull_arm(action)
                regret_sum += regret
                policy.update(context, action, reward)
            logger.log({ "regret": regret_sum })
        
        del logger
