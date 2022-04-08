import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bandits import Bandit

def get_offline_bandit(method, log):
    if method == 'dm':
        return DirectOfflineBandit(log)
    if method == 'ips':
        return InversePropensityScoreOfflineBandit(log)
    if method == 'dr':
        return DoublyRobustOfflineBandit(log)
    if method == 'replay':
        return ReplyOfflineBandit(log)

class DirectOfflineBandit(Bandit):
    def __init__(self, log):
        super().__init__()
        df = pd.read_csv(log)
        headers = list(df.columns.values)
        ctx_headers = [str for str in headers if 'ctx' in str]
        self.K = len([str for str in headers if 'action' in str])
        self.regr = RandomForestRegressor()
        
        y = df["reward"]
        x = df.drop(["reward"], axis=1)
        self.regr.fit(x.values, y.values)

        self.x = df[ctx_headers]

    def reward(self, contexts, actions):
        return self.regr.predict(np.concatenate([contexts, actions], axis=1))
    
    def pull_arm(self, actions):
        last = self.idx + len(actions) 
        if last > len(self.x):
            last = len(self.x)
        
        contexts = np.array(self.x[self.idx : last], dtype=np.float)

        reward = self.reward(contexts, actions)
        self.idx = last % (len(self.x) - 1)

        return reward
    
    def arms_count(self):
        return self.K

class InversePropensityScoreOfflineBandit(Bandit):
    def __init__(self, log):
        super().__init__(log)
        df = pd.read_csv(log)
        headers = list(df.columns.values)
        to_remove = [str for str in headers if 'ctx' not in str]
        action_headers = [str for str in headers if 'action' in str]
        self.K = len(action_headers)
        self.classifier = RandomForestClassifier()
        
        self.actions = df[action_headers]
        self.rewards = df['reward']
        self.x = df.drop(to_remove, axis=1)
        
        self.classifier.fit(self.x.values, self.actions.values)
        # self.reward = np.vectorize(InversePropensityScoreOfflineBandit._reward, excluded=['self'], signature='(),(m),(n),(o),(p)->()')

    def reward(self, contexts, actions, real_rewards, real_actions):
        est = [np.concatenate([prob[:,1] for prob in self.classifier.predict_proba(contexts)])]
        return  real_rewards * (np.array_equal(actions, real_actions))  / max(np.sum(est * actions), 0.001)

    def pull_arm(self, actions):
        last = self.idx + len(actions) 
        if last > len(self.x):
            last = len(self.x)
        
        contexts = np.array(self.x[self.idx : last], dtype=np.float)
        rewards = np.array(self.rewards[self.idx : last], dtype=np.float)
        real_actions = np.array(self.actions[self.idx : last], dtype=np.float)

        est_rewards = self.reward(contexts, actions, rewards, real_actions)
        self.idx = last % (len(self.x) - 1)

        return est_rewards
    
    def arms_count(self):
        return self.K

class DoublyRobustOfflineBandit(InversePropensityScoreOfflineBandit, DirectOfflineBandit):
    def __init__(self, log):
        super(DoublyRobustOfflineBandit, self).__init__(log)

    def reward(self, contexts, actions, real_rewards, real_actions):
        est = [np.concatenate([prob[:,1] for prob in self.classifier.predict_proba(contexts)])]
        ra_est = self.regr.predict(np.concatenate([contexts, real_actions], axis=1))
        dm_reward = self.regr.predict(np.concatenate([contexts, actions], axis=1))
        return  (((real_rewards - ra_est) * (np.array_equal(actions, real_actions))  / max(np.sum(est * actions), 0.001)) + dm_reward)

class ReplyOfflineBandit():
    def __init__(self, log):
        df = pd.read_csv(log)
        headers = list(df.columns.values)
        to_remove = [str for str in headers if 'ctx' not in str]
        action_headers = [str for str in headers if 'action' in str]
        
        self.K = len(action_headers)
        self.actions = df[action_headers]
        self.rewards = df['reward']
        self.x = df.drop(to_remove, axis=1)

    def context_size(self):
        return self.x.shape[1] 
   
    def arms_count(self):
        return self.K
    
    def get_dataset(self):
        return {'context': self.x.to_numpy(dtype=np.float32), 'action': self.actions.to_numpy(dtype=np.float32), 'reward': self.rewards.to_numpy(dtype=np.float32)}