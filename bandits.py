import numpy as np
import pandas as pd

def get_bandit(dataset, test=False):
    if dataset == 'mushroom':
        return MushroomBandit()
    if dataset == 'ecoli':
        return ColiBandit()
    if dataset == 'mnist':
        return MnistBandit(test) 
    if dataset == 'shuttle':
        return ShuttleBandit(test)
    return None

class Bandit: 
    def __init__(self):
        self.idx = 0
        self.epoch = 0

    def __len__(self):
        return len(self.x)

    def oracle(self):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    def get_context(self, batch_size):
        last = self.idx + batch_size 
        if  last >= len(self.x):
            last = len(self.x) - 1
        return np.array(self.x[self.idx : last], dtype=np.float32)

    def context_size(self):
        return self.x[self.idx : self.idx + 1].shape[1]

    def pull_arm(self, actions):
        last = self.idx + len(actions) 
        if last > len(self.x):
            last = len(self.x)

        targets = np.array(self.y[self.idx : last], dtype=np.float32)

        reward = self.reward(actions, targets)
        oracle = self.oracle(targets)

        self.idx = last % (len(self.x) - 1)
        return reward, oracle - reward

class MushroomBandit(Bandit):
    def __init__(self):
        super().__init__()
        df = pd.read_csv('data/shuffled-mushrooms.csv')
        df = pd.get_dummies(df)
        df = df.sample(frac=1).reset_index(drop=True)
        self.y = df["Class_poisonous"]
        self.x = df.drop(["Class_poisonous", "Class_edible"], axis=1)
        
        self.oracle = np.vectorize(MushroomBandit._oracle)
        self.reward = np.vectorize(MushroomBandit._reward, signature='(n),()->()')
    
    def _oracle(poisonous):
        if poisonous == 1:
            return 0.
        return 5.

    def _reward(action, poisonous):
        if action[0] == 1 and poisonous == 1 and np.random.binomial(1, 0.5) == 1:
            return -35.
        if action[1] == 1:
            return 0
        return 5.

    def arms_count(self):
        return 2 

class MnistBandit(Bandit):
    def __init__(self, test):
        super().__init__()
        
        if test:
            df = pd.read_csv('data/mnist_test.csv')
        else:
            df = pd.read_csv('data/mnist_train.csv')
        df = df.sample(frac=1).reset_index(drop=True)
        
        self.y = pd.get_dummies(df["label"])
        self.x = df.drop(["label"], axis=1)
        self.x /= 255.
        print(self.x.head())
        
        self.oracle = np.vectorize(MnistBandit._oracle, signature='(m)->()', )
        self.reward = np.vectorize(MnistBandit._reward, signature='(n),(m)->()')
    
    def _oracle(poisonous):
        return 1

    def _reward(action, target):
        if np.array_equal(action, target):
            return 1.
        return 0.

    def arms_count(self):
        return 10

class ColiBandit(Bandit):
    def __init__(self):
        super().__init__()
        df = pd.read_csv('data/ecoli.csv', sep=' ', header=None)
        col_names = ["squence_name","mcg","gvh","lip","chg","aac","alm1","alm2","site"]
        df.columns = col_names
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(['squence_name', 'chg', 'aac'], axis=1)
        df = pd.get_dummies(df)

        to_remove = [str for str in df.columns if 'site' in str]
        not_to_remove = [str for str in df.columns if 'site' not in str]

        self.K_arms = len(to_remove)
        
        self.y = df.drop(not_to_remove, axis=1)
        self.x = df.drop(to_remove, axis=1)

        self.oracle = np.vectorize(ColiBandit._oracle, signature='(m)->()', )
        self.reward = np.vectorize(ColiBandit._reward, signature='(n),(m)->()')
    
    def _oracle(target):
        return 1.

    def _reward(action, target):
        if np.array_equal(action, target):
            return 1.
        return 0.

    def arms_count(self):
        return self.K_arms

class ShuttleBandit(Bandit):
    def __init__(self, test):
        super().__init__()
        
        if test:
            df = pd.read_csv('data/shuttle.tst', header=None, delimiter=' ')
        else:
            df = pd.read_csv('data/shuttle.trn', header=None, delimiter=' ')
        df = df.sample(frac=1).reset_index(drop=True)
        
        self.y = pd.get_dummies(df.iloc[: , -1])
        self.x = df.iloc[: , :-1]
        self.x = (self.x-self.x.mean())/self.x.std()

        self.K_arms = self.y.shape[1]
        
        self.oracle = np.vectorize(MnistBandit._oracle, signature='(m)->()', )
        self.reward = np.vectorize(MnistBandit._reward, signature='(n),(m)->()')
    
    def _oracle(poisonous):
        return 1

    def _reward(action, target):
        if np.array_equal(action, target):
            return 1.
        return 0.

    def arms_count(self):
        return self.K_arms
