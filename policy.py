from cmath import inf
import torch.nn as nn
import torch
import torchbnn
from torchhk import transform_model
import random
import numpy as np

def get_policy(name, K_arms, context_size, args):
    if name == 'random':
        return RandomPolicy()
    if name == 'lin_ucb':
        return LinUcbPolicy(K_arms, context_size, **args)
    if name == 'neural_ucb':
        return NeuralUcbPolicy(K_arms, context_size, **args)
    
    return EpsilonPerceptron(K_arms, context_size, **args)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield np.array(lst[i:i + n])

def chunks2(lst1, lst2, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst1), n):
        yield lst1[i:i + n], lst2[i:i + n]

def chunks3(lst1, lst2, lst3, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst1), n):
        yield lst1[i:i + n], lst2[i:i + n], lst3[i:i + n]

class Perceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Perceptron, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
        )

    def forward(self, x):
        return self.model(x)


class EpsilonPerceptron():
    def __init__(self, K_arms, context_size, epsilon, learning_rate, weight_decay, batch_size, hidden_size, bayes = False, kl_weight = 0.001, train_every = 100, epochs = 1, lr_gamma=1.0):
        self.model = Perceptron(context_size + K_arms, hidden_size)
        self.kl_loss = None
        self.lr_gamma = lr_gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if bayes:
            transform_model(self.model, nn.Linear, torchbnn.BayesLinear, 
            args={"prior_mu":0, "prior_sigma":0.001, "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"
                 }, 
            attrs={"weight_mu" : ".weight"})
            self.kl_loss = torchbnn.BKLLoss(reduction='mean', last_layer_only=False)
            self.kl_weight = kl_weight
        else:
            self.model.apply(init_weights)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, lr_gamma)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.K_arms = K_arms
        self.replay_buffer = []
        self.train_every = train_every
        self.epochs = epochs
        self.iter = 0
    
    def reset(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, self.lr_gamma)

    def get_name(self):
        if self.kl_loss:
            return "bayes_by_backprob"
        
        return "epsilon_greedy(" + str(self.epsilon) + ")"

    def get_action(self, context):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.eye(self.K_arms, dtype=np.float32)[np.random.choice(self.K_arms,1)]

        possible_actions = np.eye(self.K_arms, dtype=np.float32)
        inputs = []   
        for action in possible_actions:
            inputs.append(np.append(context, [action], axis=1))

        estimated_rewards = []
        self.model.eval()
        for input in inputs:
            
            self.model.eval()
            sample_reward = self.model(torch.tensor(input).float())

            if self.kl_loss:
                sample_reward += self.model(torch.tensor(input).float())
                sample_reward /= 2.

            estimated_rewards.append(sample_reward)
            
        return possible_actions[[np.argmax(estimated_rewards)]]
    
    def train(self):
        self.replay_buffer = self.replay_buffer[-4096:]
        buffer = random.sample(self.replay_buffer, len(self.replay_buffer))

        self.model.train()
        for _ in range(self.epochs):
            for batch in chunks(buffer, self.batch_size):
                self.optimizer.zero_grad()
                
                batch_context = torch.from_numpy(batch[:, :-1])
                batch_target = torch.from_numpy(batch[:, -1])
                pred = self.model(batch_context.float())
                loss = self.criterion(pred.squeeze(), batch_target.float())
                if self.kl_loss is not None:
                    kl = self.kl_loss(self.model)
                    loss = loss + self.kl_weight*kl
            
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        return

    def pretrain(self, history, logger):
        self.model.train()
        bs = self.batch_size
        for k, v in history.items():
            history[k] = torch.from_numpy(v).float()

        for _ in range(self.epochs):
            for context, action, reward in chunks3(history['context'], history['action'], history['reward'], bs):
                self.optimizer.zero_grad()
                batch_context = torch.cat([context, action], 1) 
                pred = self.model(batch_context)
                loss = self.criterion(pred.squeeze(), reward)
                if self.kl_loss is not None:
                    kl = self.kl_loss(self.model)
                    loss = loss + self.kl_weight*kl

                logger.log({"train/loss" : loss})
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        return


    def update(self, context, action, reward):
        sample = np.append(context, action, axis=1)
        sample = np.append(sample, reward)
        self.replay_buffer.append(sample)

        if self.iter % self.train_every == 0:
            self.train()

        self.iter += 1
        return

class RandomPolicy():
    def __init__(self):
        pass
    def get_name(self):
        return "random"

    def get_action(self, context):
        return np.eye(2)[np.random.choice(2,1)]
    
    def train(self):
        pass
    def update(self, context, action, reward):
        pass


class LinUcbDisjointArm():
    
    def __init__(self, arm_index, d, alpha):
        
        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of alpha
        self.alpha = alpha
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d, dtype=np.float32)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1], dtype=np.float32)
        
    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        
        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)
        
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))
        
        return p
    
    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x


class LinUcbPolicy():
    
    def __init__(self, K_arms, context_size, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [LinUcbDisjointArm(arm_index = i, d = context_size, alpha = alpha) for i in range(K_arms)]
        
    def get_action(self, x_array):
        highest_ucb = -1
        
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)
            
            if arm_ucb > highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        choosen_arm = np.random.choice(candidate_arms)

        action = np.zeros(self.K_arms, dtype=np.float32)
        action[choosen_arm] = 1.

        return [action]

    def get_name(self):
        return f"lin_ucb({self.linucb_arms[0].alpha})"
    
    def update(self, x_array, action, reward):
        arm_index = np.where(action[0] == 1)[0][0]
        return self.linucb_arms[arm_index].reward_update(reward, x_array)


class NeuralUcbPolicy():
    def __init__(self, K_arms, context_size, alpha, learning_rate, weight_decay, batch_size, epochs = 1,
                 replay_buffer_size = 4096, train_every = 100, reg_factor = 1.0, hidden_size = 20, lr_gamma = 1.0, device = 'cpu'):
        self.K_arms = K_arms
        self.n_features = context_size * K_arms
        self.model = Perceptron(self.n_features, hidden_size).to(device)
        self.hidden_size = hidden_size
        self.lr_gamma = lr_gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        self.reg_factor = reg_factor
        self.diag_Z = torch.from_numpy(np.ones(self.approximator_dim, dtype=np.float32)).to(device)/self.reg_factor

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, lr_gamma)
        self.batch_size = batch_size
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.epochs = epochs
        self.alpha = alpha
        self.iter = 0
        self.train_every = train_every

    def reset(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, self.lr_gamma)


    def __disjoint_context(self, context, arm):
        l = context.shape[-1]
        if l == self.n_features:
            return context
        
        ctx = torch.zeros(self.n_features).to(self.device)
        ctx[l * arm : l * (arm + 1)] = context
        return ctx.float()

    def train(self):
        self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]
        buffer = random.sample(self.replay_buffer, len(self.replay_buffer))

        self.model.train()
        for _ in range(self.epochs):
            for batch in chunks(buffer, self.batch_size):
                self.optimizer.zero_grad()
                
                batch_context = torch.from_numpy(batch[:, :-1])
                batch_target = torch.from_numpy(batch[:, -1])
                
                pred = self.model(batch_context.float())
                loss = self.criterion(pred.squeeze(), batch_target.float())
                
                loss.backward()
                self.optimizer.step()
    
    def pretrain(self, history, logger, pretrain_ucb = False):
        self.model.train()
        
        new_shape = (history['context'].shape[0], self.n_features)
        new = np.zeros(new_shape)
        
        for i in range(history['context'].shape[0]):
            new[i] = self.__disjoint_context(history['context'][i], np.where(history['action'][i] == 1)[0][0])
        
        history['context'] = torch.from_numpy(new).float()
        history['reward'] = torch.from_numpy(history['reward']).float()
        
        for _ in range(self.epochs):
            for context, reward in chunks2(history['context'], history['reward'], self.batch_size):
                self.optimizer.zero_grad()
                pred = self.model(context.to(self.device))
                loss = self.criterion(pred.squeeze(), reward.to(self.device()))

                logger.log({"train/loss" : loss})
                loss.backward()
                self.optimizer.step()
                if (pretrain_ucb):
                    for c in context:
                        self.update_Z(c)

            self.scheduler.step()
        return


    def approx_grad(self, disjoint_context):
        self.model.zero_grad()
        y = self.model(disjoint_context)
        y.backward()

        grad_approx = torch.cat(
            [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
        )

        return grad_approx

    @property
    def diag_Z_inv(self):
        return 1/self.diag_Z

    def update_Z(self, disjoint_context):
        '''Since it is computionally expensive we will approximate Z_inv by its diagonal '''
        grad = self.approx_grad(disjoint_context)
        self.diag_Z += grad * grad.T

    def update(self, context, action, reward):
        disjoint_context = self.__disjoint_context(context, np.where(action[0] == 1)[0][0])
        sample = np.append(disjoint_context, reward)
        self.replay_buffer.append(sample)
        
        if self.iter % self.train_every == 0:
            self.train()
        
        self.update_Z(disjoint_context)
        self.iter += 1

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def get_action(self, context):
        highest_ucb = -inf
        
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            disjoint_context = self.__disjoint_context(context[0], arm_index)
            grad = self.approx_grad(disjoint_context)
            self.model.eval()
            arm_ucb = self.model.forward(disjoint_context) + self.alpha * torch.sqrt(torch.dot(grad, self.diag_Z_inv * grad.T))
            if arm_ucb > highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms = [arm_index]

            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)
        
        choosen_arm = np.random.choice(candidate_arms)

        action = np.zeros(self.K_arms)
        action[choosen_arm] = 1.

        return [action]

    def get_name(self):
        return f"neural_ucb({self.alpha})"