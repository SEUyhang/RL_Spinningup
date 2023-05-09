import gym.spaces
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

#维度连接
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

#构建全连接层nn.Identity() 是个函数，恒等函数f(x)=x
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

#计算每一层参数数量
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

#计算累计折扣回报
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):
    #生成一个分布
    def _distribution(self, obs):
        raise NotImplementedError()
    #在分布pi中act的概率密度的log值
    def _log_prob_from_distributions(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distributions(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim],activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits = logits)

    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        #方差是可学习的参数
        self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32))
        #神经网络的输出是各个维度动作的均值
        self.mu_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim],activation)
    #根据方差和均值构造一个多维正太分布
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)
    #由于是多维正态分布，某一个取值的概率密度是多个分量的乘积，取对数就是相加(最后一个维度)
    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim]+list(hidden_sizes)+[1],activation)

    def forward(self, obs):
        #由于神经网络输入和输出的维度是一致的，输入是[batch,obs_dim]，输出是[batch,1]，把最后一个维度给简化
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64),activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, gym.spaces.Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0],hidden_sizes,activation)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
    #根据当前的策略函数采样动作，在此过程中不记录梯度
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distributions(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()