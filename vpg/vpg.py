import os

import gym
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import Adam
import core

class GAEBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        #observation, 状态
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        #action, 动作
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        #advantage, 优势函数
        self.adv_buf = np.zeros(size,dtype=np.float32)
        #reward, 奖励
        self.rew_buf = np.zeros(size, dtype=np.float32)
        #reward to go 未来的累积折扣回报
        self.ret_buf = np.zeros(size, dtype=np.float32)
        #value, 状态价值
        self.val_buf = np.zeros(size, dtype=np.float32)
        #log概率
        self.logp_buf = np.zeros(size, dtype=np.float32)
        #折扣因子gamma和GAE参数lamda
        self.gamma, self.lam = gamma, lam
        #buffer指针(指向当前进行存储操作的位置的下标)，当前这一幕数据在buffer中的起始位置 ， buffer的最大存储量
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        #将采样数据存入buffer中
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val = 0):
        #当一幕数据结束的时候调用这个函数，将计算GAE估计，reward to go
        #当前这一幕数据在buffer中的切片，[起始点，结束位置的下一个位置的坐标(当前指针的位置))
        path_slice = slice(self.path_start_idx, self.ptr)

        #由于终结状态不会被存进去，规定终结状态的reward为0
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        #计算GAE
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma*self.lam)

        #计算reward-to-go
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        #当收集完一个epoch所需的steps后调用这个函数，输出buffer数据，并清空buffer数据
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        #标准化优势函数
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean)/adv_std

        data = dict(
            obs = torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act = torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret = torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv = torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp = torch.as_tensor(self.logp_buf, dtype=torch.float32)
        )
        return data



def vpg(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    lam=0.97,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    max_ep_len=1000,
    save_freq=10,
    save_path = "./history.txt"
):
    '''
     Vanilla Policy Gradient

	env_fn，gym仿真环境函数，例如lamda : gym.make('LunarLander-v2')
	actor_critic, 之前写的ActorCritic类，如果需要雅达利环境，可以改写一个cnn网络
	ac_kwargs, 网络需要的参数，例如dict(hidden_sizes=[64,64], activation=nn.Relu)
	seed, 随机种子
	steps_per_epoch， 一个epoch最多多少步数据
	epochs， 学习的epoch数量
	gamma， 折扣因子
	pi_lr，策略学习率
	vf_lr，价值学习率
	train_V_iters，一个epoch对价值函数的学习次数
	lam, GAE参数
	max_ep_len, 定义仿真中一幕的最大长度
	save_freq， 保存频率
	save_path， 保存数据
    '''

    #设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    #创建环境
    env = env_fn()
    #保存状态空间和动作空间的大小
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    #创建actor critic
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    #保存学习过程的历史数据，便于后期观察
    history = {}
    history["reward"] = []
    #创建GAE buffer
    buf = GAEBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    #计算策略损失
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs,act)
        # 注意这里因为pytorch的优化是做梯度下降，所以这里相比理论推导的梯度那里多了个负号
        # 这样就等效成梯度上升了
        loss_pi = -(logp*adv).mean()

        # 有用的数据，可以方便分析学习过程
        # kl散度表示了学习前后的策略概率分布的变化
        approx_kl = (logp_old - logp).mean().item()
        # 策略概率分布的熵，当熵值非常低的时候，就表示策略变成几乎是确定性策略
        ent = pi.entropy().mean().item()
        pi_info = dict(
            kl = approx_kl,
            ent = ent
        )
        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    #创建优化器
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    #学习
    def update():
        #从buffer中获取数据
        data = buf.get()

        #使用梯度更新策略
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        #使用梯度更新价值
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
        kl, ent = pi_info['kl'], pi_info['ent']
        return kl, ent, loss_pi, loss_v

    #开始采样
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        #平均一个幕的回报
        mean_ep_ret = 0
        #指示当前是第几幕
        ep_ret_idx = 0
        #平均一个幕的长度
        mean_ep_len = 0
        #指示当前是第几幕(从0开始)
        ep_len_idx = 0
        max_ret = -99999
        min_ret = 99999
        for t in range(steps_per_epoch):
            #采样一个动作
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            #执行动作
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            #向buffer存入数据
            buf.store(o, a, r, v, logp)

            o = next_o
            timeout = (ep_len == max_ep_len)
            terminal = (d or timeout)
            epoch_ended = (t == steps_per_epoch-1)

            #检查是否是一幕结束的点
            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    #在一幕结束的时候，计算一下最大/最小/平均累计汇报和长度
                    mean_ep_ret = (ep_ret_idx/(ep_ret_idx+1))*mean_ep_ret + (1/(ep_ret_idx+1))*ep_ret
                    mean_ep_len = (ep_len_idx/(ep_len_idx+1))*mean_ep_len + (1/(ep_len_idx+1))*ep_len
                    ep_ret_idx += 1
                    ep_len_idx += 1
                    if ep_ret > max_ret:
                        max_ret = ep_ret
                    if ep_ret < min_ret:
                        min_ret = ep_ret
                o, ep_len, ep_ret = env.reset(), 0 ,0

        kl, ent, loss_pi, loss_v = update()
        # 输出有关信息，方便观察学习过程
        print(f'Epoch: {epoch + 1}')
        print('------------------------------------')
        print(f'EpRet: {mean_ep_ret}')
        print(f'EpLen: {mean_ep_len}')
        print(f'KL: {kl}')
        print(f'Ent: {ent}')
        print(f'LossPi: {loss_pi}')
        print(f'LossV: {loss_v}')
        print(f'MaxRet: {max_ret}')
        print(f'MinRet: {min_ret}')
        print('------------------------------------\n')
        history['reward'].append(mean_ep_ret)

        if epoch % save_freq == 0 and save_path != "":
            torch.save(ac.state_dict(), save_path)
'''
lambda : gym.make("LunarLander-v2")
这个参数传递的实际上是一个函数，而不是一个直接的gym.Env对象。
当你调用这个函数时，它会创建一个新的"LunarLander-v2"环境对象并返回它。
这种方式是为了确保在每次调用函数时都会创建一个新的环境对象，而不是使用旧的环境，从而避免不同的实验共享同一个环境，导致结果出现混淆。
'''
if __name__ == "__main__":
	historty = vpg(
        lambda : gym.make("LunarLander-v2"),
	    actor_critic=core.MLPActorCritic,
	    ac_kwargs=dict(
	        hidden_sizes=(64, 64, 64),
	        activation=nn.Tanh
	    ),
	    gamma=0.99,
	    seed=0,
	    steps_per_epoch=5000,
	    epochs=300,
	    lam=0.95,
	    train_v_iters=80,
	)

