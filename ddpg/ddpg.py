import os

import gym
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import Adam
import core

class ReplayBuffer:
    '''只需要存放(s,a,r,s',d)数据'''
    def __init__(self,obs_dim,act_dim,size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size),dtype=np.float32)
        self.done_buf = np.zeros(core.combined_shape(size), dtype=np.float32)
        self.ptr,self.size,self.max_size = 0, 0, size

    def store(self,obs,act,rew,next_obs,done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs = self.obs_buf[idxs],
                     obs2 = self.obs2_buf[idxs],
                     rew = self.rew_buf[idxs],
                     act = self.act_buf[idxs],
                     done = self.done_buf[idxs]
                     )
        '''因为batch数据要放到神经网络中因此要转换成tensor 注意一定要用batch.items()不然无法解包(必须是元组才能被解包)'''
        return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in batch.items()}

def ddpg(
    env_fn = lambda : gym.make("LunarLanderContinuous-v2"),
    actor_critic = core.MLPActorCritic,
    ac_kwargs = dict(
        hidden_sizes=(255,255),
        activation=nn.ReLU
    ),
    seed = 0,
    steps_per_epoch = 4000,
    epochs = 100,
    replay_size = int(1e6),
    gamma = 0.99,
    polyak = 0.995,
    pi_lr = 1e-3,
    q_lr = 1e-3,
    batch_size = 100,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_noise = 0.1,
    num_test_episodes = 10,
    max_ep_len = 1000,
    save_freq = 10,
    save_path = ''
):
    '''
    steps_per_epoch: 一个epoch多少个episode
    polyak: 用于延迟更新target网络参数
    start_steps: 在这交互步之前，动作都会在动作空间中随机采样，以丰富经验池中存放
    update_after: 在这个交互步以前不会进行网络更新
    update_every: 每隔多少部进行更新
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    ac = actor_critic(env.observation_space,env.action_space,**ac_kwargs)
    ac_targ = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    history = dict(
        reward = []
    )

    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim,act_dim,replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'] ,data['rew'], data['obs2'], data['done']
        q = ac.q(o, a)
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * q_pi_targ * (1 - d)

        loss_q = ((q - backup)**2).mean()
        return loss_q

    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = torch.optim.Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1-polyak) * p.data)
        return loss_q, loss_pi

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        mean_ep_ret = 0
        mean_ep_len = 0
        max_ep_ret = -99999
        min_ep_ret = 99999
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                #测试阶段不加入噪声，noise_scale为0
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            mean_ep_ret = (j/(j+1))*mean_ep_ret + (1/(j+1))*ep_ret
            mean_ep_len = (j/(j+1))*mean_ep_len + (1/(j+1))*ep_len
            if ep_ret > max_ep_ret:
                max_ep_ret = ep_ret
            if ep_ret < min_ep_ret:
                min_ep_ret = ep_ret
            return mean_ep_len,mean_ep_ret,max_ep_ret,min_ep_ret

    total_steps = steps_per_epoch * epochs

    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_q, loss_pi = update(batch)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ac.state_dict(), save_path)

            mean_ep_len, mean_ep_ret, max_ret, min_ret = test_agent()
            history['reward'].append(mean_ep_ret)
            print(f'Epoch: {epoch}')
            print('-------------------------')
            print(f'EpRet: {mean_ep_ret}')
            print(f'EpLen: {mean_ep_len}')
            print(f'MaxRet: {max_ret}')
            print(f'MinRet: {min_ret}')
            print(f'LossPi: {loss_pi}')
            print(f'LossQ: {loss_q}')
            print('------------------------------------\n')
            print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    return history

if __name__ == '__main__':
    ddpg()