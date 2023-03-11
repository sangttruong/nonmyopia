#!/usr/bin/env python
"""
Reinformcement learning baseline for non-myopic
Bayesian optimization
"""

from synthfunc import SynthFunc
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from argparse import Namespace, ArgumentParser
__author__ = "Sang T. Truong, Willie Neiswanger, Shengjia Zhao, Stefano Ermon"
__copyright__ = "Copyright 2022, Stanford University"

import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})


class SynthFuncEnv(gym.Env):
    def __init__(self, n_dim):
        self.T = 100
        self.state_dim = 1
        self.n_dim = n_dim
        hypers = {"ls": 1.0, "alpha": 2.0, "sigma": 1e-2, "n_dimx": self.n_dim}
        self.sf_bounds = [0, 10]
        sf_domain = [self.sf_bounds] * self.n_dim
        self.lower_bounds = [self.sf_bounds[0]] * self.n_dim
        self.upper_bounds = [self.sf_bounds[1]] * self.n_dim
        self.action_space = gym.spaces.Box(
            low=np.array(self.lower_bounds, dtype=np.float32),
            high=np.array(self.upper_bounds, dtype=np.float32),
        )
        self.state = torch.tensor([])
        self.sf = SynthFunc(seed=12, hypers=hypers, n_obs=50, domain=sf_domain)

    def step(self, action):
        reward = self.sf(action)
        self.state = torch.cat([self.state, action, torch.tensor([reward])])
        self.state_dim = self.state_dim + self.n_dim + 1 if self.state_dim > 1 else 3
        self.T = self.T - 1
        done = False if self.T > 0 else True
        return self.state, reward, done

    def reset(self):
        """
        Return the initial state
        """
        return torch.tensor(0)

    def update_action_bound(self, action, r):
        self.lower_bounds = []
        self.upper_bounds = []
        for i in range(self.n_dim):
            lower_bound = action[i].item() - r
            lower_bound = lower_bound if lower_bound >= self.sf_bounds[0] else self.sf_bounds[0]
            self.lower_bounds.append(lower_bound)

            upper_bound = action[i].item() + r
            upper_bound = upper_bound if upper_bound <= self.sf_bounds[1] else self.sf_bounds[1]
            self.upper_bounds.append(upper_bound)

        self.action_space = gym.spaces.Box(
            low=np.array(self.lower_bounds, dtype=np.float32),
            high=np.array(self.upper_bounds, dtype=np.float32),
        )

        self.lower_bounds = torch.tensor(self.lower_bounds)
        self.upper_bounds = torch.tensor(self.upper_bounds)

        return self.lower_bounds, self.upper_bounds


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done"))

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Noise:
    def __init__(self, params):
        self.mu = params.mu
        self.theta = params.theta
        self.sigma = params.sigma
        self.reset()

    def reset(self):
        self.state = np.full(params.action_dim, self.mu)

    def make_noise(self):
        state = self.state
        delta = self.theta * (self.mu - state) + \
            self.sigma * np.random.randn(len(state))
        self.state = state + delta
        return self.state


class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(params.state_dim,  params.act_hid_1)
        self.fc2 = nn.Linear(params.act_hid_1, params.act_hid_2)
        self.fc3 = nn.Linear(params.act_hid_2,  params.action_dim)

    def forward(self, x):
        self.__init__(params)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(params.state_dim, params.crit_hid_1)
        self.fc2 = nn.Linear(params.crit_hid_1 +
                             params.action_dim, params.crit_hid_2)
        self.fc3 = nn.Linear(params.crit_hid_2, 1)

    def forward(self, x, action):
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Model:
    def __init__(self, params):

        self.device = params.device

        self.actor = Actor(params).to(self.device)
        self.actor_target = Actor(params).to(self.device)
        self.critic = Critic(params).to(self.device)
        self.critic_target = Critic(params).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params.lr_critic)

        self.tau_actor = params.tau_actor
        self.tau_critic = params.tau_critic

        self.__update(self.actor_target, self.actor)
        self.__update(self.critic_target, self.critic)

    def __update(self, target, local):
        target.load_state_dict(local.state_dict())

    def __soft_update(self, target, local, tau):
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

    def update_target_nn(self):
        self.__soft_update(self.actor_target, self.actor, self.tau_actor)
        self.__soft_update(self.critic_target, self.critic, self.tau_critic)


class DDPG:
    def __init__(self, params):

        self.device = params.device
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.act_up, self.act_down = params.act_up, params.act_down

        self.explor_noise = Noise(params)
        self.buffer = ReplayBuffer(params.buffer_size)
        self.model = Model(params)

    def update(self):
        if len(self.buffer) <= self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.buffer.transition(*zip(*transitions))

        state_batch = self.tensor(batch.state).float()
        action_batch = self.tensor(batch.action).float()
        reward_batch = self.tensor(batch.reward).float()
        next_state_batch = self.tensor(batch.next_state).float()

        with torch.no_grad():
            next_actions = self.model.actor_target(next_state_batch)
        Q_next = self.model.critic_target(
            next_state_batch, next_actions).detach()

        reward_batch = reward_batch.unsqueeze(1)
        not_terminate_batch = ~torch.tensor(
            batch.done).to(self.device).unsqueeze(1)

        Q = self.model.critic(state_batch, action_batch)
        Q_expected = reward_batch + self.gamma * Q_next * not_terminate_batch

        L = F.mse_loss(Q, Q_expected)
        self.model.critic_optimizer.zero_grad()
        L.backward()
        for param in self.model.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.model.critic_optimizer.step()

        a = self.model.actor(state_batch)
        L_policy = - self.model.critic(state_batch, a).mean()
        self.model.actor_optimizer.zero_grad()
        L_policy.backward()
        for param in self.model.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.model.actor_optimizer.step()

        self.model.update_target_nn()

    def act(self, state, eps=0):
        state = state.float().unsqueeze(0)
        with torch.no_grad():
            action = self.model.actor(state)
        action = action + eps*self.explor_noise.make_noise()
        return np.clip(action, self.act_down, self.act_up)

    def reset(self):
        self.explor_noise.reset()

    def update_action_bound(self):
        self.act_up, self.act_down = params.act_up, params.act_down


class Paramaters:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.state_dim  # env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.act_up = env.upper_bounds  # env.action_space.high[0]
        self.act_down = env.lower_bounds  # env.action_space.low[0]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1
        self.gamma = 0.9
        self.episodes = 1  # there is a single episode
        self.max_steps = 1000
        self.batch_size = 256
        self.buffer_size = int(1e4)

        self.eps = 1.0
        self.eps_decay = 0.95
        self.eps_min = 0.01

        self.tau_actor = 0.1
        self.tau_critic = 0.3
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3

        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2

        self.act_hid_1, self.act_hid_2 = 128, 128
        self.crit_hid_1, self.crit_hid_2 = 128, 128

        self.reward_coef = 20

        self.froze_seed()

        self.r = 0.01  # spotlight radius

    def froze_seed(self):
        self.env.reset()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def update_eps(self):
        self.eps = max(self.eps_min, self.eps*self.eps_decay)

    def update_state_dim(self):
        self.state_dim = self.env.state_dim

    def update_action_bound(self, action):
        self.env.update_action_bound(action, self.r)
        self.act_down = env.lower_bounds
        self.act_up = env.upper_bounds


def run(params, agent, plot_reward=True, plot_action=True):
    rewards = []
    actions = []
    for i in range(1, params.episodes+1):
        state = env.reset()
        agent.reset()
        params.update_eps()
        total_reward, steps = 0, 0
        done = False
        t = 1
        while not done:
            action = agent.act(state, params.eps).squeeze()
            if len(actions) > 0:
                d = ((action - actions[-1])**2).sum().sqrt()
                # print(d)
            actions.append(action)
            params.update_action_bound(action)
            agent.update_action_bound()
            next_state, reward, done = env.step(action)
            params.update_state_dim()
            agent.buffer.push(state, action, next_state, reward, done)
            state = next_state
            agent.update()
            total_reward += reward
            steps += 1

            print("Time step {}, Action: {}, Reward {:.2f}".format(
                t, action.numpy(), reward))
            t = t + 1
        rewards.append(total_reward)
        # print(f"Episode {i}, reward: {total_reward}")

    if plot_action:
        actions = torch.stack(actions)
        # print(actions.shape)
        plt.scatter(actions[:, 0], actions[:, 1],
                    c=torch.arange(0, actions.shape[0]))
        plt.title("Action")
        plt.show()

    if plot_reward:
        plt.plot(range(1, i+1), rewards)
        plt.ylabel("Reward")
        plt.xlabel("Episodes")
        plt.title("Training scores")
        plt.show()
    return agent


if __name__ == "__main__":
    env = SynthFuncEnv(n_dim=2)
    params = Paramaters(env)
    agent = DDPG(params)
    run(params=params, agent=agent)
