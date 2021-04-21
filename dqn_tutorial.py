import os
import random
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data._utils import collate

import wandb


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(dqn, env):
    episode_rewards = []
    for episode in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = dqn.get_action(obs)
            new_obs, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward
            obs = new_obs
        episode_rewards.append(episode_reward)
    print("Average episodes (10): ", np.mean(episode_rewards))


class ReplayBuffer:
    def __init__(
        self, replay_size, batch_size, device, observation_space, action_space
    ):
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.device = device
        self.index = 0
        self.full = False

        self.states = np.empty((replay_size, observation_space), dtype=np.float32)
        self.next_states = np.empty((replay_size, observation_space), dtype=np.float32)
        self.actions = np.empty((replay_size, 1), dtype=np.int)
        self.rewards = np.empty((replay_size, 1), dtype=np.float32)
        self.dones = np.empty((replay_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.states[self.index], obs)
        np.copyto(self.actions[self.index], action)
        np.copyto(self.rewards[self.index], reward)
        np.copyto(self.next_states[self.index], next_obs)
        np.copyto(self.dones[self.index], float(done))

        self.index = (self.index + 1) % self.replay_size
        self.full = self.full or self.index == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.replay_size if self.full else self.index, size=self.batch_size
        )

        states = self.states[idxs]
        next_states = self.next_states[idxs]

        states = torch.as_tensor(states, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = (
            torch.as_tensor(self.rewards[idxs], device=self.device).squeeze().float()
        )
        next_states = torch.as_tensor(next_states, device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device).squeeze().float()
        return states, actions, rewards, next_states, dones


class MLP(nn.Module):
    def __init__(self, layers, observation_space, action_space):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN:
    def __init__(
        self,
        env,
        rbuffer,
        value,
        target,
        optimizer,
        device,
        eps_start,
        eps_decay,
        eps_end,
        totalnum_iterations,
        gamma,
        update_target,
    ):
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.epsilon = max(self.eps_start * self.eps_decay, self.eps_end)
        self.batch_size = batch_size
        self.env = env
        self.rbuffer = rbuffer
        self.value = value
        self.target = target
        self.optimizer = optimizer
        self.device = device
        self.totalnum_iterations = totalnum_iterations
        self.gamma = gamma
        self.update_target = update_target
        self.loss_fn = torch.nn.MSELoss()

    @torch.no_grad()
    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            return torch.argmax(self.value(obs)).item()

    def train(self):
        for episode in tqdm(range(0, self.totalnum_iterations)):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(obs)
                new_obs, reward, done, _ = env.step(action)
                self.rbuffer.add(obs, action, reward, new_obs, done)
                episode_reward += reward
                obs = new_obs

            if self.rbuffer.index > self.batch_size:
                states, actions, rewards, next_states, dones = self.rbuffer.sample()
                next_qs = dones * rewards + (1 - dones) * (
                    rewards
                    + self.gamma * self.target(next_states).detach().max(dim=1).values
                )
                current_qs = torch.gather(self.value(states), 1, actions).squeeze()
                loss = self.loss_fn(current_qs, next_qs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.epsilon = max(self.epsilon * self.eps_decay, self.eps_end)
                wandb.log(
                    {
                        "Episode reward": episode_reward,
                        "Epsilon": self.epsilon,
                        "Loss": loss,
                    },
                    step=episode,
                )

            if episode % self.update_target == 0:
                self.target.load_state_dict(self.value.state_dict())


seed = np.random.randint(1000)
eps_start = 1
eps_decay = 0.999
eps_end = 0.05
batch_size = 128
totalnum_iterations = 2000
gamma = 0.9
update_target = 50
lr = 1e-3
env_name = "CartPole-v0"
device = "cpu"
replay_size = 10000
layers = [32, 64]
env = gym.make(env_name)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

wandb.init(entity="agkhalil", project="dqn-tutorial")
wandb.config["seed"] = seed
wandb.config["eps_start"] = eps_start
wandb.config["eps_decay"] = eps_decay
wandb.config["eps_end"] = eps_end
wandb.config["batch_size"] = batch_size
wandb.config["totalnum_iterations"] = totalnum_iterations
wandb.config["gamma"] = gamma
wandb.config["update_target"] = update_target
wandb.config["lr"] = lr
wandb.config["env_name"] = env_name
wandb.config["device"] = device
wandb.config["replay_size"] = replay_size
wandb.config["observation_space"] = observation_space
wandb.config["action_space"] = action_space

set_all_seeds(seed)

env = gym.make(env_name)
device = torch.device(device)

rbuffer = ReplayBuffer(replay_size, batch_size, device, observation_space, action_space)

value = MLP(layers, observation_space, action_space).to(device)
target = MLP(layers, observation_space, action_space).to(device)
wandb.watch(value)
wandb.watch(target)

optimizer = optim.Adam(params=value.parameters(), lr=lr)
dqn = DQN(
    env,
    rbuffer,
    value,
    target,
    optimizer,
    device,
    eps_start,
    eps_decay,
    eps_end,
    totalnum_iterations,
    gamma,
    update_target,
)
dqn.train()

torch.save(value.state_dict(), "model.h5")

evaluate(dqn, env)
