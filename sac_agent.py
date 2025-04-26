import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.99
tau = 0.005
alpha = 0.2
lr = 3e-4
batch_size = 256
buffer_limit = int(1e6)

# Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self):
        mini_batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, done = zip(*mini_batch)
        return (torch.tensor(s, dtype=torch.float).to(device),
                torch.tensor(a, dtype=torch.float).to(device),
                torch.tensor(r, dtype=torch.float).unsqueeze(1).to(device),
                torch.tensor(s_prime, dtype=torch.float).to(device),
                torch.tensor(done, dtype=torch.float).unsqueeze(1).to(device))

    def size(self):
        return len(self.buffer)

# SAC Agent
class SACAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.q1 = Critic(obs_dim, act_dim).to(device)
        self.q2 = Critic(obs_dim, act_dim).to(device)
        self.q1_target = Critic(obs_dim, act_dim).to(device)
        self.q2_target = Critic(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay = ReplayBuffer()

    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

    def update(self):
        if self.replay.size() < batch_size:
            return

        s, a, r, s_prime, done = self.replay.sample()

        with torch.no_grad():
            a_prime, log_pi = self.actor.sample(s_prime)
            min_q = torch.min(
                self.q1_target(s_prime, a_prime),
                self.q2_target(s_prime, a_prime)
            )
            target_q = r + gamma * (1 - done) * (min_q - alpha * log_pi)

        q1_loss = nn.MSELoss()(self.q1(s, a), target_q)
        q2_loss = nn.MSELoss()(self.q2(s, a), target_q)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        a_pred, log_pi = self.actor.sample(s)
        min_q_pi = torch.min(self.q1(s, a_pred), self.q2(s, a_pred))
        actor_loss = (alpha * log_pi - min_q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self.soft_update(self.q1_target, self.q1)
        self.soft_update(self.q2_target, self.q2)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        return action.squeeze(0).detach().cpu().numpy()