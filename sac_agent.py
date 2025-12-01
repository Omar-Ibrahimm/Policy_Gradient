"""
SAC (Soft Actor-Critic) Agent Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from collections import deque
import random


class Actor(nn.Module):
    """Actor network for SAC."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            return action, None
        
        dist = Normal(mean, std)
        x = dist.rsample()  # Reparameterization trick
        action = torch.tanh(x)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC (Q-function)."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs),
            np.array(action),
            np.array(reward),
            np.array(next_obs),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """SAC Agent with custom implementation."""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,  # Entropy temperature
        batch_size=256,
        buffer_size=1000000,
        learning_starts=1000,
        train_freq=1,
        gradient_steps=1,
        auto_alpha=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.device = device
        self.auto_alpha = auto_alpha
        
        # Create networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Entropy temperature
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.train_step = 0
        
    def select_action(self, obs, deterministic=False):
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
            action_np = action.cpu().numpy()[0]
        
        return action_np
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def update(self):
        """Update networks using SAC."""
        if len(self.replay_buffer) < self.learning_starts:
            return {}
        
        if self.train_step % self.train_freq != 0:
            self.train_step += 1
            return {}
        
        if len(self.replay_buffer) < self.batch_size:
            self.train_step += 1
            return {}
        
        metrics = {}
        
        for _ in range(self.gradient_steps):
            # Sample batch
            obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
            
            obs = torch.FloatTensor(obs).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # Update critic
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_obs)
                q1_next, q2_next = self.critic_target(next_obs, next_actions)
                q_next = torch.min(q1_next, q2_next)
                
                if self.auto_alpha:
                    alpha = self.log_alpha.exp()
                else:
                    alpha = self.alpha
                    
                target_q = rewards + (1 - dones) * self.gamma * (q_next - alpha * next_log_probs)
            
            q1, q2 = self.critic(obs, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update actor
            new_actions, log_probs = self.actor.sample(obs)
            q1_new, q2_new = self.critic(obs, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            if self.auto_alpha:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
            
            actor_loss = (alpha * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
                metrics['alpha'] = self.alpha.item()
                metrics['alpha_loss'] = alpha_loss.item()
            
            # Update target network
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            metrics['critic_loss'] = critic_loss.item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['q_value'] = q_new.mean().item()
        
        self.train_step += 1
        
        return metrics
    
    def save(self, path):
        """Save model."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        
        if self.auto_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_alpha and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
