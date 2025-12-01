"""
PPO (Proximal Policy Optimization) Agent Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal


class PPONetwork(nn.Module):
    """Neural network for PPO actor-critic."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64, continuous=False):
        super(PPONetwork, self).__init__()
        self.continuous = continuous
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.shared(x)
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std)
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value
    
    def get_value(self, x):
        features = self.shared(x)
        return self.critic(features)


class PPOAgent:
    """PPO Agent with custom implementation."""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        continuous=False,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        batch_size=64,
        n_epochs=10,
        n_steps=2048,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Create network
        self.network = PPONetwork(
            obs_dim, action_dim, continuous=continuous
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for rollouts
        self.reset_storage()
        
    def reset_storage(self):
        """Reset rollout storage."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def select_action(self, obs, deterministic=False):
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                mean, std, value = self.network(obs_tensor)
                if deterministic:
                    action = mean
                    action_np = action.cpu().numpy()[0]
                    return action_np
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    action_np = action.cpu().numpy()[0]
                    log_prob_np = log_prob.cpu().item()
            else:
                logits, value = self.network(obs_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    action_np = action.cpu().item()
                    return action_np
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    action_np = action.cpu().item()
                    log_prob = dist.log_prob(action)
                    log_prob_np = log_prob.cpu().item()
            
            value_np = value.cpu().item()
        
        return action_np, value_np, log_prob_np
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in rollout buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step] 
                + self.gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_obs):
        """Update policy and value function using PPO."""
        if len(self.observations) == 0:
            return {}
        
        # Get next value for GAE
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.network.get_value(next_obs_tensor).cpu().item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)) if self.continuous else torch.LongTensor(self.actions)
        actions_tensor = actions_tensor.to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        n_updates = 0
        
        dataset_size = len(self.observations)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                if self.continuous:
                    mean, std, values = self.network(batch_obs)
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    logits, values = self.network(batch_obs)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                
                # PPO clipped loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                n_updates += 1
        
        # Reset storage
        self.reset_storage()
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def save(self, path):
        """Save model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
