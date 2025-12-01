"""
A2C (Advantage Actor-Critic) Agent Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal


class ActorCriticNetwork(nn.Module):
    """Neural network for both actor and critic."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64, continuous=False):
        super(ActorCriticNetwork, self).__init__()
        self.continuous = continuous
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
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


class A2CAgent:
    """A2C Agent with custom implementation."""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        continuous=False,
        learning_rate=7e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        n_steps=5,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Create network
        self.network = ActorCriticNetwork(
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
    
    def compute_returns(self, next_value, dones):
        """Compute discounted returns."""
        returns = []
        R = next_value
        
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        
        return returns
    
    def update(self, next_obs):
        """Update policy and value function."""
        if len(self.observations) == 0:
            return {}
        
        # Get next value for bootstrapping
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.continuous:
                _, _, next_value = self.network(next_obs_tensor)
            else:
                _, next_value = self.network(next_obs_tensor)
            next_value = next_value.cpu().item()
        
        # Compute returns
        returns = self.compute_returns(next_value, self.dones)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.observations)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(self.values).to(self.device)
        
        # Forward pass
        if self.continuous:
            mean, std, values_pred = self.network(obs_tensor)
            dist = Normal(mean, std)
            actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            logits, values_pred = self.network(obs_tensor)
            dist = Categorical(logits=logits)
            actions_tensor = torch.LongTensor(self.actions).to(self.device)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
        
        # Compute advantages
        advantages = returns_tensor - values_tensor
        
        # Actor loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        value_loss = F.mse_loss(values_pred.squeeze(), returns_tensor)
        
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
        
        # Reset storage
        self.reset_storage()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
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
