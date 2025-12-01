"""
Environment wrappers for adapting action spaces and observations.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscretePendulumWrapper(gym.Wrapper):
    """
    Wraps Pendulum-v1 to discretize its continuous action space.
    Converts discrete action indices to continuous torque values.
    """
    
    def __init__(self, env, n_bins=11):
        """
        Args:
            env: The Pendulum environment to wrap
            n_bins: Number of discrete actions to create
        """
        super().__init__(env)
        self.n_bins = n_bins
        
        # Get original action space bounds
        self.orig_action_space = env.action_space
        assert isinstance(self.orig_action_space, spaces.Box), \
            "Original action space must be continuous (Box)"
        
        low = self.orig_action_space.low[0]
        high = self.orig_action_space.high[0]
        
        # Create discrete action mapping
        self.action_mapping = np.linspace(low, high, n_bins)
        
        # Replace action space with discrete
        self.action_space = spaces.Discrete(n_bins)
    
    def step(self, action):
        """Convert discrete action to continuous and step."""
        # Map discrete action to continuous value
        continuous_action = np.array([self.action_mapping[action]], dtype=np.float32)
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)


class ContinuousToDiscreteWrapper(gym.ActionWrapper):
    """
    Generic wrapper to convert discrete actions to continuous actions.
    Useful for running SAC on discrete environments.
    """
    
    def __init__(self, env, n_bins_per_action=5):
        """
        Args:
            env: Environment with discrete action space
            n_bins_per_action: How many continuous values to map each discrete action to
        """
        super().__init__(env)
        
        assert isinstance(env.action_space, spaces.Discrete), \
            "This wrapper requires a discrete action space"
        
        self.n_discrete = env.action_space.n
        
        # Create continuous action space [0, 1] for each discrete action
        # The continuous value will be mapped back to discrete
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
    def action(self, action):
        """Convert continuous action to discrete."""
        # Map continuous [0, 1] to discrete action
        continuous_val = np.clip(action[0], 0.0, 1.0)
        discrete_action = int(continuous_val * self.n_discrete)
        # Ensure we don't exceed bounds
        discrete_action = min(discrete_action, self.n_discrete - 1)
        return discrete_action
