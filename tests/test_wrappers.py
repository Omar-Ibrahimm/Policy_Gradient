"""
Unit tests for environment wrappers.
"""
import pytest
import gymnasium as gym
import numpy as np
from common.env_wrappers import DiscretePendulumWrapper, ContinuousToDiscreteWrapper


def test_discrete_pendulum_wrapper_action_space():
    """Test that DiscretePendulumWrapper creates correct discrete action space."""
    env = gym.make("Pendulum-v1")
    n_bins = 11
    wrapped_env = DiscretePendulumWrapper(env, n_bins=n_bins)
    
    # Check action space is discrete
    assert isinstance(wrapped_env.action_space, gym.spaces.Discrete)
    assert wrapped_env.action_space.n == n_bins
    
    wrapped_env.close()


def test_discrete_pendulum_wrapper_action_mapping():
    """Test that action mapping is correct."""
    env = gym.make("Pendulum-v1")
    n_bins = 11
    wrapped_env = DiscretePendulumWrapper(env, n_bins=n_bins)
    
    # Get original action bounds
    orig_low = env.action_space.low[0]
    orig_high = env.action_space.high[0]
    
    # Check mapping bounds
    assert wrapped_env.action_mapping[0] == pytest.approx(orig_low)
    assert wrapped_env.action_mapping[-1] == pytest.approx(orig_high)
    assert len(wrapped_env.action_mapping) == n_bins
    
    # Check all mapped values are within bounds
    for mapped_val in wrapped_env.action_mapping:
        assert orig_low <= mapped_val <= orig_high
    
    wrapped_env.close()


def test_discrete_pendulum_wrapper_step():
    """Test that stepping with discrete actions works."""
    env = gym.make("Pendulum-v1")
    wrapped_env = DiscretePendulumWrapper(env, n_bins=11)
    
    obs, info = wrapped_env.reset()
    
    # Take a step with discrete action
    discrete_action = 5  # Middle action
    obs, reward, done, truncated, info = wrapped_env.step(discrete_action)
    
    # Check outputs have correct types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    
    wrapped_env.close()


def test_discrete_pendulum_wrapper_episode():
    """Test running a full episode with the wrapper."""
    env = gym.make("Pendulum-v1")
    wrapped_env = DiscretePendulumWrapper(env, n_bins=11)
    
    obs, info = wrapped_env.reset()
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated) and steps < 200:
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        steps += 1
    
    assert steps > 0
    wrapped_env.close()


def test_continuous_to_discrete_wrapper():
    """Test ContinuousToDiscreteWrapper."""
    env = gym.make("CartPole-v1")
    wrapped_env = ContinuousToDiscreteWrapper(env)
    
    # Check action space is now continuous
    assert isinstance(wrapped_env.action_space, gym.spaces.Box)
    assert wrapped_env.action_space.shape == (1,)
    
    # Test action conversion
    continuous_action = np.array([0.3])
    discrete_action = wrapped_env.action(continuous_action)
    
    assert isinstance(discrete_action, (int, np.integer))
    assert 0 <= discrete_action < env.action_space.n
    
    wrapped_env.close()


def test_continuous_to_discrete_full_range():
    """Test that ContinuousToDiscreteWrapper maps full continuous range."""
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    wrapped_env = ContinuousToDiscreteWrapper(env)
    
    # Test boundary cases
    assert wrapped_env.action(np.array([0.0])) == 0
    assert wrapped_env.action(np.array([1.0])) == n_actions - 1
    
    # Test middle
    mid_action = wrapped_env.action(np.array([0.5]))
    assert 0 <= mid_action < n_actions
    
    wrapped_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
