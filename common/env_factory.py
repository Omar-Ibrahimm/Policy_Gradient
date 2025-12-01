"""
Environment factory for creating and configuring gym environments.
"""
import gymnasium as gym
from common.env_wrappers import DiscretePendulumWrapper, ContinuousToDiscreteWrapper


SUPPORTED_ENVS = [
    "CartPole-v1",
    "Acrobot-v1", 
    "MountainCar-v0",
    "Pendulum-v1"
]


def make_env(env_id, algo_type="discrete", n_bins=11, render_mode=None):
    """
    Create an environment, applying necessary wrappers based on algorithm type.
    
    Args:
        env_id: Gym environment ID
        algo_type: "discrete" (A2C, PPO) or "continuous" (SAC)
        n_bins: Number of discrete bins for Pendulum wrapper
        render_mode: Rendering mode for the environment
        
    Returns:
        Configured gym environment
    """
    if env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported environment: {env_id}. "
                        f"Supported: {SUPPORTED_ENVS}")
    
    # Create base environment
    env = gym.make(env_id, render_mode=render_mode)
    
    # Apply wrappers based on environment and algorithm compatibility
    if env_id == "Pendulum-v1" and algo_type == "discrete":
        # Wrap Pendulum for discrete action algorithms
        env = DiscretePendulumWrapper(env, n_bins=n_bins)
        print(f"Applied DiscretePendulumWrapper with {n_bins} bins")
    
    elif is_discrete_env(env_id) and algo_type == "continuous":
        # Wrap discrete envs for continuous action algorithms (SAC)
        env = ContinuousToDiscreteWrapper(env)
        print(f"Applied ContinuousToDiscreteWrapper for {env_id}")
    
    return env


def is_discrete_env(env_id):
    """Check if environment has discrete action space."""
    discrete_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    return env_id in discrete_envs


def is_continuous_env(env_id):
    """Check if environment has continuous action space."""
    return env_id == "Pendulum-v1"


def get_algo_type(algo_name):
    """
    Determine if algorithm expects discrete or continuous actions.
    
    Args:
        algo_name: Algorithm name (A2C, PPO, or SAC)
        
    Returns:
        "discrete" or "continuous"
    """
    algo_name = algo_name.upper()
    
    if algo_name in ["A2C", "PPO"]:
        # These can handle both, but we default to discrete
        # Can be configured per-environment
        return "discrete"
    elif algo_name == "SAC":
        return "continuous"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
