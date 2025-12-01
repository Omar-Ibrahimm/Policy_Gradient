"""
Evaluation utilities for trained RL agents.
"""
import os
import json
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def evaluate_model(agent, env, num_episodes=100, deterministic=True):
    """
    Evaluate a trained agent on an environment.
    
    Args:
        agent: Trained agent
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    metrics = {
        'test_mean_reward': float(np.mean(episode_rewards)),
        'test_std_reward': float(np.std(episode_rewards)),
        'test_mean_length': float(np.mean(episode_lengths)),
        'test_std_length': float(np.std(episode_lengths)),
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths
    }
    
    return metrics


def record_episodes(agent, env_id, algo_type, record_dir, record_indices, 
                   num_total_episodes=100, deterministic=True, n_bins=11):
    """
    Record specific episodes and save as videos.
    
    Args:
        agent: Trained agent
        env_id: Environment ID
        algo_type: Algorithm type (discrete/continuous)
        record_dir: Directory to save videos
        record_indices: List of episode indices to record (e.g., [20, 40, 60, 80, 100])
        num_total_episodes: Total episodes to run
        deterministic: Use deterministic policy
        n_bins: Number of bins for Pendulum wrapper
        
    Returns:
        None (saves videos to disk)
    """
    from common.env_factory import make_env
    
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to 0-indexed
    record_indices_set = set([idx - 1 for idx in record_indices if idx <= num_total_episodes])
    
    print(f"Recording episodes at indices: {sorted(record_indices)}")
    
    for ep_idx in range(num_total_episodes):
        if ep_idx in record_indices_set:
            # Create environment with video recording for this episode
            video_folder = str(record_dir)
            
            # Create base env
            env = make_env(env_id, algo_type=algo_type, n_bins=n_bins, 
                          render_mode="rgb_array")
            
            # Wrap with RecordVideo for just this episode
            env = RecordVideo(
                env, 
                video_folder=video_folder,
                name_prefix=f"episode_{ep_idx + 1}",
                episode_trigger=lambda x: x == 0  # Record first episode in this env
            )
            
            obs, info = env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
            
            env.close()
            print(f"Recorded episode {ep_idx + 1}")


def save_results(results_dir, metrics, variant, algo, env_id):
    """
    Save evaluation results to JSON file.
    
    Args:
        results_dir: Directory to save results
        metrics: Dictionary of metrics
        variant: Variant name
        algo: Algorithm name
        env_id: Environment ID
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    results_file = results_path / "results.json"
    
    # Prepare results dictionary with required fields
    results = {
        "train_final_avg_reward": metrics.get('train_final_avg_reward', 0.0),
        "test_mean_reward": metrics['test_mean_reward'],
        "test_std_reward": metrics['test_std_reward'],
        "test_mean_length": metrics['test_mean_length'],
        "test_std_length": metrics['test_std_length'],
        "variant": variant,
        "algo": algo,
        "env": env_id
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    return results
