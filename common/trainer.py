"""
Training utilities for custom RL agents.
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb


def train_a2c(agent, env, config, success_threshold=None, max_episodes=10000, 
              eval_freq=50, eval_episodes=10):
    """
    Train A2C agent.
    
    Args:
        agent: A2C agent instance
        env: Gym environment
        config: Configuration dictionary
        success_threshold: Average reward threshold for success
        max_episodes: Maximum number of episodes
        eval_freq: Frequency of evaluation
        eval_episodes: Number of episodes for evaluation
        
    Returns:
        Training metrics
    """
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    step_count = 0
    
    n_steps = config.get('n_steps', 5)
    
    pbar = tqdm(total=max_episodes, desc="Training A2C")
    
    while episode_count < max_episodes:
        # Collect rollout
        for _ in range(n_steps):
            action, value, log_prob = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.store_transition(obs, action, reward, value, log_prob, done or truncated)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Log to WandB
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_length,
                    'train/episode': episode_count
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.2f}',
                    'avg_reward': f'{np.mean(episode_rewards[-100:]):.2f}'
                })
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Check for early stopping
                if episode_count >= eval_freq and episode_count % eval_freq == 0:
                    eval_reward = evaluate_agent(agent, env, eval_episodes)
                    eval_rewards.append(eval_reward)
                    
                    wandb.log({
                        'eval/mean_reward': eval_reward,
                        'train/episode': episode_count
                    })
                    
                    # Check success threshold
                    if success_threshold is not None and eval_reward >= success_threshold:
                        print(f"\nSuccess! Eval reward {eval_reward:.2f} >= {success_threshold}")
                        break
                
                if episode_count >= max_episodes:
                    break
        
        # Update policy
        if len(agent.observations) > 0:
            metrics = agent.update(obs)
            if metrics:
                wandb.log({
                    'train/loss': metrics.get('loss', 0),
                    'train/policy_loss': metrics.get('policy_loss', 0),
                    'train/value_loss': metrics.get('value_loss', 0),
                    'train/entropy': metrics.get('entropy', 0),
                    'train/episode': episode_count
                })
    
    pbar.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'train_final_avg_reward': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards))
    }


def train_ppo(agent, env, config, success_threshold=None, max_episodes=10000,
              eval_freq=50, eval_episodes=10):
    """
    Train PPO agent.
    
    Args:
        agent: PPO agent instance
        env: Gym environment
        config: Configuration dictionary
        success_threshold: Average reward threshold for success
        max_episodes: Maximum number of episodes
        eval_freq: Frequency of evaluation
        eval_episodes: Number of episodes for evaluation
        
    Returns:
        Training metrics
    """
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    step_count = 0
    
    n_steps = config.get('n_steps', 2048)
    
    pbar = tqdm(total=max_episodes, desc="Training PPO")
    
    while episode_count < max_episodes:
        # Collect rollout
        for _ in range(n_steps):
            action, value, log_prob = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.store_transition(obs, action, reward, value, log_prob, done or truncated)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Log to WandB
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_length,
                    'train/episode': episode_count
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.2f}',
                    'avg_reward': f'{np.mean(episode_rewards[-100:]):.2f}'
                })
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Check for early stopping
                if episode_count >= eval_freq and episode_count % eval_freq == 0:
                    eval_reward = evaluate_agent(agent, env, eval_episodes)
                    eval_rewards.append(eval_reward)
                    
                    wandb.log({
                        'eval/mean_reward': eval_reward,
                        'train/episode': episode_count
                    })
                    
                    # Check success threshold
                    if success_threshold is not None and eval_reward >= success_threshold:
                        print(f"\nSuccess! Eval reward {eval_reward:.2f} >= {success_threshold}")
                        break
                
                if episode_count >= max_episodes:
                    break
        
        # Update policy
        if len(agent.observations) > 0:
            metrics = agent.update(obs)
            if metrics:
                wandb.log({
                    'train/loss': metrics.get('loss', 0),
                    'train/policy_loss': metrics.get('policy_loss', 0),
                    'train/value_loss': metrics.get('value_loss', 0),
                    'train/entropy': metrics.get('entropy', 0),
                    'train/episode': episode_count
                })
    
    pbar.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'train_final_avg_reward': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards))
    }


def train_sac(agent, env, config, success_threshold=None, max_episodes=10000,
              eval_freq=50, eval_episodes=10):
    """
    Train SAC agent.
    
    Args:
        agent: SAC agent instance
        env: Gym environment
        config: Configuration dictionary
        success_threshold: Average reward threshold for success
        max_episodes: Maximum number of episodes
        eval_freq: Frequency of evaluation
        eval_episodes: Number of episodes for evaluation
        
    Returns:
        Training metrics
    """
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    episode_count = 0
    
    pbar = tqdm(total=max_episodes, desc="Training SAC")
    
    while episode_count < max_episodes:
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            if agent.train_step < agent.learning_starts:
                # Random action during warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)
            
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done or truncated)
            
            # Update
            metrics = agent.update()
            if metrics:
                wandb.log({
                    'train/critic_loss': metrics.get('critic_loss', 0),
                    'train/actor_loss': metrics.get('actor_loss', 0),
                    'train/q_value': metrics.get('q_value', 0),
                    'train/alpha': metrics.get('alpha', 0),
                    'train/step': agent.train_step
                })
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_count += 1
        
        # Log to WandB
        wandb.log({
            'train/episode_reward': episode_reward,
            'train/episode_length': episode_length,
            'train/episode': episode_count
        })
        
        pbar.update(1)
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'avg_reward': f'{np.mean(episode_rewards[-100:]):.2f}'
        })
        
        # Check for early stopping
        if episode_count >= eval_freq and episode_count % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, eval_episodes)
            eval_rewards.append(eval_reward)
            
            wandb.log({
                'eval/mean_reward': eval_reward,
                'train/episode': episode_count
            })
            
            # Check success threshold
            if success_threshold is not None and eval_reward >= success_threshold:
                print(f"\nSuccess! Eval reward {eval_reward:.2f} >= {success_threshold}")
                break
    
    pbar.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'train_final_avg_reward': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards))
    }


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate agent performance.
    
    Args:
        agent: Agent instance
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def save_model_and_config(agent, save_dir, config, variant_name):
    """
    Save trained model and its configuration.
    
    Args:
        agent: Trained agent
        save_dir: Directory to save to
        config: Configuration dictionary
        variant_name: Name of the hyperparameter variant
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = save_path / "model.pt"
    agent.save(str(model_file))
    
    # Save config
    config_copy = config.copy()
    config_copy['variant'] = variant_name
    config_file = save_path / "config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    print(f"Model and config saved to {save_path}")
