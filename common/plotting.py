"""
Plotting utilities for training and evaluation results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

sns.set_style("whitegrid")


def plot_training_curve(rewards, lengths, save_dir, window=10):
    """
    Plot training reward and length curves.
    
    Args:
        rewards: List of episode rewards
        lengths: List of episode lengths
        save_dir: Directory to save plots
        window: Moving average window size
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    if len(rewards) > 0:
        episodes = np.arange(len(rewards))
        axes[0].plot(episodes, rewards, alpha=0.3, label='Episode Reward')
        
        # Moving average
        if len(rewards) >= window:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            axes[0].plot(episodes, moving_avg, linewidth=2, label=f'{window}-Episode MA')
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot lengths
    if len(lengths) > 0:
        episodes = np.arange(len(lengths))
        axes[1].plot(episodes, lengths, alpha=0.3, label='Episode Length')
        
        # Moving average
        if len(lengths) >= window:
            moving_avg = pd.Series(lengths).rolling(window=window).mean()
            axes[1].plot(episodes, moving_avg, linewidth=2, label=f'{window}-Episode MA')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Training Episode Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def plot_test_results(test_rewards, test_lengths, save_dir):
    """
    Plot test episode results.
    
    Args:
        test_rewards: List of test episode rewards
        test_lengths: List of test episode lengths
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reward histogram
    axes[0, 0].hist(test_rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(test_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(test_rewards):.2f}')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Test Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward over episodes
    episodes = np.arange(len(test_rewards))
    axes[0, 1].plot(episodes, test_rewards, marker='o', markersize=3, alpha=0.6)
    axes[0, 1].axhline(np.mean(test_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(test_rewards):.2f}')
    axes[0, 1].set_xlabel('Test Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Test Rewards Over Episodes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Length histogram
    axes[1, 0].hist(test_lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(test_lengths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(test_lengths):.2f}')
    axes[1, 0].set_xlabel('Episode Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Episode Length Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Length over episodes
    axes[1, 1].plot(episodes, test_lengths, marker='o', markersize=3, alpha=0.6)
    axes[1, 1].axhline(np.mean(test_lengths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(test_lengths):.2f}')
    axes[1, 1].set_xlabel('Test Episode')
    axes[1, 1].set_ylabel('Length')
    axes[1, 1].set_title('Test Lengths Over Episodes')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test results plots saved to {save_dir / 'test_results.png'}")


def plot_comparison(results_list, save_path, metric='test_mean_reward'):
    """
    Plot comparison across multiple variants.
    
    Args:
        results_list: List of result dictionaries with 'variant' and metrics
        save_path: Path to save the comparison plot
        metric: Metric to compare
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    variants = [r['variant'] for r in results_list]
    values = [r[metric] for r in results_list]
    errors = [r.get(metric.replace('mean', 'std'), 0) for r in results_list]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(variants))
    
    ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Variant')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison of {metric.replace("_", " ").title()} Across Variants')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")
