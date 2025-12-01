"""
Common utilities for RL experiments.
"""

__version__ = "1.0.0"

from common.env_factory import make_env, is_discrete_env, is_continuous_env, get_algo_type
from common.env_wrappers import DiscretePendulumWrapper, ContinuousToDiscreteWrapper
from common.trainer import train_a2c, train_ppo, train_sac, evaluate_agent, save_model_and_config
from common.evaluator import evaluate_model, record_episodes, save_results
from common.plotting import plot_training_curve, plot_test_results, plot_comparison
from common.wandb_helper import init_wandb, log_metrics, finish_wandb, log_summary_metrics

__all__ = [
    'make_env',
    'is_discrete_env',
    'is_continuous_env',
    'get_algo_type',
    'DiscretePendulumWrapper',
    'ContinuousToDiscreteWrapper',
    'train_a2c',
    'train_ppo',
    'train_sac',
    'evaluate_agent',
    'save_model_and_config',
    'evaluate_model',
    'record_episodes',
    'save_results',
    'plot_training_curve',
    'plot_test_results',
    'plot_comparison',
    'init_wandb',
    'log_metrics',
    'finish_wandb',
    'log_summary_metrics',
]
