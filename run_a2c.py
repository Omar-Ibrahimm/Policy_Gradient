"""
A2C Experiment Runner
Runs baseline + hyperparameter variants for A2C algorithm.
"""
import argparse
import json
import time
from pathlib import Path

from a2c_agent import A2CAgent
from common.env_factory import make_env, get_algo_type
from common.trainer import train_a2c, save_model_and_config
from common.evaluator import evaluate_model, record_episodes, save_results
from common.plotting import plot_training_curve, plot_test_results, plot_comparison
from common.wandb_helper import init_wandb, log_summary_metrics, finish_wandb


def load_config(config_path="configs/a2c.json"):
    """Load A2C configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_variants(base_config):
    """
    Generate hyperparameter variants from base configuration.
    Returns list of (variant_name, config_dict) tuples.
    """
    variants = []
    base_hp = base_config['hyperparameters']
    tunable = base_config['tunable_params']
    
    # Baseline variant
    variants.append(('baseline', base_hp.copy()))
    
    # Generate high/low variants for each tunable parameter
    for param_name, param_config in tunable.items():
        base_value = base_hp[param_name]
        
        # High variant
        high_config = base_hp.copy()
        if param_config.get('type') == 'absolute':
            high_config[param_name] = param_config['high_value']
        elif param_config.get('type') == 'int_absolute':
            high_config[param_name] = int(param_config['high_value'])
        elif param_config.get('type') == 'int':
            high_val = int(base_value * param_config['high_multiplier'])
            high_val = max(high_val, param_config.get('min_value', 1))
            high_config[param_name] = high_val
        else:
            high_config[param_name] = base_value * param_config['high_multiplier']
        
        variants.append((f'{param_name}_high', high_config))
        
        # Low variant
        low_config = base_hp.copy()
        if param_config.get('type') == 'absolute':
            low_config[param_name] = param_config['low_value']
        elif param_config.get('type') == 'int_absolute':
            low_config[param_name] = int(param_config['low_value'])
        elif param_config.get('type') == 'int':
            low_val = int(base_value * param_config['low_multiplier'])
            low_val = max(low_val, param_config.get('min_value', 1))
            low_config[param_name] = low_val
        else:
            low_config[param_name] = base_value * param_config['low_multiplier']
        
        variants.append((f'{param_name}_low', low_config))
    
    return variants


def run_single_experiment(env_id, variant_name, hyperparams, config, 
                         project_name, num_test_runs, record_episodes_list, n_bins=11):
    """Run a single experiment with given hyperparameters."""
    
    algo_name = "A2C"
    run_name = f"{env_id}-{algo_name}-{variant_name}"
    
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"{'='*60}\n")
    
    # Initialize WandB
    wandb_run = init_wandb(
        project_name=project_name,
        run_name=run_name,
        config={
            'algorithm': algo_name,
            'env': env_id,
            'variant': variant_name,
            **hyperparams,
            'max_episodes': config.get('max_episodes', 2000)
        },
        tags=[algo_name, env_id, variant_name]
    )
    
    # Create environment
    algo_type = get_algo_type(algo_name)
    env = make_env(env_id, algo_type=algo_type, n_bins=n_bins)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    # Create agent
    agent = A2CAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=continuous,
        **hyperparams
    )
    
    # Train agent
    train_metrics = train_a2c(
        agent=agent,
        env=env,
        config=hyperparams,
        success_threshold=config.get('success_threshold'),
        max_episodes=config.get('max_episodes', 2000),
        eval_freq=config.get('eval_freq', 50),
        eval_episodes=config.get('eval_episodes', 10)
    )
    
    # Plot training curves
    plots_dir = f"plots/{env_id.lower()}/{algo_name.lower()}/{variant_name}"
    if len(train_metrics['episode_rewards']) > 0:
        plot_training_curve(
            train_metrics['episode_rewards'],
            train_metrics['episode_lengths'],
            plots_dir
        )
    
    # Save model
    model_save_dir = f"saved_models/{env_id.lower()}/{algo_name.lower()}/{variant_name}"
    save_model_and_config(agent, model_save_dir, hyperparams, variant_name)
    
    # Evaluate agent
    eval_env = make_env(env_id, algo_type=algo_type, n_bins=n_bins)
    test_metrics = evaluate_model(
        agent=agent,
        env=eval_env,
        num_episodes=num_test_runs,
        deterministic=True
    )
    eval_env.close()
    
    # Plot test results
    plot_test_results(
        test_metrics['all_rewards'],
        test_metrics['all_lengths'],
        plots_dir
    )
    
    # Record episodes
    record_dir = f"recorded_videos/{env_id.lower()}/{algo_name.lower()}/{variant_name}"
    record_episodes(
        agent=agent,
        env_id=env_id,
        algo_type=algo_type,
        record_dir=record_dir,
        record_indices=record_episodes_list,
        num_total_episodes=num_test_runs,
        deterministic=True,
        n_bins=n_bins
    )
    
    # Save results
    results_dir = f"results/{env_id.lower()}/{algo_name.lower()}/{variant_name}"
    combined_metrics = {**train_metrics, **test_metrics}
    results = save_results(
        results_dir=results_dir,
        metrics=combined_metrics,
        variant=variant_name,
        algo=algo_name,
        env_id=env_id
    )
    
    # Log summary to WandB
    log_summary_metrics(test_metrics, train_metrics)
    
    # Finish WandB run
    finish_wandb()
    
    env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run A2C experiments')
    parser.add_argument('--env', type=str, required=True,
                       help='Environment ID (e.g., CartPole-v1)')
    parser.add_argument('--project', type=str, default=None,
                       help='WandB project name')
    parser.add_argument('--num-test-runs', type=int, default=100,
                       help='Number of test episodes')
    parser.add_argument('--record-episodes', type=int, nargs='+',
                       default=[20, 40, 60, 80, 100],
                       help='Episode indices to record')
    parser.add_argument('--n-bins', type=int, default=11,
                       help='Number of discrete bins for Pendulum wrapper')
    parser.add_argument('--sleep-time', type=float, default=5.0,
                       help='Sleep time (seconds) between runs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Apply environment-specific overrides
    if args.env in config.get('env_overrides', {}):
        overrides = config['env_overrides'][args.env]
        for key, value in overrides.items():
            if key in ['max_episodes', 'success_threshold', 'eval_freq', 'eval_episodes']:
                config[key] = value
            elif key in config['hyperparameters']:
                config['hyperparameters'][key] = value
    
    # Set project name
    if args.project is None:
        args.project = f"rl-A2C-{args.env}"
    
    print(f"\n{'#'*60}")
    print(f"# A2C Experiment Suite")
    print(f"# Environment: {args.env}")
    print(f"# WandB Project: {args.project}")
    print(f"# Max Episodes: {config.get('max_episodes', 2000)}")
    print(f"# Success Threshold: {config.get('success_threshold', 'None')}")
    print(f"# Sleep Time: {args.sleep_time}s")
    print(f"{'#'*60}\n")
    
    # Generate variants
    variants = generate_variants(config)
    print(f"Generated {len(variants)} variants:")
    for variant_name, _ in variants:
        print(f"  - {variant_name}")
    print()
    
    # Run all experiments
    all_results = []
    for idx, (variant_name, hyperparams) in enumerate(variants):
        try:
            results = run_single_experiment(
                env_id=args.env,
                variant_name=variant_name,
                hyperparams=hyperparams,
                config=config,
                project_name=args.project,
                num_test_runs=args.num_test_runs,
                record_episodes_list=args.record_episodes,
                n_bins=args.n_bins
            )
            all_results.append(results)
            
            # Sleep between runs to avoid WandB issues
            if idx < len(variants) - 1:
                print(f"\nSleeping for {args.sleep_time} seconds before next run...\n")
                time.sleep(args.sleep_time)
                
        except Exception as e:
            print(f"Error in variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Test Mean Reward':<20} {'Test Std Reward':<20}")
    print(f"{'-'*60}")
    for result in all_results:
        print(f"{result['variant']:<20} {result['test_mean_reward']:<20.2f} "
              f"{result['test_std_reward']:<20.2f}")
    print(f"{'='*60}\n")
    
    # Create comparison plot
    if len(all_results) > 0:
        comparison_dir = f"plots/{args.env.lower()}/a2c"
        Path(comparison_dir).mkdir(parents=True, exist_ok=True)
        plot_comparison(
            all_results,
            f"{comparison_dir}/variant_comparison.png",
            metric='test_mean_reward'
        )
    
    print(f"All experiments completed! Results saved to results/, plots/, and saved_models/")


if __name__ == "__main__":
    main()
