"""
WandB integration utilities.
"""
import wandb


def init_wandb(project_name, run_name, config, tags=None):
    """
    Initialize a WandB run.
    
    Args:
        project_name: WandB project name
        run_name: Name for this specific run
        config: Configuration dictionary to log
        tags: Optional list of tags
        
    Returns:
        WandB run object
    """
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True  # Allow multiple runs in same script
    )
    
    return run


def log_metrics(metrics_dict, step=None):
    """
    Log metrics to WandB.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Optional step number
    """
    if step is not None:
        wandb.log(metrics_dict, step=step)
    else:
        wandb.log(metrics_dict)


def finish_wandb():
    """Finish the current WandB run."""
    wandb.finish()


def log_summary_metrics(test_metrics, train_metrics=None):
    """
    Log final summary metrics to WandB.
    
    Args:
        test_metrics: Dictionary of test metrics
        train_metrics: Optional dictionary of training metrics
    """
    summary_dict = {}
    
    # Test metrics
    for key, value in test_metrics.items():
        if key not in ['all_rewards', 'all_lengths']:  # Skip raw data
            summary_dict[f'final/{key}'] = value
    
    # Training metrics
    if train_metrics:
        for key, value in train_metrics.items():
            summary_dict[f'final/{key}'] = value
    
    wandb.log(summary_dict)
    
    # Also set as summary for the run
    for key, value in summary_dict.items():
        wandb.run.summary[key] = value
