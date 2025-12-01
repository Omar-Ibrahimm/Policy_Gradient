# RL Experiment Pipeline

A complete, reproducible reinforcement learning experiment pipeline with **custom PyTorch implementations** of A2C, PPO, and SAC algorithms on classic control environments with automated hyperparameter tuning.

## Features

- **Custom RL Implementations**: Built from scratch in PyTorch (no Stable-Baselines3)
  - A2C (Advantage Actor-Critic) 
  - PPO (Proximal Policy Optimization) with GAE
  - SAC (Soft Actor-Critic) with automatic entropy tuning
- **Four Environments**: CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1
- **Smart Training**: Success threshold and maximum episode limits
- **Automated Hyperparameter Tuning**: Baseline + high/low variants for each tunable parameter
- **Comprehensive Logging**: WandB integration with training curves and metrics
- **Video Recording**: Automatic recording of test episodes at specified indices
- **Visualization**: Training curves, test distributions, and variant comparisons
- **Environment Adapters**: Automatic wrappers for algorithm-environment compatibility
- **Safe WandB Usage**: Built-in sleep time between runs to prevent API crashes

## Project Structure

```
project-root/
├── a2c_agent.py                # Custom A2C implementation
├── ppo_agent.py                # Custom PPO implementation  
├── sac_agent.py                # Custom SAC implementation
├── run_a2c_experiments.py      # A2C experiment runner
├── run_ppo_experiments.py      # PPO experiment runner
├── run_sac_experiments.py      # SAC experiment runner
├── common/
│   ├── env_factory.py          # Environment creation and selection
│   ├── env_wrappers.py         # Environment wrappers
│   ├── trainer.py              # Training loops for each algorithm
│   ├── evaluator.py            # Evaluation and video recording
│   ├── plotting.py             # Plotting utilities
│   └── wandb_helper.py         # WandB integration
├── configs/
│   ├── a2c.json                # A2C hyperparameters and success thresholds
│   ├── ppo.json                # PPO hyperparameters and success thresholds
│   └── sac.json                # SAC hyperparameters and success thresholds
├── tests/
│   └── test_env_wrappers.py    # Unit tests
├── saved_models/               # Trained model checkpoints
├── recorded_videos/            # Recorded episode videos
├── results/                    # JSON results summaries
├── plots/                      # Training and evaluation plots
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- ffmpeg (for video recording)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd rl-experiment-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up WandB:
```bash
wandb login
```

## Usage

### Basic Usage

Run experiments with default settings:

```bash
# A2C on CartPole
python run_a2c_experiments.py --env CartPole-v1

# PPO on Acrobot
python run_ppo_experiments.py --env Acrobot-v1

# SAC on Pendulum
python run_sac_experiments.py --env Pendulum-v1
```

### Advanced Options

All experiment scripts support the following arguments:

- `--env`: Environment ID (required)
- `--project`: WandB project name (default: `rl-{ALGO}-{ENV}`)
- `--num-test-runs`: Number of test episodes (default: 100)
- `--record-episodes`: Episode indices to record as videos (default: [20, 40, 60, 80, 100])
- `--n-bins`: Number of discrete bins for Pendulum wrapper (default: 11)
- `--sleep-time`: Sleep duration in seconds between runs (default: 5.0)

**Example:**

```bash
python run_a2c_experiments.py \
    --env CartPole-v1 \
    --project my-rl-research \
    --num-test-runs 200 \
    --record-episodes 50 100 150 200 \
    --sleep-time 10.0
```

## Training Behavior

### Success Thresholds

Training will stop early if the agent achieves the success threshold during evaluation:

| Environment | Success Threshold | Max Episodes |
|------------|------------------|--------------|
| CartPole-v1 | 475.0 | 1000 |
| Acrobot-v1 | -100.0 | 2000 |
| MountainCar-v0 | -110.0 | 3000 |
| Pendulum-v1 | -200.0 | 1500 |

These can be customized in the config files (`configs/*.json`).

### Training Loop

Each algorithm follows this training pattern:

1. **Collect Experience**: Agent interacts with environment
2. **Update Policy**: Algorithm-specific update (A2C rollouts, PPO epochs, SAC replay buffer)
3. **Evaluate**: Periodic evaluation every `eval_freq` episodes
4. **Check Success**: If eval reward ≥ threshold, training stops early
5. **Max Episodes**: Training stops if max episodes reached

### Sleep Time Between Runs

To prevent WandB API rate limiting when running multiple variants:
- Default 5-second sleep between runs
- Configurable via `--sleep-time` argument
- Recommended: 5-10 seconds for stability

## Custom Agent Implementations

### A2C (Advantage Actor-Critic)

**Architecture:**
- Shared feature extractor (2-layer MLP, 64 hidden units)
- Separate actor and critic heads
- Discrete: Categorical distribution
- Continuous: Gaussian distribution with learnable std

**Key Features:**
- N-step returns for bootstrapping
- Entropy regularization
- Gradient clipping

### PPO (Proximal Policy Optimization)

**Architecture:**
- Shared feature extractor (2-layer MLP, 64 hidden units)
- Separate actor and critic heads
- Uses Tanh activations

**Key Features:**
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Multiple update epochs on collected data
- Mini-batch updates
- Entropy bonus

### SAC (Soft Actor-Critic)

**Architecture:**
- Actor: 2-layer MLP (256 hidden units)
- Twin critics: Two Q-networks
- Target network with soft updates

**Key Features:**
- Experience replay buffer
- Reparameterization trick for continuous actions
- Automatic entropy temperature tuning
- Twin Q-networks (reduces overestimation)
- Soft target updates

## Algorithm Details

### A2C (5 tunable hyperparameters → 11 variants)

**Baseline Hyperparameters:**
- `learning_rate`: 7e-4 (×2.0, ×0.5)
- `gamma`: 0.99 (→ 0.999, → 0.95)
- `entropy_coef`: 0.01 (×2.0, ×0.5)
- `value_loss_coef`: 0.5 (×2.0, ×0.5)
- `n_steps`: 5 (×2, ÷2, min=1)

### PPO (8 tunable hyperparameters → 17 variants)

**Baseline Hyperparameters:**
- `learning_rate`: 3e-4 (×2.0, ×0.5)
- `gamma`: 0.99 (→ 0.999, → 0.95)
- `gae_lambda`: 0.95 (→ 0.98, → 0.9)
- `clip_range`: 0.2 (→ 0.3, → 0.1)
- `entropy_coef`: 0.01 (×2.0, ×0.5)
- `batch_size`: 64 (→ 128, → 32)
- `n_epochs`: 10 (→ 20, → 3)
- `n_steps`: 2048 (→ 4096, → 1024)

### SAC (5 tunable hyperparameters → 11 variants)

**Baseline Hyperparameters:**
- `learning_rate`: 3e-4 (×2.0, ×0.5)
- `gamma`: 0.99 (→ 0.999, → 0.95)
- `tau`: 0.005 (×2.0, ×0.5)
- `batch_size`: 256 (→ 512, → 128)
- `buffer_size`: 100000 (→ 200000, → 50000)

## Environment Wrappers

### DiscretePendulumWrapper

Converts Pendulum-v1's continuous action space to discrete for use with A2C/PPO.

**Parameters:**
- `n_bins`: Number of discrete actions (default: 11)
- Maps discrete indices to equally-spaced continuous torque values

**Usage:**
```python
from common.env_wrappers import DiscretePendulumWrapper
import gymnasium as gym

env = gym.make("Pendulum-v1")
wrapped_env = DiscretePendulumWrapper(env, n_bins=11)
```

### ContinuousToDiscreteWrapper

Converts discrete action spaces to continuous for use with SAC.

**Usage:**
```python
from common.env_wrappers import ContinuousToDiscreteWrapper
import gymnasium as gym

env = gym.make("CartPole-v1")
wrapped_env = ContinuousToDiscreteWrapper(env)
```

## Output Structure

After running an experiment, outputs are organized as follows:

### 1. Saved Models
```
saved_models/{env}/{algo}/{variant}/
├── model.pt            # PyTorch model checkpoint
└── config.json         # Hyperparameters used
```

### 2. Results
```
results/{env}/{algo}/{variant}/
└── results.json        # Test metrics summary
```

**results.json format:**
```json
{
  "train_final_avg_reward": 195.5,
  "test_mean_reward": 198.2,
  "test_std_reward": 12.4,
  "test_mean_length": 198.2,
  "test_std_length": 12.4,
  "variant": "baseline",
  "algo": "A2C",
  "env": "CartPole-v1"
}
```

### 3. Plots
```
plots/{env}/{algo}/{variant}/
├── training_curves.png     # Training reward and length over time
├── test_results.png        # Test distributions and episode plots
└── variant_comparison.png  # Comparison across all variants (in parent dir)
```

### 4. Videos
```
recorded_videos/{env}/{algo}/{variant}/
├── episode_20-episode-0.mp4
├── episode_40-episode-0.mp4
├── episode_60-episode-0.mp4
├── episode_80-episode-0.mp4
└── episode_100-episode-0.mp4
```

## WandB Integration

Each experiment automatically logs to WandB:

- **Project name**: `rl-{ALGO}-{ENV}` (or custom via `--project`)
- **Run name**: `{ENV}-{ALGO}-{variant}`
- **Logged metrics**:
  - Training: episode reward, episode length, loss components
  - Evaluation: mean/std reward and length
  - Algorithm-specific: entropy, Q-values, alpha (SAC), etc.

**Viewing results:**
```
https://wandb.ai/{your-username}/rl-A2C-CartPole-v1
```

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=common --cov-report=html

# Run specific test file
pytest tests/test_env_wrappers.py -v
```

**Test coverage:**
- Environment wrapper action space conversions
- Discrete-continuous mapping correctness
- Episode execution with wrappers

## Example Workflows

### 1. Quick Test Run

For testing the pipeline quickly:

```bash
# Edit configs/a2c.json to reduce max_episodes
# Change "max_episodes": 2000 to "max_episodes": 200

python run_a2c_experiments.py --env CartPole-v1
```

### 2. Comparing Algorithms

```bash
python run_a2c_experiments.py --env CartPole-v1 --project cartpole-comparison
python run_ppo_experiments.py --env CartPole-v1 --project cartpole-comparison
python run_sac_experiments.py --env CartPole-v1 --project cartpole-comparison
```

### 3. Longer Sleep for Stability

```bash
python run_ppo_experiments.py --env Acrobot-v1 --sleep-time 10.0
```

## Troubleshooting

### Common Issues

**1. WandB rate limiting:**
```bash
# Increase sleep time
python run_a2c_experiments.py --env CartPole-v1 --sleep-time 10.0
```

**2. Out of memory:**
- Reduce `buffer_size` in SAC config
- Reduce `batch_size` in configs
- Use smaller `n_steps` for PPO/A2C

**3. Video recording fails:**
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS
```

**4. Import errors:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**5. CUDA out of memory:**
```python
# Edit agent files to use CPU
device='cpu'  # Instead of 'cuda'
```

## Performance Tips

1. **Use GPU**: PyTorch automatically uses CUDA if available
2. **Parallelize**: Run different algorithms/environments in separate terminals
3. **Monitor WandB**: Watch training curves in real-time
4. **Adjust hyperparameters**: Start with baseline, then experiment

## Key Differences from Stable-Baselines3

This implementation is **built from scratch** with:
- ✅ Custom PyTorch neural networks
- ✅ Manual training loops with episode/success tracking
- ✅ Custom replay buffer (SAC)
- ✅ GAE implementation (PPO)
- ✅ Full control over architecture and hyperparameters
- ✅ Educational and transparent code

## Citation

This pipeline implements algorithms from:
- **A2C**: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **SAC**: Haarnoja et al., "Soft Actor-Critic" (2018)

## License

MIT License - See LICENSE file for details

## Contact

For issues, questions, or contributions, please open an issue on GitHub.
