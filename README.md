# Flappy Transformer: Deep Q-Network with Transformer Architecture

This project implements a Deep Q-Network (DQN) using Transformer architecture for playing Flappy Bird, based on the research paper "Enhancing Security in Visible Light Communication: A Tabu-Search-Based Method for Transmitter Selection" published in Sensors journal (https://www.mdpi.com/1424-8220/24/6/1905).

## Overview

This implementation combines the power of Transformer neural networks with Deep Q-Learning to create an intelligent agent capable of playing Flappy Bird. The model uses a sequence-based approach where the agent considers a history of past states to make decisions, leveraging the attention mechanisms of Transformers to focus on relevant past experiences.

## Key Features

- **Transformer-based DQN**: Uses multi-head attention and positional encoding for sequence processing
- **Experience Replay**: Implements prioritized experience replay for efficient learning
- **Dueling Network Architecture**: Separates value and advantage streams for better Q-value estimation
- **Comprehensive Testing**: Includes stress testing suite for model validation
- **Configurable Hyperparameters**: YAML-based configuration system

## Architecture

The model consists of:

1. **PreProcessing Layer**: Linear transformation with positional encoding
2. **Multi-Head Attention**: 8-head attention mechanism for sequence understanding
3. **Feed-Forward Networks**: Position-wise feed-forward layers
4. **Dueling DQN**: Separate value and advantage streams
5. **Experience Replay Buffer**: Efficient storage and sampling of past experiences

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flappy_transformer
```

2. Create a virtual environment:
```bash
python -m venv flappy_env
source flappy_env/bin/activate  # On Windows: flappy_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script:
```bash
python agent.py
```

The model will train for the number of episodes specified in `hyperparameters.yml` and save the trained model as `trained_model.pth`.

### Running Tests

Execute the comprehensive stress testing suite:
```bash
python test_stress.py
```

This will run various tests including:
- Unit component tests
- Gradient flow analysis
- Numerical stability checks
- Memory management validation
- Performance benchmarks
- Convergence testing
- Hyperparameter sensitivity analysis

### Configuration

Modify `hyperparameters.yml` to adjust training parameters:

```yaml
flappybird1:
  env_id: "FlappyBird-v0"
  num_episodes: 100000
  learning_rate: 0.0003
  network_sync_rate: 100
  num_hidden_1: 256
  num_hidden_2: 256
  d_model: 128
  d_ff: 512
  heads: 8
  dropout: 0.1
  N: 6
  seq_len: 16
  maxlen: 50000
  batch_size: 64
  discount_factor_g: 0.99
```

## Project Structure

```
flappy_transformer/
├── agent.py              # Main training agent
├── model.py              # Transformer DQN model implementation
├── experience_replay.py  # Experience replay buffer
├── hyperparameters.yml  # Configuration file
├── test_stress.py        # Comprehensive testing suite
├── new.py                # Development/testing utilities
└── requirements.txt      # Python dependencies
```

## Model Architecture Details

### Transformer Components

- **Positional Encoding**: Sinusoidal positional encoding for sequence awareness
- **Multi-Head Attention**: 8 attention heads with scaled dot-product attention
- **Layer Normalization**: Custom implementation with learnable parameters
- **Feed-Forward Networks**: Two-layer MLP with ReLU activation
- **Residual Connections**: Skip connections for gradient flow

### DQN Implementation

- **Dueling Architecture**: Separate value and advantage estimation
- **Target Network**: Periodic updates for stable learning
- **Experience Replay**: Efficient sampling of past experiences
- **Epsilon-Greedy**: Exploration strategy with decay

## Research Foundation

This implementation is based on the research paper:
> "Enhancing Security in Visible Light Communication: A Tabu-Search-Based Method for Transmitter Selection"
> 
> Published in Sensors journal, Volume 24, Issue 6, Article 1905
> 
> DOI: https://doi.org/10.3390/s24061905

The paper explores advanced optimization techniques that inspired the architectural decisions in this Transformer-based DQN implementation.

## Performance

The model is designed to handle:
- **Sequence Length**: 16 timesteps of history
- **Batch Size**: 64 samples per training step
- **Memory**: Efficient GPU memory usage with gradient checkpointing
- **Training**: Stable convergence with target network updates

## Testing

The project includes a comprehensive testing suite that validates:
- Model architecture integrity
- Gradient flow and backpropagation
- Numerical stability with edge cases
- Memory management and leak detection
- Performance benchmarks
- Learning convergence
- Hyperparameter sensitivity

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- Flappy Bird Gymnasium
- PyYAML
- NumPy
- Matplotlib (for testing)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on research from Sensors journal (MDPI)
- Flappy Bird Gymnasium environment
- PyTorch and Gymnasium communities
