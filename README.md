# Random-Network-Distillation
This repository contains an implementation of the Random Network Distillation (RND) algorithm for improving exploration in reinforcement learning environments. It is configured for the Atari environment `ALE/FlagCapture-v5` and uses PyTorch for neural network operations.
## Features

- Supports Atari Environments: Preconfigured for `ALE/FlagCapture-v5`, with support for other environments.
- Random Network Distillation: Implements RND to encourage exploration by rewarding novel states.
- Configurable Training Parameters: Modify hyperparameters such as learning rate, discount factors, and preprocessing settings in config.conf.
- Customizable Options: Easily switch between environments and training methods.

## Files
- `rnd.py`: Main script containing the RND implementation and training logic.
- `config.conf`: Configuration file specifying training parameters and options.

## Usage
Ensure you have the following installed:

- Python 3.8 or later
- Required libraries (install using `requirements.txt`)
```
    pip install -r requirements.txt
```

## Hyperparameter Configuration
Edit the `config.conf` file to modify the training parameters. For a complete list of settings, refer to the config.conf file.

## Acknowledgment
RND implementation is derived from [this source](https://github.com/jcwleo/random-network-distillation-pytorch)
