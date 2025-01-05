# FBPINNs Saturated Growth Model Training

## Overview
This Python script, `FBPINNs_SaturatedGrowth.py`, trains a Saturated Growth Model using FBPINNs. It's designed for experimenting with neural network configurations using domain decomposition in solving differential equations.

## Prerequisites
- jax==0.4.23
- jaxlib==0.4.23
- optax==0.1.9
- numpy==1.26.4
- matplotlib==3.8.3
- scipy==1.12.0
- pandas==2.2.1 
- seaborn==0.13.2
- tensorboardX==2.6.2.2
- ipython>=8.12.0

## Features
- Configurable neural network architecture.
- Select #of subdomain and overlap
- Noisy training data generation.
- Model saving.
- Data visualization.

## Usage
Execute with:
```bash
python FBPINNs_SaturatedGrowth.py
```

#### Model Parameters
- `-ic`, `--initial_conditions`: Initial conditions for the model. Default: `0.01`.
- `--tend`: End time for the model simulation. Default: `24`.
- `--tbegin`: Begin time for the model simulation. Default: `0`.
- `-tl`, `--time_limit`: Time window for the training data. Default: `[0, 24]`.
- `-nx`, `--numx`: Number of training data points. Default: `100`.

## Model Parameters

- `-nsub`, `--num_subdomain`: Number of sub-domains. Default: `2`.
- `-wo`, `--window_overlap`: Window overlap for each subdomain. Default: `1.9`.
- `-wi`, `--window_noDataRegion`: Window overlap for no data region. Default: `1.0005`.
- `--unnorm_mean`: Mean for unnormalization. Default: `0.0`.
- `--unnorm_sd`: Standard deviation for unnormalization. Default: `1.0`.
- `-l`, `--layers`: Number of layers and neurons. Default: `2 layers [1, 5, 5, 5, 1]`.
- `-pl`, `--pinns_layers`: Number of PINNs layers and neurons. Default: `3 layers [1, 5, 5, 5, 1]`.
- `-lp`, `--lambda_phy`: Weight for physics loss. Default: `1e0`.
- `-ld`, `--lambda_data`: Weight for data loss. Default: `1e0`.
- `-nc`, `--num_collocation`: Number of collocation points. Default: `200`.
- `--sampler`: Collocation sampler, one of `["grid", "uniform", "sobol", "halton"]`. Default: `["grid"]`.
- `-nt`, `--num_test`: Number of test points. Default: `200`.
- `-e`, `--epochs`: Number of epochs. Default: `50000`.
- `--rootdir`: Root directory for saving models and summaries. Default: `SGModels`.
- `--tag`: Tag for identifying the run. Default: `ablation`.
- `--sparse`: Sparsity of training data. Default: `False`.
- `--loss_landscape`: Whether to plot loss landscape for FBPINNs. Default: `False`.
- `-nl`, `--noise_level`: Noise level in training data. Default: `0.05`.
- `-pt`, `--pinn_trainer`: Whether to train PINN trainer. Default: `False`.


### Outputs
- Model Comparision.
- Window Function.
- Model.
- Parameter and metrics csv file.
- Training plot.
- Loss landscape plot(optional).

*This script is part of a project focusing on Model learning using machine learning and domain decomposition.*

