# Competition Model Training

## Overview
This Python script, `SciANN-CompetitionModel.py`, is designed to train a Competition Model using neural networks. It allows for flexible configuration of neural network parameters and is suitable for research in competitive dynamics.

## Prerequisites
- numpy==1.21.6
- sciann==0.7.0.1
- matplotlib==3.5.3
- scipy==1.10.1
- pandas==1.3.5
- plotly==5.18.0
- tensorflow==2.11.0
- ipykernel==6.29.3 
- ipywidgets==8.1.2 
- h5py==3.10.0 
- seaborn==0.13.2

## Features
- Customizable neural network architecture.
- Options for generating training data with noise.
- Saving and loading of model weights.
- Visualization capabilities for model analysis.

## Usage
Execute with:
```bash
python SciANN_CompetitionModel.py
```
### Additional Arguments
- `-l`, `--layers`: Number of layers and neurons per layer. Default: `[5, 5, 5]`.
- `-af`, `--actf`: Activation function. Default: `tanh`.
- `-nx`, `--numx`: Number of nodes in X. Default: `100`.
- `-bs`, `--batchsize`: Batch size for Adam optimizer. Default: `25`.
- `-e`, `--epochs`: Maximum number of epochs. Default: `2000`.
- `-lr`, `--learningrate`: Initial learning rate. Default: `0.001`.
- `-rlr`, `--reduce_learning_rate`: Epoch interval to reduce learning rate. Default: `100`.
- `-in`, `--independent_networks`: Use independent networks for each variable. Default: `True`.
- `-v`, `--verbose`: Verbosity level (check Keras.fit). Default: `2`.

#### Model Parameters
- `-ic`, `--initial_conditions`: Initial conditions for the model. Default: `0.01`.
- `--tend`: End time for the model simulation. Default: `24`.
- `--model_type`: Survival or co-existence model. Default: `Survival`

#### Data Settings
- `--sparse`: Sparsity of training data. Default: `True`.
- `--noise_level`: Level of noise in training data. Default: `0.005`.
- `-tl`, `--time_limit`: Time window for the training data. Default: `[10, 30]`.
- `-sf`, `--show_figure`: Show training data plot. Default: `True`.
- `--shuffle`: Shuffle data for training. Default: `True`.
- `--stopafter`: Patience argument from Keras. Default: `500`.
- `--savefreq`: Frequency to save weights (each n-epoch). Default: `100000`.
- `--dtype`: Data type for weights and biases. Default: `float64`.
- `--gpu`: Use GPU if available. Default: `False`.
- `-op`, `--outputpath`: Output path. Default: `./output`.
- `-of`, `--outputprefix`: Output prefix. Default: `res`.
- `--plot`: Plot the model. Default: `False`.


### Outputs
- Training plots (optional).
- Model summary and training logs.
- Model weights (HDF5 format).
- Post-processing: `Loss`, `Model comparison`

*This script is part of a project focusing on Neural Networks in long-term behavior analysis and mechanics learning.*