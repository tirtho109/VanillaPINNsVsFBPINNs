"""
    This file contains the implementation of GUI for SciANN_SaturatedGrowthModel.py file.
    it shold be run in the same directory as SciANN_SaturatedGrowthModel.py file.
    it should use the argument parser to get the input parameters in the gui interface to run in the terminal.
    The argument parser's are as follows in tha SciANN_SaturatedGrowthModel.py file:
    # Input interface for python. 
    parser = argparse.ArgumentParser(description='''
            SciANN code for Separating longtime behavior and learning of mechanics  \n
            Saturated Growth Model'''
    )

    parser.add_argument('-l', '--layers', help='Num layers and neurons (default 4 layers each 40 neurons [5, 5, 5])', type=int, nargs='+', default=[5]*3)
    parser.add_argument('-af', '--actf', help='Activation function (default tanh)', type=str, nargs=1, default=['tanh'])
    parser.add_argument('-nx', '--numx', help='Num Node in X (default 100)', type=int, nargs=1, default=[100])
    parser.add_argument('-bs', '--batchsize', help='Batch size for Adam optimizer (default 25)', type=int, nargs=1, default=[100])
    parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 2000)', type=int, nargs=1, default=[2000])
    parser.add_argument('-lr', '--learningrate', help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])

    parser.add_argument('-in', '--independent_networks', help='Use independent networks for each var (default True)', type=bool, nargs=1, default=[True])
    parser.add_argument('-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

    # model parameters
    parser.add_argument('-ic', '--initial_conditions', help='Initial conditions for the model (default 0.01)', type=float, default=0.01)
    parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
    parser.add_argument('--nCol', help='Number of collocation points(default 200)', type=int, default=200)
    parser.add_argument('--nTest', help='Number of collocation points(default 500)', type=int, default=500)

    # arguments for training data generator
    parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[True])
    parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [10, 30])', type=int, nargs=2, default=[0, 24])
    parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.05)', type=float, default=0.005)
    parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])

    parser.add_argument('--shuffle', help='Shuffle data for training (default True)', type=bool, nargs=1, default=[True])
    parser.add_argument('--stopafter', help='Patience argument from Keras (default 500)', type=int, nargs=1, default=[500])
    parser.add_argument('--savefreq', help='Frequency to save weights (each n-epoch)', type=int, nargs=1, default=[100000])
    parser.add_argument('--dtype', help='Data type for weights and biases (default float64)', type=str, nargs=1, default=['float64'])
    parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
    parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['SGModels'])
    parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

    parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

    args = parser.parse_args()
"""

import argparse
import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from tkinter import StringVar
from tkinter import IntVar

def run_sciann_gui():
    # Get the user-specified parameters from the GUI
    layers = list(map(int, layers_var.get().split()))
    time_limit = list(map(int, time_limit_var.get().split()))
    print(layers, time_limit)
    model_args = {
        'layers': layers,
        'actf': actf_var.get(),
        'numx': numx_var.get(),
        'batchsize': batchsize_var.get(),
        'epochs': epochs_var.get(),
        'learningrate': learningrate_var.get(),
        'independent_networks': independent_networks_var.get(),
        'verbose': verbose_var.get(),
        'initial_conditions': initial_conditions_var.get(),
        'tend': tend_var.get(),
        'nCol': nCol_var.get(),
        'nTest': nTest_var.get(),
        'sparse': sparse_var.get(),
        'outputpath': outputpath_var.get(),
        'outputprefix': outputprefix_var.get(),
        'plot': plot_var.get(),
        'shuffle': shuffle_var.get(),
        'stopafter': stopafter_var.get(),
        'savefreq': savefreq_var.get(),
        'dtype': dtype_var.get(),
        'gpu': gpu_var.get(),
        'noise_level': noise_level_var.get(),
        'time_limit': time_limit,
        'show_figure': show_figure_var.get()
    }
    # Prepare the command
    command = ['python', './SciANN_SaturatedGrowthModel.py']

    for key, value in model_args.items():
        command.append('--' + key)
        if isinstance(value, list):
            command.extend(map(str, value))  # add each item in the list separately
        else:
            command.append(str(value))

    # Run the command
    print("Executing command:", ' '.join(command))
    subprocess.run(command)

    
# Create the main window
root = tk.Tk()

# Create variables for each parameter and initialize them with the default values##############################
# layers_var should take a string of space-separated integers and convert it to a list of integers
layers_var = tk.StringVar(value=' '.join(map(str, [5]*3)))
actf_var = tk.StringVar(value='tanh')
numx_var = tk.IntVar(value=100)
batchsize_var = tk.IntVar(value=100)
epochs_var = tk.IntVar(value=2000)
learningrate_var = tk.DoubleVar(value=0.001)

independent_networks_var = tk.BooleanVar(value=True)
verbose_var = tk.IntVar(value=2)

# model parameters
initial_conditions_var = tk.DoubleVar(value=0.01)
tend_var = tk.IntVar(value=24)
nCol_var = tk.IntVar(value=200)
nTest_var = tk.IntVar(value=500)

# arguments for training data generator
sparse_var = tk.BooleanVar(value=True)
time_limit_var = tk.StringVar(value='0 24')
noise_level_var = tk.DoubleVar(value=0.005)
show_figure_var = tk.BooleanVar(value=True)

shuffle_var = tk.BooleanVar(value=True)
stopafter_var = tk.IntVar(value=500)
savefreq_var = tk.IntVar(value=100000)
dtype_var = tk.StringVar(value='float64')
gpu_var = tk.BooleanVar(value=False)
outputpath_var = tk.StringVar(value='SGModels')
outputprefix_var = tk.StringVar(value='res')

plot_var = tk.BooleanVar(value=False)

# Create entry fields and label for each parameter#############################################################
# First column
ttk.Label(root, text='independent_networks').grid(row=0, column=0)
ttk.Entry(root, textvariable=independent_networks_var).grid(row=0, column=1)

ttk.Label(root, text='verbose').grid(row=1, column=0)
ttk.Entry(root, textvariable=verbose_var).grid(row=1, column=1)

ttk.Label(root, text='layers').grid(row=2, column=0)
ttk.Entry(root, textvariable=layers_var).grid(row=2, column=1)

ttk.Label(root, text='actf').grid(row=3, column=0)
ttk.Entry(root, textvariable=actf_var).grid(row=3, column=1)

ttk.Label(root, text='numx').grid(row=4, column=0)
ttk.Entry(root, textvariable=numx_var).grid(row=4, column=1)

ttk.Label(root, text='batchsize').grid(row=5, column=0)
ttk.Entry(root, textvariable=batchsize_var).grid(row=5, column=1)

ttk.Label(root, text='epochs').grid(row=6, column=0)
ttk.Entry(root, textvariable=epochs_var).grid(row=6, column=1)

ttk.Label(root, text='learningrate').grid(row=7, column=0)
ttk.Entry(root, textvariable=learningrate_var).grid(row=7, column=1)

ttk.Label(root, text='initial_conditions').grid(row=8, column=0)
ttk.Entry(root, textvariable=initial_conditions_var).grid(row=8, column=1)

# Second column 
# 8 rows in the second column as well

ttk.Label(root, text='tend').grid(row=0, column=2)
ttk.Entry(root, textvariable=tend_var).grid(row=0, column=3)

ttk.Label(root, text='nCol').grid(row=1, column=2)
ttk.Entry(root, textvariable=nCol_var).grid(row=1, column=3)

ttk.Label(root, text='nTest').grid(row=2, column=2)
ttk.Entry(root, textvariable=nTest_var).grid(row=2, column=3)

ttk.Label(root, text='sparse').grid(row=3, column=2)
ttk.Entry(root, textvariable=sparse_var).grid(row=3, column=3)

ttk.Label(root, text='time_limit').grid(row=4, column=2)
ttk.Entry(root, textvariable=time_limit_var).grid(row=4, column=3)

ttk.Label(root, text='noise_level').grid(row=5, column=2)
ttk.Entry(root, textvariable=noise_level_var).grid(row=5, column=3)

ttk.Label(root, text='shuffle').grid(row=6, column=2)
ttk.Entry(root, textvariable=shuffle_var).grid(row=6, column=3)

ttk.Label(root, text='show_figure').grid(row=7, column=2)
ttk.Entry(root, textvariable=show_figure_var).grid(row=7, column=3)

# Third column
# 8 rows in the second column as well. in this case 7 rows as we have 7 parameter left
ttk.Label(root, text='shuffle').grid(row=0, column=4)
ttk.Entry(root, textvariable=shuffle_var).grid(row=0, column=5)

ttk.Label(root, text='stopafter').grid(row=1, column=4)
ttk.Entry(root, textvariable=stopafter_var).grid(row=1, column=5)

ttk.Label(root, text='savefreq').grid(row=2, column=4)
ttk.Entry(root, textvariable=savefreq_var).grid(row=2, column=5)

ttk.Label(root, text='dtype').grid(row=3, column=4)
ttk.Entry(root, textvariable=dtype_var).grid(row=3, column=5)

ttk.Label(root, text='gpu').grid(row=4, column=4)
ttk.Entry(root, textvariable=gpu_var).grid(row=4, column=5)

ttk.Label(root, text='outputpath').grid(row=5, column=4)
ttk.Entry(root, textvariable=outputpath_var).grid(row=5, column=5)

ttk.Label(root, text='outputprefix').grid(row=6, column=4)
ttk.Entry(root, textvariable=outputprefix_var).grid(row=6, column=5)

ttk.Label(root, text='plot').grid(row=7, column=4)
ttk.Entry(root, textvariable=plot_var).grid(row=7, column=5)

# Create a button to run the model
ttk.Button(root, text='Run Model', command=run_sciann_gui).grid()

root.mainloop()
