# Installation Instructions

This document provides the steps for creating a new Python environment and installing the required packages for your project using Conda and pip. This repo is build on top of the original [FBPINNs](https://github.com/benmoseley/FBPINNs) package. To run our code to recreate the results follow this installation instruction. 

## Creating a New Conda Environment

1. **Open Terminal**: Start by opening your terminal (Command Prompt or PowerShell).

2. **Create Environment**: Run the following command to create a new Conda environment named `fbpinns` with Python version 3.12.1.

    ```bash
    conda create -n fbpinns python=3.12.1
    ```

3. **Activate Environment**: Activate the environment:

    ```bash
    conda activate fbpinns
    ```

## Installing Dependencies

After activating the `fbpinns` environment, install the project's dependencies listed in the `requirements.txt` file using pip.

1. **Ensure pip is Available**: First, ensure that pip is installed in your new environment. If it is not installed, you can install it using Conda:

    ```bash
    conda install pip
    ```

2. **Install Requirements**: With pip installed and your `fbpinns` environment activated, navigate to the directory containing your `requirements.txt` file and run the following command to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Final Steps

After completing the installation of dependencies, your `fbpinns` environment is ready to use.
