# Installation Instructions

This document provides the steps for creating a new Python environment and installing the required packages for your project using Conda and pip.

## Creating a New Conda Environment

1. **Open Terminal**: Start by opening your terminal (Command Prompt or PowerShell).

2. **Create Environment**: Run the following command to create a new Conda environment named `sciann` with Python version 3.8.19.

    ```bash
    conda create -n sciann python=3.8.19
    ```

3. **Activate Environment**: Activate the environment:

    ```bash
    conda activate sciann
    ```

## Installing Dependencies

After activating the `sciann` environment, install the project's dependencies listed in the `requirements.txt` file using pip.

1. **Ensure pip is Available**: First, ensure that pip is installed in your new environment. If it is not installed, you can install it using Conda:

    ```bash
    conda install pip
    ```

2. **Install Requirements**: With pip installed and your `sciann` environment activated, navigate to the directory containing your `requirements.txt` file and run the following command to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
3. **Install Graphviz**: Graphviz is required for visualizing the structure of the model. For installation instructions, please visit the [Graphviz download page](https://graphviz.gitlab.io/download/).

## Final Steps

After completing the installation of dependencies, your `sciann` environment is ready to use.
