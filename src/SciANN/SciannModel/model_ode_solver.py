import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, initial_conditions, tend):
        if tend <= 0:
            raise ValueError("End time 'tend' must be greater than zero.")

        self.initial_conditions = initial_conditions
        self.tend = tend

    @abstractmethod
    def _model(self, state, t):
        """
        Defines the DE model.
        This abstract method must be implemented by child classes.
        """
        pass

    def solve_ode(self, initial_conditions, t_span):
        """
        Solves the ODE using scipy's ````odeint````.
        """
        solution = odeint(self._model, initial_conditions, t_span)
        return t_span, solution
    
    @abstractmethod
    def plot_solution(self, ax=None, set_title=None):
        """
        Plots the solution of the ODE.
        This abstract method must be implemented by child classes.
        """
        pass
    
    def generate_training_dataset(self, numpoints:int=500, 
                      sparse:bool=False, time_limit:list=None, 
                      noise_level:float=0.0, show_figure:bool=False,
                      save_path:str=None, seed:int=None)-> Tuple[np.ndarray, np.ndarray]:
        """
            Generates training data by solving an ODE, with options for sparsity, time limits, noise addition, and visualization.

            Parameters:
            - numpoints (int, optional): Number of data points (default 500).
            - sparse (bool, optional): Toggle for sparse data generation (default False).
            - time_limit (list/tuple, optional): Time window [start, end] for data generation (default None).
            - noise_level (float, optional): Level of Gaussian noise to add (default 0.0).
            - show_figure (bool, optional): Show scatter plot of data if True (default False).

            Returns:
            - tTrain (numpy.ndarray): Time points of the training data.
            - sol_noisy (numpy.ndarray): ODE solution at 'tTrain' with optional noise.

            Raises:
            - ValueError: For invalid 'tend', 'numpoints', or 'time_limit' values.
        """
         # check conditions
        assert numpoints > 1, "Number of points must be greater than one."
        if time_limit is not None:
            if not isinstance(time_limit, (list, tuple)) or len(time_limit) != 2:
                raise ValueError("Time limit 'time_limit' must be a list or tuple with two elements.")
            if not 0 <= time_limit[0] < time_limit[1]:
                raise ValueError("Time limit 'time_limit' must contain increasing values greater than zero.")
        else:
            time_limit = [0, self.tend]
        # set seed to 0 if seed is None else set np random seed
        # to reproduce the same results
        if seed is None:
            seed = 0
        np.random.seed(seed)

        if sparse:
            tTrain = np.sort(np.random.uniform(time_limit[0], time_limit[1], numpoints))
        else:
            tTrain = np.linspace(time_limit[0], time_limit[1], numpoints)
        
        t_span = np.arange(0, self.tend, 0.0002)
        solution = odeint(self._model, self.initial_conditions, t_span)

        sol = np.zeros((tTrain.shape[0], solution.shape[1]))

        if solution.shape[1] == 2:
            # Interpolation
            u_interp = interp1d(t_span, solution[:, 0], kind='cubic')
            v_interp = interp1d(t_span, solution[:, 1], kind='cubic')
            sol[:, 0] = u_interp(tTrain)
            sol[:, 1] = v_interp(tTrain)
        elif solution.shape[1] == 1:
            u_interp = interp1d(t_span, solution[:,0], kind='cubic')
            sol[:, 0] = u_interp(tTrain)
        print("Sol shape after: ", sol.shape)
        noise = np.random.normal(0, noise_level, sol.shape)
        sol_noisy = sol + noise
        if show_figure:
            for i in range(sol_noisy.shape[1]):
                plt.scatter(tTrain, sol_noisy[:,i], label=f'Cell Population {i+1}')
                plt.legend()
            title = f'Training Points - Noise Level: {noise_level}'
            if time_limit:
                title += f', Window: [{time_limit[0]}, {time_limit[1]}]'
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('Population')
            plt.ylim(-0.2, np.max(sol_noisy+0.2))
            plt.xlim(-0.5, self.tend+1)
            if save_path:
                if os.path.basename(save_path) == '':
                    raise ValueError("Please provide a file name with the save path.")
                plt.savefig(save_path)
            else:
                plt.show()
        return tTrain, sol_noisy


class SaturatedGrowthModel(BaseModel):
    def __init__(self, C, initial_conditions, tend):
        super().__init__(initial_conditions, tend)
        self.C = C

    def _model(self, u, t):
        return u*(self.C -u)
    
    def plot_solution(self, ax=None, set_title=None):
        t_span = np.arange(0, self.tend, 0.1)
        _, solution = self.solve_ode(initial_conditions=self.initial_conditions, t_span=t_span)
        u = solution[:, 0]

        if ax is None:
            fig, ax = plt.subplots()
            ax.plot(t_span, u)
            if set_title is None:
                ax.set_title("SG Model")
            else:
                ax.set_title(set_title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            plt.show() 
        else:
            ax.plot(t_span, u)
            if set_title is not None:
                ax.set_title(set_title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")

class CompetitionModel(BaseModel):
    def __init__(self, params, initial_conditions, tend):
        super().__init__(initial_conditions, tend)
        self.params = params

    def _model(self, q, t):
        u, v = q
        # r, a1, a2, b1,b2, e0, f0, e1, f1, e2, f2, e3, f3, e4,f4 = self.params
        r, a1, a2, b1,b2 = self.params
        dudt = u*(1-a1*u- a2*v)  # +e0 + e1*u + e2*v + e3*u*u*v + e4*u*v*v
        dvdt = r*v*(1-b1*u-b2*v) # +f0 + f1*u + f2*v + f3*u*u*v + f4*u*v*v
        return [dudt, dvdt]
    
    def plot_solution(self, ax=None, set_title=None):
        t_span = np.arange(0, self.tend, 0.1)
        _, solution = self.solve_ode(initial_conditions=self.initial_conditions, t_span=t_span)
        u, v = solution[:, 0], solution[:, 1]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        ax.plot(t_span, u, label="Population 1")
        ax.plot(t_span, v, label="Population 2")
        if set_title is None:
            ax.set_title("Comp Model")
        else:
            ax.set_title(set_title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()

        if created_fig:
            return fig

def model_comparison(true_model, predicted_model, nn_model, t_span=None, ax=None, title=None):
    """
        Compare two model (SG Model/Comp Model) along with the NN model output. 
    """
    if t_span is None:
        t_span = np.arange(0, true_model.tend, 0.01)
    _, true_sol = true_model.solve_ode(true_model.initial_conditions, t_span)
    _, predicted_sol = predicted_model.solve_ode(predicted_model.initial_conditions, t_span)
    if ax is None:
        fig, ax = plt.subplots()
    
    # Check if the solution includes both u and v
    if true_sol.shape[1] == 2:
        # Plot u and v for both true and predicted solutions
        nn_u_sol = nn_model[:,1]
        nn_v_sol = nn_model[:,2]
        ax.plot(t_span, true_sol[:, 0], label='True u', linestyle='-', color='blue', linewidth=1.5)
        ax.plot(t_span, true_sol[:, 1], label='True v', linestyle='-', color='green', linewidth=1.5)
        ax.plot(t_span, predicted_sol[:, 0], label='Learned u', linestyle='--', color='blue',linewidth=1.5)
        ax.plot(t_span, predicted_sol[:, 1], label='Learned v', linestyle='--', color='green', linewidth=1.5)
        ax.plot(t_span, nn_u_sol, label="NN pred u", linestyle='dashdot', color='blue', linewidth=1.5)
        ax.plot(t_span, nn_v_sol, label="NN pred v", linestyle='dashdot', color='green', linewidth=1.5)
    else:
        # Plot only u for both true and predicted solutions
        nn_sol = nn_model[:,1]
        ax.plot(t_span, true_sol[:, 0], label='True Solution', linestyle='-', linewidth=1.5)
        ax.plot(t_span, predicted_sol[:, 0], label='Learned Solution', linestyle='--', linewidth=1.5)
        ax.plot(t_span, nn_sol, label="NN Solution", linestyle='dashdot', linewidth=1.5)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Model Comparison')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend(fontsize='small')


def cust_semilogx(AX, X, Y, xlabel, ylabel, label):
    if X is None:
        im = AX.semilogy(Y, label=label)
    else:
        im = AX.semilogy(X, Y, label=label)
    if xlabel is not None: AX.set_xlabel(xlabel)
    if ylabel is not None: AX.set_ylabel(ylabel)


def export_mse_mae(u_exact, u_test, u_learned, file_path=None):
    """
    Calculate MSE and MAE of the predicted model and learned model
    """
    mse_test = np.mean((u_exact - u_test)**2)
    mse_learned = np.mean((u_exact - u_learned)**2)
    mae_test = np.mean(np.abs(u_exact - u_test))
    mae_learned = np.mean(np.abs(u_exact - u_learned))
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE'],
        'Test': [mse_test, mae_test],
        'Learned': [mse_learned, mae_learned]
    })
    if file_path is not None:
        metrics_df.to_csv(file_path, index=False)
    else:
        metrics_df.to_csv('metrics.csv', index=False)

def plot_loss_landscape3D(file_path, ax=None, title=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        own_figure = True
    else:
        fig = ax.figure
        own_figure = False

    df = pd.read_csv(file_path, header=None)
    x = df[0]
    y = df[1]
    z = df[2]

    resolution = int(np.sqrt(len(z)))
    X = x.values.reshape((resolution, resolution))
    Y = y.values.reshape((resolution, resolution))
    Z = z.values.reshape((resolution, resolution))

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('3D Plot')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss', labelpad=10)
    ax.zaxis.set_rotate_label(False)  
    ax.zaxis.label.set_rotation(90)
    fig.colorbar(surf, ax=ax, shrink=1, pad=0.1, location='left')

    if own_figure:
        plt.show()
    
    return ax

def plot_loss_landscape_contour(file_path, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        own_figure = True
    else:
        fig = ax.figure 
        own_figure = False

    df = pd.read_csv(file_path, header=None)
    x, y, z = df[0].values, df[1].values, df[2].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x,y), z, (X,Y), method='cubic')

    contour = ax.contour(X, Y, Z, levels=np.linspace(z.min(), z.max(), 100), cmap=plt.cm.viridis)
    fig.colorbar(contour, ax=ax, shrink=1, aspect=14)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Contour Plot')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if own_figure:
        plt.show()
    return ax