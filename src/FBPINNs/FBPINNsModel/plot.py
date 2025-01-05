"""
Defines some helper functions for plotting
"""

import jax.numpy as jnp

from fbpinns.analysis import load_model
from fbpinns.analysis import FBPINN_solution as FBPINN_solution_
from fbpinns.analysis import PINN_solution as PINN_solution_
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FBPINNsModel.problems import SaturatedGrowthModel, CompetitionModel
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')
"""
i = epochs

"""

def load_FBPINN(tag, problem, network, l, w, h, p, n, rootdir="results/"):
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_PINN(tag, problem, network, h, p, n, rootdir="results/"):
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def exact_solution(c, model):
    all_params, domain, problem = model[1], c.domain, c.problem
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_exact = problem.exact_solution(all_params, x_batch, batch_shape=c.n_test)
    return u_exact.reshape(c.n_test)

def FBPINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    active = jnp.ones((all_params["static"]["decomposition"]["m"]))
    u_test = FBPINN_solution_(c, all_params, active, x_batch)
    return u_test.reshape(c.n_test)

def PINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_test = PINN_solution_(c, all_params, x_batch)
    return u_test.reshape(c.n_test)

def plot_model_comparison(c, model, type=None, ax=None):
    if type not in ["FBPINN", "PINN"]:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    u_exact, u_test, u_learned = get_us(c, model, type=type)
    x_batch = get_x_batch(c, model)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_batch, u_exact[:, 0], 'r-', label='u true')
    ax.plot(x_batch, u_test[:, 0], 'r:', label='u pred')
    ax.plot(x_batch, u_learned[:, 0], 'r-.', label='u learned')
    if u_exact.shape[1]==2:
        ax.plot(x_batch, u_exact[:, 1], 'b-', label='v true')
        ax.plot(x_batch, u_test[:, 1], 'b:', label='v pred')
        ax.plot(x_batch, u_learned[:, 1], 'b-.', label='v learned')
    ax.legend()
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.set_title(f"{type} Models Comparison")
    if ax is None:
        plt.show()

def get_x_batch(c, model):
    """
    Extract x_batch
    """
    all_params, domain= model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    return x_batch

def get_us(c, model, type=None):
    """
    Extract u_exact, u_test, u_learned
    using model and constants
    """
    if type not in ["FBPINN", "PINN"]:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    all_params, domain, problem, active = model[1], c.domain, c.problem, model[3]
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_exact = problem.exact_solution(all_params, x_batch, batch_shape=c.n_test)
    if type=="FBPINN":
        u_test = FBPINN_solution_(c, all_params, active, x_batch)
    elif type=="PINN":
        u_test = PINN_solution_(c, all_params, x_batch)
    else:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    u_learned = problem.learned_solution(all_params, x_batch.reshape(-1))
    return u_exact, u_test, u_learned

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

def export_parameters(c, model, file_path=None):
    """
    This function takes the uploaded constants and jax model.And, 
    export the learned and true params in a csv file.
    """
    all_params = model[1]  
    if c.problem==CompetitionModel:
        true_keys = ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')
        learned_keys = ('r', 'a1', 'a2', 'b1', 'b2')
        true_params = [float(all_params['static']["problem"][key]) for key in true_keys]
        learned_params = [round(float(all_params['trainable']["problem"][key]), 4)  for key in learned_keys]
        data = {
            'Parameter': ['r', 'a1', 'a2', 'b1', 'b2'],
            'True': true_params,
            'Learned': learned_params
        }
    elif c.problem==SaturatedGrowthModel:
        true_params = [float(all_params['static']['problem']['C_true'])]
        learned_params = [round(float(all_params['trainable']['problem']['C']), 4)]
        data = {
            'Parameter': ['C'],
            'True': true_params,
            'Learned': learned_params
        }
    else:
        raise ValueError("Unsupported problem type.")

    # Create DataFrame and save to CSV
    parameters_df = pd.DataFrame(data)
    if file_path is not None:
        parameters_df.to_csv(file_path, index=False)
    else:
        parameters_df.to_csv('parameters.csv', index=False)
       
# Energy plots
def plot_energy(phi_values, u, v, ax=None, set_title=None):
    if ax is None:
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
    
    contour = ax.contourf(u, v, phi_values, levels=50, cmap='jet')
    plt.colorbar(contour, ax=ax)
    
    if set_title is None:
        ax.set_title("Energy Landscape (Lyapunov Function)")
    else:
        ax.set_title(set_title)
    
    ax.set_xlabel("$u$ (Immune Cells)")
    ax.set_ylabel("$v$ (Virus)")

    if ax is None:
        plt.show()

# Calculate Lyapunov Function
"""
    Choosen:
    dphi/du = -b1*b2*a1*r*(1-a1*u-a2*v)
    dphi/dv = -a1*a2*b2*r*(1-b1*u-b2*v)

    phi_dot = -b1*b2*a1*r*u(1-a1*u-a2*v)**2 - a1*a2*b2*r**2*v*(1-b1u-b2v)**2 <= 0
    phi = (-a1*b2*r*(b1*u+a2*v) + a1*a2*b1*b2*r*u*v + 0.5*r*a1*b2*(a1*b1*u*u + a2*b2*v*v))

"""
def phi_comp(u, v, params):
     r, a1, a2, b1, b2 = params
     #TODO Check later (sign)
    #  return (- b1 * b2 * u + 0.5 * b1 * b2 * a1 * u**2 + b1 * b2 * a2 * v * u -
    #         a1 * a2 * r * v + a1 * a2 * r * b1 * u * v + 0.5 * a1 * a2 * r * b2 * v**2)
     return (-a1*b2*r*(b1*u+a2*v) + a1*a2*b1*b2*r*u*v + 0.5*r*a1*b2*(a1*b1*u*u + a2*b2*v*v))

def export_energy_plot(c, model, model_type="survival", file_path=None):
    if c.problem.__name__ != 'CompetitionModel':
        raise ValueError("Unsupported problem type.")
    all_params = model[1]
    true_keys = ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')
    learned_keys = ('r', 'a1', 'a2', 'b1', 'b2')
    true_params = [float(all_params['static']["problem"][key]) for key in true_keys]
    learned_params = [float(all_params['trainable']["problem"][key]) for key in learned_keys]

    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) 
    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1]) 
    # make grid
    if model_type not in ["survival", "coexistence"]:
        raise ValueError("Unsupported model type")
    
    if model_type=="survival":
        u_range = np.linspace(-.5, 5, 500)
        v_range = np.linspace(0, 5, 500)
        u, v = np.meshgrid(u_range, v_range)
    else:
        u_range = np.linspace(-.5, 2.5, 500)
        v_range = np.linspace(0, 3, 500)
        u, v = np.meshgrid(u_range, v_range)

    phi_comp_values_learned = phi_comp(u, v, learned_params)
    phi_comp_value_true = phi_comp(u, v, true_params)

    plot_energy(phi_values=phi_comp_values_learned, u=u, v=v, ax=ax1, set_title="Learned")
    plot_energy(phi_values=phi_comp_value_true, u=u, v=v, ax=ax2, set_title="True")
    param_learned_formatted = ['{:.2f}'.format(param) for param in learned_params]

    # table
    columns = ['r', 'a1', 'a2', 'b1', 'b2']
    rows = ['Learned Params', 'True Params']
    cell_text = [param_learned_formatted, true_params]

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    fig.suptitle(f"Energy Plot({model_type} model)")
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.savefig("energy_plot.png")



################## Plot functions for loss landscape ###############
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
    fig.colorbar(contour, ax=ax, shrink=0.8, aspect=14)
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

def plot_loss_landscape_contour_with_limit(file_path, ax=None, title=None, limit=None):
    if ax is None:
        fig, ax = plt.subplots()
        own_figure = True
    else:
        fig = ax.figure 
        own_figure = False

    df = pd.read_csv(file_path, header=None)

    if limit is not None:
        low_limit, high_limit = -limit, limit
        df = df[(df[0] > low_limit) & (df[0] < high_limit) & (df[1] > low_limit) & (df[1] < high_limit)]
        
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. Please use a larger limit.")
    except ValueError as e:
        print(e)
        return

    x, y, z = df[0].values, df[1].values, df[2].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x,y), z, (X,Y), method='cubic')

    contour = ax.contour(X, Y, Z, levels=np.linspace(z.min(), z.max(), 100), cmap=plt.cm.viridis)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8, aspect=14)
    # cbar.ax.tick_params(labelsize=16)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # Set to 5 ticks on the y-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    cbar_ticks = np.linspace(z.min(), z.max(), 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.3f}" for tick in cbar_ticks])
    cbar.ax.tick_params(labelsize=16)

    if title is not None:
        ax.set_title(title, fontsize=24)
    else:
        ax.set_title('Contour Plot', fontsize=24)
    # ax.set_xlabel('Direction 1', fontsize=20)  
    # ax.set_ylabel('Direction 2', fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=16)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=16)

    plt.tight_layout()
    if own_figure:
        plt.show()
    return ax