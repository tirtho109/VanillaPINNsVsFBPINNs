import numpy as np
import matplotlib.pyplot as plt

def energy_plotter(phi_values, x1, x2, title=None, axis=None):
    if axis is None:
        plt.figure(figsize=(4, 3))
        axis = plt.gca()  # Get the current axis
    
    contour = axis.contourf(x1, x2, phi_values, levels=50, cmap='jet')
    plt.colorbar(contour, ax=axis)
    if title is None:
        axis.set_title("Energy Landscape (Lyapunov Function)")
    else:
        axis.set_title(title)
    axis.set_xlabel("u")
    axis.set_ylabel("v")

    if axis is None:
        plt.show()

def calculate_energy(u, v, params):
    r, a1, a2, b1, b2 = params
    return (-a1*b2*r*(b1*u+a2*v) + a1*a2*b1*b2*r*u*v + 0.5*r*a1*b2*(a1*b1*u*u + a2*b2*v*v))

def plot_energy_from_params(params, model_type="coexistence", axis=None, title=None):
    if axis is None:
        plt.figure(figsize=(4, 3))
        axis = plt.gca()  # Get the current axis
    if model_type=="coexistence":
        u_range = np.linspace(-0.5, 2.5, 500)
        v_range = np.linspace(0, 3, 500)
        u, v = np.meshgrid(u_range, v_range)
    elif model_type=="survival":
        u_range = np.linspace(-0.5, 5, 500)
        v_range = np.linspace(0, 5, 500)
        u, v = np.meshgrid(u_range, v_range)
    else:  
        raise ValueError("Unsupported model type.")
    
    phi_comp_values = calculate_energy(u, v, params)
    energy_plotter(phi_comp_values, u, v, title=title, axis=axis)
    




