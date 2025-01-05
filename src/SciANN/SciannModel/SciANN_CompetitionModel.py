import os, time
import shutil
import numpy as np

from keras import backend as K
import sciann as sn
from sciann.utils.math import tanh, diff
from sciann import SciModel, Functional, Parameter
from sciann import Data, Tie
from sciann import Variable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
from model_ode_solver import *

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        SciANN code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

parser.add_argument('-l', '--layers', help='Num layers and neurons (default 4 layers each 40 neurons [5, 5, 5])', type=int, nargs='+', default=[5]*3)
parser.add_argument('-af', '--actf', help='Activation function (default tanh)', type=str, nargs=1, default=['tanh'])
parser.add_argument('-nx', '--numx', help='Num Node in X (default 20)', type=int, nargs=1, default=[100])
parser.add_argument('--seed', help='add seed for reproducibility (default 0)', type=int, default=0)
parser.add_argument('--nCol', help='Number of collocation points(default 200)', type=int, default=200)
parser.add_argument('--nTest', help='Number of collocation points(default 500)', type=int, default=500)
parser.add_argument('-bs', '--batchsize', help='Batch size for Adam optimizer (default 25)', type=int, nargs=1, default=[25])
parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 2000)', type=int, nargs=1, default=[2000])
parser.add_argument('-lr', '--learningrate', help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('-in', '--independent_networks', help='Use independent networks for each var (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

# weights for loss function
parser.add_argument('--lambda_ode_u', help='loss weight for the ode of u(default 1)', type=float, default=1e0)
parser.add_argument('--lambda_data_u', help='loss weight for the data of u(default 1)', type=float, default=1e0)
parser.add_argument('--lambda_ode_v', help='loss weight for the ode of v(default 1)', type=float, default=1e0)
parser.add_argument('--lambda_data_v', help='loss weight for the data of v(default 1)', type=float, default=1e0)

# model parameters
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0,v0) for the model (default [2,1])', type=float, nargs=2, default=[2,1])
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('--sd', help='sd (default 0.1)', type=float, default=0.1)
parser.add_argument('--model_type', help='Survival or co-existence model (default survival**)', type=str, nargs=1, default=['survival'])
# parser.add_argument('--higher_order', help='Higher order model parameter (default False)', type=bool, nargs=1, default=[False])

# arguments for training data generator
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [0, 24])', type=int, nargs=2, default=[0, 24])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.05)', type=float, default=0.05)
parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])

parser.add_argument('--shuffle', help='Shuffle data for training (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('--savefreq', help='Frequency to save weights (each n-epoch)', type=int, nargs=1, default=[100000])
parser.add_argument('--dtype', help='Data type for weights and biases (default float64)', type=str, nargs=1, default=['float64'])
parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['CompModels'])
parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()

def clear_dir(directory):
    """
    Removes all files in the given directory.
    """
    # important! if None passed to os.listdir, current directory is wiped (!)
    if not os.path.isdir(directory): raise Exception(f"{directory} is not a directory")
    if type(directory) != str: raise Exception(f"string type required for directory: {directory}")
    if directory in ["..",".", "","/","./","../","*"]: raise Exception("trying to delete current directory, probably bad idea?!")

    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

if not args.gpu[0]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Manage Files
if not os.path.isdir(args.outputpath[0]):
        os.mkdir(args.outputpath[0])

folder_name = (
    f"SciANN_CM_{args.model_type[0]}_"
    f"actf_{args.actf[0]}_"
    f"l_{'x'.join(map(str, args.layers))}_"
    f"nD_{args.numx[0]}_"
    f"bs_{args.batchsize[0]}_"
    f"e_{args.epochs[0]}_"
    f"lr_{args.learningrate[0]}_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}_"
    f"nC_{args.nCol}_"
    f"nT_{args.nTest}_"
)

output_folder = os.path.join(args.outputpath[0], folder_name)

if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

if args.plot==False:    
    clear_dir(output_folder)

output_file_name = os.path.join(output_folder, args.outputprefix[0])
# fname = (
#     f"{output_file_name}_"
#     f"actf_{args.actf[0]}_"
#     f"layers_{'x'.join(map(str, args.layers))}_"
#     f"numx_{args.numx[0]}_"
#     f"bs_{args.batchsize[0]}_"
#     f"epochs_{args.epochs[0]}_"
#     f"lr_{args.learningrate[0]}_"
#     f"tl_{'-'.join(map(str, args.time_limit))}_"
#     f"nl_{args.noise_level}"
# )

#TODO
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

#TODO 
# Phase Plot
def phase_plot(model, ax=None, title=None):
    t = np.linspace(0, model.tend, 500)
    if ax is None:
        fig, ax = plt.subplots()
    Y, X = np.mgrid[0:5:15j, 0:5:15j]
    U, V = model._model([X, Y], t)
    speed = np.sqrt((U**2 + V**2))
    UN = np.nan_to_num(U / speed)
    VN = np.nan_to_num(V / speed)

    Q = ax.quiver(X,Y, UN, VN, color='r', scale=50) # scale = 50

    # Nullclines
    r, a1, a2, b1, b2 = model.params
    v_range = np.linspace(0, 4.0, 200)
    u_range = v_range
    u_nullcline = (1 - a2 * v_range) / a1
    v_nullcline = (1 - b1 * u_range) / b2 
    ax.plot(u_nullcline, v_range, 'b--', label=r'$1 - a_1 \cdot u - a_2 \cdot v$')  # u Nullcline
    ax.plot(u_range, v_nullcline, 'r--', label=r'$1 - b_1 \cdot u - b_2 \cdot v$')  # v Nullcline

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.legend()
    if title is not None:
         ax.set_title(title)
    else:
         ax.set_title("Phase plot")
    ax.grid(True)
    if ax is None:
         plt.show()

network_description = (
     f"Layers: {'x'.join(map(str, args.layers))}\n"
     f"Activation Function: {args.actf[0]}\n"
     f"Num Node in X: {args.numx[0]}\n"
     f"Batch Size: {args.batchsize[0]}\n"
     f"Epochs: {args.epochs[0]}\n"
     f"Learning Rate: {args.learningrate[0]}\n"
     f"Time Window: {'-'.join(map(str, args.time_limit))}\n"
     f"Independent Networks: {args.independent_networks[0]}\n"
     f"Noise Level: {args.noise_level}\n"
     f"Sparse: {args.sparse[0]}\n"
     f"Model Type: {args.model_type[0]}\n"
     f"ODE_u Loss Weight: {args.lambda_ode_u}\n"
     f"Data_u Loss Weight: {args.lambda_data_u}\n"
     f"ODE_v Loss Weight: {args.lambda_ode_v}\n"
     f"Data_v Loss Weight: {args.lambda_data_v}\n"
)

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def train_comp_model():
    # sn.reset_session()
    sn.set_random_seed(args.seed)
    # NN Setup
    # using Hard constraints
    t = Variable("t", dtype=args.dtype[0])              # input

    #TODO: Need to check why independent network is not working for False, while calling from other script
    # if args.independent_networks[0]:  
    #     x = Functional("x", t, args.layers, args.actf[0])   # output
    #     y = Functional("y", t, args.layers, args.actf[0])
    # else:
    #      x, y = Functional(
    #                     ["x", "y"], t, 
    #                     args.layers, 
    #                     args.actf[0])#.split()
    x, y = Functional(["x", "y"], t, 
                        args.layers, 
                        args.actf[0])
    
    u = args.initial_conditions[0] + tanh(t/args.sd)*x
    v = args.initial_conditions[1] + tanh(t/args.sd)*y
    r = Parameter(0.5, inputs=t, name="r" )                 # Learnable parameters
    a1 = Parameter(0.5, inputs=t, name="a1")
    a2 = Parameter(0.5, inputs=t, name="a2") 
    b1 = Parameter(0.5, inputs=t, name="b1" ) 
    b2 = Parameter(0.5, inputs = t, name = "b2") 
    
    u_t = diff(u,t)
    v_t = diff(v,t)

    d1 = Data(u)
    d2 = Data(v)

    c1 = Tie(u_t, u*(1-a1*u-a2*v))
    c2 = Tie(v_t, r*v*(1-b1*u-b2*v))

    # SciModel doesn't support custom loss weights, so we need to subclass it
    # and add a method to set the loss weights after the model is created
    class CustomSciModel(SciModel):
        def set_loss_weights(self, weights):
            if len(weights) != len(self._loss_weights):
                raise ValueError("Weight array length must match the number of output variables")
            for idx, weight in enumerate(weights):
                K.set_value(self._loss_weights[idx], weight)
            self.compile() 

    model = CustomSciModel(
        inputs=[t],
        targets=[d1, d2, c1, c2],
        loss_func="mse",
        plot_to_file=output_file_name+"__model.png",
    )
    model.set_loss_weights([args.lambda_data_u, args.lambda_data_v, args.lambda_ode_u, args.lambda_ode_v])

    with open("{}_summary".format(output_file_name), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))

    # prepeare training data
    #TODO: What about e's and f's has some value
    if args.model_type[0]=="survival":
         # [r, a1, a2, b1, b2]
         comp_params = [0.5, 0.3, 0.6, 0.7, 0.3] 
    else:
         comp_params = [0.5, 0.7, 0.3, 0.3, 0.6] 
    
    comp_model = CompetitionModel(params=comp_params,
                                  initial_conditions=args.initial_conditions,
                                  tend=args.tend)
    np.savetxt(output_file_name+"_params", comp_params, delimiter=', ')
    np.random.seed(0)
    plot_save_file = f"{args.outputprefix[0]}__training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    data_time, data_constraints = comp_model.generate_training_dataset(
                                                                numpoints=args.numx[0], 
                                                                sparse=args.sparse[0], 
                                                                time_limit=args.time_limit, 
                                                                noise_level=args.noise_level, 
                                                                show_figure=args.show_figure[0],
                                                                save_path=plot_save_path if args.show_figure[0] else None,
                                                                seed=args.seed,
                                                                )
    collocation_points = np.linspace(0, args.tend, args.nCol)
    t_total = np.unique(np.sort(np.concatenate((collocation_points, data_time))))
    data_index = np.searchsorted(t_total, data_time)

    assert t_total[data_index].all() == data_time.all(), "Check index and position of data point and collocation point"
    
    # set data constraints properly
    data_constraints_full_1 = np.zeros_like(t_total)
    data_constraints_full_2 = np.zeros_like(t_total)
    data_constraints_full_1[data_index] = data_constraints[:,0]
    data_constraints_full_2[data_index] = data_constraints[:,1]
    data_d1 = data_constraints_full_1
    data_d2 = data_constraints_full_2
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    cp_index = np.searchsorted(t_total, collocation_points)
    target_data = [(data_index, data_d1),
               (data_index, data_d2),
               (cp_index, data_c1), 
               (cp_index, data_c2)]

    training_time = time.time()
    save_weights_config = {
        'path': output_file_name,  
        #'freq': 1000,  
        'best': True  
        }
    log_loss_landscape_config = {
        'norm': 2,  # L2 norm
        'resolution': 40, 
        'path': output_folder, 
        'trials': 1  
    }
    history = model.train(
        x_true=[t_total],
        y_true=target_data,
        epochs=args.epochs[0],
        batch_size=args.batchsize[0],
        shuffle=args.shuffle[0],
        learning_rate=args.learningrate[0],
        verbose=args.verbose[0],
        save_weights=save_weights_config,
        log_loss_landscape=log_loss_landscape_config,
    )
    training_time = time.time() - training_time
    np.savetxt(output_file_name+"_SciANN_training_time.txt", [training_time])

    weights_file_name = output_file_name + "_weights.hdf5"

    # Save the model weights as an .hdf5 file
    model.save_weights(weights_file_name)

    for loss in history.history:
        np.savetxt(output_file_name+"_{}".format("_".join(loss.split("/"))), 
                    np.array(history.history[loss]).reshape(-1, 1))
    
    time_steps = np.linspace(0, training_time, len(history.history["loss"]))
    np.savetxt(output_file_name+"_Time", time_steps.reshape(-1,1))

    # Post-processing
    tspan = np.linspace(0, args.tend, args.nTest)
    # Learned Params
    r_pred = r.eval(model, tspan)[0]
    a1_pred = a1.eval(model, tspan)[0]
    a2_pred = a2.eval(model, tspan)[0]
    b1_pred = b1.eval(model, tspan)[0]
    b2_pred = b2.eval(model, tspan)[0]
    
    learned_comp_params = [r_pred, a1_pred, a2_pred, b1_pred, b2_pred]
    comp_learned_model = CompetitionModel(params=learned_comp_params, initial_conditions=args.initial_conditions, tend=args.tend)
    _, sol = comp_learned_model.solve_ode(initial_conditions=args.initial_conditions, t_span=tspan)
    u_learned = sol[:, 0]           # using learned params
    v_learned = sol[:, 1]

    u_pred = u.eval(model, tspan)   # NN pred
    v_pred = v.eval(model, tspan)

    np.savetxt(output_file_name+"_t", tspan, delimiter=', ')
    np.savetxt(output_file_name+"_learned_params", learned_comp_params, delimiter=', ')

    combined_learned_data = np.column_stack((tspan, u_learned, v_learned))
    np.savetxt(output_file_name+"_t_u_v_learned.csv", combined_learned_data, delimiter=', ', header='t, u_learned, v_learned', comments='')

    combined_pred_data = np.column_stack((tspan, u_pred, v_pred))
    np.savetxt(output_file_name+"_t_u_v_pred.csv", combined_pred_data, delimiter=', ', header='t, u_pred, v_pred', comments='')

def plot():
     # Loss plots
    total_loss = np.loadtxt(output_file_name+"_loss")
    u_loss = np.loadtxt(output_file_name+"_add_loss")
    ode1_loss = np.loadtxt(output_file_name+"_sub_2_loss")
    v_loss = np.loadtxt(output_file_name+"_add_2_loss")
    ode2_loss = np.loadtxt(output_file_name+"_sub_4_loss")
    time = np.loadtxt(output_file_name+"_time")

    fig, ax = plt.subplots(1, 3, figsize=(9, 4), dpi=300)
    
    cust_semilogx(ax[0], None, total_loss/total_loss[0], "epochs", "L/L0", label="Total_Loss")
    cust_semilogx(ax[0], None, u_loss/u_loss[0],xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[0], None, ode1_loss/ode1_loss[0],xlabel=None, ylabel=None, label="ODE1_Loss")
    cust_semilogx(ax[0], None, v_loss/v_loss[0],xlabel=None, ylabel=None, label="V_Loss")
    cust_semilogx(ax[0], None, ode2_loss/ode2_loss[0],xlabel=None, ylabel=None, label="ODE2_Loss")
    ax[0].legend(fontsize='small')

    cust_semilogx(ax[1], None, total_loss,  "epochs", "L", label="Total_Loss")
    cust_semilogx(ax[1], None, u_loss, xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[1], None, ode1_loss, xlabel=None, ylabel=None, label="ODE1_Loss")
    cust_semilogx(ax[1], None, v_loss, xlabel=None, ylabel=None, label="V_Loss")
    cust_semilogx(ax[1], None, ode2_loss, xlabel=None, ylabel=None, label="ODE2_Loss")
    ax[1].legend(fontsize='small')
    

    ax[2].axis('off')  # Turn off axis
    ax[2].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')

    fig.suptitle("Loss")
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(f"{output_file_name}__loss.png")

    # Model plot
    param_true = np.loadtxt(output_file_name+"_params")
    param_learned = np.loadtxt(output_file_name+"_learned_params")
    true_comp_model = CompetitionModel(param_true, args.initial_conditions, args.tend)
    learned_comp_model= CompetitionModel(param_learned, args.initial_conditions, args.tend)

    # calculate mse and save it
    t_span = np.linspace(0, args.tend, args.nTest)

    _, true_solution = true_comp_model.solve_ode(args.initial_conditions, t_span)
    _, learned_solution = learned_comp_model.solve_ode(args.initial_conditions, t_span)
    nn_pred_solution = np.loadtxt(output_file_name+"_t_u_v_pred.csv", delimiter=', ', skiprows=1)
    file_path = os.path.join(output_folder, "metrices.csv")
    print(true_solution.shape, learned_solution.shape, nn_pred_solution[:,1:3].shape)
    assert true_solution.shape == learned_solution.shape
    assert true_solution.shape == nn_pred_solution[:, 1:3].shape
    export_mse_mae(true_solution, nn_pred_solution[:,1:3], learned_solution, file_path=file_path)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=300)
    model_comparison(true_comp_model, learned_comp_model, nn_pred_solution, t_span, ax[0])

    ax[1].axis('off')  # Turn off axis
    ax[1].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"{output_file_name}__model_comparison.png")

    # Energy plot
    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) 

    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1]) 

    if args.model_type[0]=="survival":
        u_range = np.linspace(-.5, 5, 500)
        v_range = np.linspace(0, 5, 500)
        u, v = np.meshgrid(u_range, v_range)
    else:
        u_range = np.linspace(-.5, 2.5, 500)
        v_range = np.linspace(0, 3, 500)
        u, v = np.meshgrid(u_range, v_range)

    phi_comp_values_learned = phi_comp(u, v, param_learned)
    plot_energy(phi_values=phi_comp_values_learned, u=u, v=v, ax=ax1, set_title=args.model_type[0]+" Learned")

    phi_comp_value_true = phi_comp(u, v, param_true)
    plot_energy(phi_values=phi_comp_value_true, u=u, v=v, ax=ax2, set_title=args.model_type[0]+" True")

    param_learned_formatted = ['{:.2f}'.format(param) for param in param_learned]

    # table
    columns = ['r', 'a1', 'a2', 'b1', 'b2']
    rows = ['Learned Params', 'True Params']
    cell_text = [param_learned_formatted, param_true]

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    fig.suptitle("Energy Plot")
    plt.savefig(f"{output_file_name}__Energy.png")

    # Trajectories Plot
    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) 

    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1]) 
    phase_plot(learned_comp_model, ax=ax1, title="Learned")
    phase_plot(true_comp_model, ax=ax2, title="True")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    fig.suptitle("Phase Plot")
    plt.savefig(f"{output_file_name}__Trajectories.png")

    # plot loss_landscape
    file_path = output_folder + "/loss-landscape-history-landscape.csv"
    fig = plt.figure(figsize=(12, 4), dpi=300) 
    ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
    ax2 = fig.add_subplot(1, 2, 2)
    plot_loss_landscape3D(file_path, ax1)
    plot_loss_landscape_contour(file_path, ax2)
    fig.suptitle("Loss Landscape", fontsize=16, fontweight='bold')
    plt.subplots_adjust(wspace=0.5) 
    plt.savefig(f"{output_file_name}__loss_landscape_viz.png", bbox_inches='tight') 


if __name__=="__main__":
    if args.plot==False:
         train_comp_model()
         plot()
    else:
         plot()
