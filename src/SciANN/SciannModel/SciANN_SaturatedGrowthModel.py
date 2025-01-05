from keras import backend as K
import sciann as sn
from sciann.utils.math import diff, tanh
from sciann import SciModel

import os, time
import shutil
import numpy as np

import matplotlib.pyplot as plt
import argparse
from model_ode_solver import *

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

# weights for loss function
parser.add_argument('--lambda_ode', help='loss weight for the ode (default 1)', type=float, default=1e0)
parser.add_argument('--lambda_data', help='loss weight for the data (default 1)', type=float, default=1e0)

# model parameters
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions for the model (default 0.01)', type=float, default=0.01)
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('--seed', help='add seed for reproducibility (default 0)', type=int, default=0)
parser.add_argument('--nCol', help='Number of collocation points(default 200)', type=int, default=200)
parser.add_argument('--nTest', help='Number of collocation points(default 500)', type=int, default=500)

# arguments for training data generator
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [0, 24])', type=int, nargs=2, default=[0, 24])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.05)', type=float, default=0.05)
parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])

parser.add_argument('--shuffle', help='Shuffle data for training (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('--savefreq', help='Frequency to save weights (each n-epoch)', type=int, nargs=1, default=[100000])
parser.add_argument('--dtype', help='Data type for weights and biases (default float64)', type=str, nargs=1, default=['float64'])
parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['SGModels'])
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
        "SciANN_SG_"
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

#TODO
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
    f"Model Type: Saturated Growth\n" 
    f"ODE Loss Weight: {args.lambda_ode}\n"
    f"Data Loss Weight: {args.lambda_data}\n"
)

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def train_sg_model():
    # sn.reset_session()
    sn.set_random_seed(args.seed)
    # NN Setup
    # using Hard constraints
    t = sn.Variable("t", dtype='float64')
    C = sn.Parameter(0.5, inputs=t, name="C")
    v = sn.Functional("v", [t], args.layers, args.actf[0])
    u = 0.01 + tanh(t/0.1)*v 
    u_t = diff(u,t)
    c1 = sn.Tie(u_t, u*(C-u))
    d1 = sn.Data(u)

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
        targets=[d1, c1],
        loss_func="mse",
        plot_to_file=output_file_name+"__model.png",
    )
    model.set_loss_weights([args.lambda_data, args.lambda_ode])
    
    with open("{}_summary".format(output_file_name), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))

    sg_model = SaturatedGrowthModel(C=1.0, initial_conditions=args.initial_conditions, tend=args.tend)
    np.savetxt(output_file_name+"_C_true",[1.0], delimiter=', ')
    np.random.seed(args.seed)
    plot_save_file = f"{args.outputprefix[0]}__training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    # prepare training data
    data_time, data_constraints = sg_model.generate_training_dataset(
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
    
    data_constraints_full = np.zeros_like(t_total)
    data_constraints_full[data_index] = data_constraints.reshape(-1)
    data_d1 = data_constraints_full
    data_c1 = 'zeros'
    target_data = [(data_index, data_d1), 
                   data_c1]

    training_time = time.time()
    save_weights_config = {
        'path': output_file_name,  
        #'freq': 500,  
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

    C_pred = C.eval(model, tspan)      # Learned param
    print(C_pred)
    sg_learned_model = SaturatedGrowthModel(C=C_pred, initial_conditions=args.initial_conditions, tend=args.tend)
    _, u_learned = sg_learned_model.solve_ode(initial_conditions=args.initial_conditions, t_span=tspan)

    u_pred = u.eval(model, tspan)   	# NN pred

    np.savetxt(output_file_name+"_t", tspan, delimiter=', ')
    np.savetxt(output_file_name+"_C_learned", C_pred, delimiter=', ')

    combined_learned_data = np.column_stack((tspan, u_learned))
    np.savetxt(output_file_name+"_t_u_learned.csv", combined_learned_data, delimiter=', ', header='t, u_learned', comments='')

    combined_data = np.column_stack((tspan, u_pred))
    np.savetxt(output_file_name+"_t_u_pred.csv", combined_data, delimiter=', ', header='t, u_pred', comments='')

def plot():
    # Loss plots
    total_loss = np.loadtxt(output_file_name+"_loss")
    u_loss = np.loadtxt(output_file_name+"_add_loss")
    ode_loss = np.loadtxt(output_file_name+"_sub_2_loss")
    time = np.loadtxt(output_file_name+"_time")

    fig, ax = plt.subplots(1, 3, figsize=(9, 4), dpi=300)
    
    cust_semilogx(ax[0], None, total_loss/total_loss[0], "epochs", "L/L0", label="Total_Loss")
    cust_semilogx(ax[0], None, u_loss/u_loss[0],xlabel=None, ylabel=None, label="u_Loss")
    cust_semilogx(ax[0], None, ode_loss/ode_loss[0],xlabel=None, ylabel=None, label="ODE_Loss")
    ax[0].legend(fontsize='small')

    cust_semilogx(ax[1], None, total_loss,  "epochs", "L", label="Total_Loss")
    cust_semilogx(ax[1], None, u_loss, xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[1], None, ode_loss, xlabel=None, ylabel=None, label="ODE_Loss")
    ax[1].legend(fontsize='small')

    ax[2].axis('off')  # Turn off axis
    ax[2].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')

    fig.suptitle("Loss")
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(f"{output_file_name}__loss.png")

    # Model plot
    C_true = np.loadtxt(output_file_name+"_C_true")
    C_learned = np.loadtxt(output_file_name+"_C_learned")
    true_sg_model = SaturatedGrowthModel(C_true, args.initial_conditions, args.tend)
    learned_sg_model= SaturatedGrowthModel(C_learned, args.initial_conditions, args.tend)
    
    # calculate mse and save it
    t_span = np.linspace(0,args.tend, args.nTest)
    _, true_solution = true_sg_model.solve_ode(args.initial_conditions, t_span=t_span)
    _, learned_solution = learned_sg_model.solve_ode(args.initial_conditions, t_span=t_span)
    nn_pred_solution = np.loadtxt(output_file_name+"_t_u_pred.csv", delimiter=',', skiprows=1) #model pred
    file_path = os.path.join(output_folder, "metrices.csv")
    assert true_solution.shape == learned_solution.shape
    assert true_solution.shape == nn_pred_solution[:, 1:2].shape
    export_mse_mae(true_solution, nn_pred_solution[:,1:2], learned_solution, file_path=file_path)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=300)
    model_comparison(true_sg_model, learned_sg_model, nn_pred_solution, t_span, ax[0])

    ax[1].axis('off')  # Turn off axis
    ax[1].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"{output_file_name}__model_comparison.png")

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
         train_sg_model()
         plot()
    else:
         plot()