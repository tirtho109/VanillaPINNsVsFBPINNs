import sys
import os
import time

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import argparse
from fbpinns.domains import RectangularDomainND
from FBPINNsModel.problems import CompetitionModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.analysis import load_model
import matplotlib.pyplot as plt
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters, export_energy_plot, plot_loss_landscape3D, plot_loss_landscape_contour
from FBPINNsModel.subdomain_helper import get_subdomain_xsws
from FBPINNsModel.FBPINNs_loss_landscape import _loss_lanscape

def get_parser():
    # Input interface for python. 
    parser = argparse.ArgumentParser(description='''
            FBPINNs code for Separating longtime behavior and learning of mechanics  \n
            Saturated Growth Model'''
    )

    parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0,v0) for the model (default [2,1])', type=float, nargs=2, default=[2,1])
    parser.add_argument('--model_type', help='Survival or co-existence model (default survival**)', type=str, nargs=1, default=['survival'])
    parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
    parser.add_argument('--tbegin', help='Begin time for the model simulation (default 0)', type=int, default=0)
    parser.add_argument('-nx','--numx', help='Number of data points (default 100)', type=int, default=100)
    parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [0, 24])', type=int, nargs=2, default=[0, 24])

    parser.add_argument('-nsub','--num_subdomain', help='Number of sub-domains (default 2)', type=int, default=2)
    parser.add_argument('-wo','--window_overlap', help='Window overlap for each subdomain (default 1.9)', type=float, default=1.9)
    parser.add_argument('-wi','--window_noDataRegion', help='Window overlap for no data region (default 1.0005)', type=float, default=1.0005)
    parser.add_argument('--unnorm_mean', help='Mean for unnormalization (default 0.)', type=float, default=0.)
    parser.add_argument('--unnorm_sd', help='Standard deviation for unnormalization (default 1.)', type=float, default=1.)

    parser.add_argument('-l', '--layers', help='Num layers and neurons (default 2 layers [1, 5, 5, 5, 2])', type=int, nargs='+', default=[1, 5, 5, 5, 2])
    parser.add_argument('-pl', '--pinns_layers', help='Num of pinns layers and neurons (default 3 layers [1, 5, 5, 5, 2])', type=int, nargs='+', default=[1, 5, 5, 5, 2])
    parser.add_argument('-lp','--lambda_phy', help='Weight for physics loss (default 1e0)', type=float, default=1e0)
    parser.add_argument('-ld','--lambda_data', help='Weight for data loss (default 1e0)', type=float, default=1e0)
    parser.add_argument('--lambda_param', help='Weight for parameter loss (default 1e6)', type=float, default=1e6)

    parser.add_argument('-nc','--num_collocation', help='Number of collocation points (default 200)', type=int, default=200)
    parser.add_argument('--sampler', help='Collocation sampler, one of ["grid", "uniform", "sobol", "halton"] (default: "grid")', type=str, nargs=1, default=['grid'])
    parser.add_argument('-nt','--num_test', help='Number of test points (default 500)', type=int, default=500)
    parser.add_argument('-e','--epochs', help='Number of epochs (default 50000)', type=int, default=50000)

    parser.add_argument('--rootdir', type=str, default='CompModels', help='Root directory for saving models and summaries (default: "CompModels")')
    parser.add_argument('--tag', type=str, default='ablation', help='Tag for identifying the run (default: "ablation")')
    parser.add_argument('--sparse', help='Sparsity of training data (default False)', type=bool, nargs=1, default=[False])
    parser.add_argument('--loss_landscape', help='Whether to plot loss_landscape for FBPINNs (default False)', type=bool, nargs=1, default=[False])
    parser.add_argument('-nl','--noise_level', help='Noise level in training data (default 0.05)', type=float, default=0.05)   
    parser.add_argument('-pt','--pinn_trainer', help='Whether to train PINN trainer (default Faöse)', type=bool, nargs=1, default=[False])

    return parser

def train_comp_model():
    parser = get_parser()
    args = parser.parse_args()
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        
    # step 1: Define Domain
    domain = RectangularDomainND
    domain_init_kwargs = dict(
        xmin = np.array([args.tbegin,]),
        xmax = np.array([args.tend,], dtype=float)
    )

    # step 2: Define problem ode to solve
    problem = CompetitionModel
    if args.model_type[0] == "survival":
        params = [0.5, 0.3, 0.6, 0.7, 0.3]
    elif args.model_type[0] == "coexistence":
        params = [0.5, 0.7, 0.3, 0.3, 0.6]
    else:
        raise ValueError(f"Invalid model type: '{args.model_type}'. Expected 'survival' or 'coexistence'.")
    problem_init_kwargs = dict(
        params=params, 
        u0=args.initial_conditions[0], 
        v0 = args.initial_conditions[1], 
        sd=0.1, 
        time_limit=args.time_limit, 
        numx=args.numx,
        lambda_phy=args.lambda_phy,
        lambda_data = args.lambda_data,
        sparse=args.sparse, noise_level=args.noise_level,
    )

    # step 3: Define domain decomposition used by the FBPINNs
    decomposition = RectangularDecompositionND
    subdomain_xs, subdomain_ws = get_subdomain_xsws(args.time_limit, args.tbegin, 
                                                    args.tend, args.num_subdomain, 
                                                    args.window_overlap, args.window_noDataRegion)
    decomposition_init_kwargs = dict(
        subdomain_xs = subdomain_xs,
        subdomain_ws = subdomain_ws,
        unnorm = (args.unnorm_mean, args.unnorm_sd),
    )

    # step 4: define the NN placed in each subdomain
    network = FCN
    network_init_kwargs = dict(
        layer_size = args.layers,
    )

    # step 5: Create a constants object

    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{args.rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{args.rootdir}/models/{self.run}/"
        
    tag = args.tag
    h = len(args.layers) - 2  # Number of hidden layers
    p = sum(args.layers[1:-1])  # Sum of neurons in hidden layers
    n = (args.num_collocation,) # number of training point(collocation)
    run = f"FBPINN_{tag}_CM_{args.model_type[0]}_{network.__name__}_{args.num_subdomain}-ns_{args.window_overlap}-l_{h}-l_{p}-h_{n[0]}-nC_"
    run += f"{args.epochs}-e_{args.numx}-nD_{args.time_limit}-tl_{args.num_test}-nT_{args.initial_conditions}-ic_{args.sparse[0]}-sp_{args.noise_level}-nl_"

    c = ModifiedConstants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=(n,),# use 200 collocation points for training
        n_test=(args.num_test,),# use 500 points for testing
        n_steps=args.epochs,# number of training steps
        clear_output=True,
        save_figures=True,
        show_figures=False,
        sampler=args.sampler[0],
        model_save_freq = 5000,
    )

    # Train the FBPINNs usnig FBPINNTrainer
    training_time = time.time()
    FBPINNrun = FBPINNTrainer(c)
    FBPINNrun.train()

    training_time = time.time() - training_time
    file_path = os.path.join(c.summary_out_dir, "FBPINNs_training_time.txt")
    with open(file_path, 'w') as file:
        file.write(str(training_time))

    if args.loss_landscape[0]:
        # export _loss_landscape
        _loss_lanscape(FBPINNrun, run, rootdir=args.rootdir+"/", resolution=20, norm=2, path=c.summary_out_dir)
        # plot loss_landscape
        csv_path = c.summary_out_dir + "/loss-landscape.csv"
        fig = plt.figure(figsize=(12, 4), dpi=300) 
        ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
        ax2 = fig.add_subplot(1, 2, 2)
        plot_loss_landscape3D(csv_path, ax1)
        plot_loss_landscape_contour(csv_path, ax2)
        fig.suptitle("Loss Landscape", fontsize=16, fontweight='bold')
        plt.subplots_adjust(wspace=0.5) 
        file_path = os.path.join(c.summary_out_dir, "loss_landscape_viz.png")
        plt.savefig(file_path, bbox_inches='tight') 

    # import model
    c_out, model = load_model(run, rootdir=args.rootdir+"/")

    # plots
    # 1. model comparisoin
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    plot_model_comparison(c_out, model, type="FBPINN", ax=ax)
    file_path = os.path.join(c.summary_out_dir, "model_comparison.png")
    plt.savefig(file_path)

    # Export params(true & learned)
    file_path = os.path.join(c.summary_out_dir, "parameters.csv")
    export_parameters(c, model, file_path)

    # Mse & Mae
    u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
    file_path = os.path.join(c.summary_out_dir, "metrices.csv")
    export_mse_mae(u_exact, u_test, u_learned, file_path)

    # plots
    # 2. N-l1 test loss vs training steps
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    ax.plot(i, l1n, label=f"FBPINN {h} l {p} hu {args.num_subdomain} ns")
    ax.set_yscale('log')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Normalized l1 test loss')
    ax.set_title('Loss vs Training Steps')
    ax.legend()
    file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
    plt.savefig(file_path)

    # plots
    # Energy plot
    file_path = os.path.join(c.summary_out_dir, "energy_plot.png")
    export_energy_plot(c_out, model, model_type=args.model_type[0],file_path=file_path)

    if args.pinn_trainer[0]:
        # PINN trainer
        h = len(args.pinns_layers) - 2  # Number of hidden layers
        p = sum(args.pinns_layers[1:-1])  # Sum of neurons in hidden layers
        run = f"PINN_{tag}_CM_{args.model_type[0]}_{network.__name__}_{h}-l_{p}-h_{n[0]}-nC_"
        run += f"{args.epochs}-e_{args.numx}-nD_{args.time_limit}-tl_{args.num_test}-nT_{args.initial_conditions}-ic_"
        run += f"{args.sparse[0]}-sp_{args.noise_level}-nl_"
        c["network_init_kwargs"] = dict(layer_sizes=args.pinns_layers)# use a larger neural network
        c["run"] = run

        training_time = time.time()
        PINNrun = PINNTrainer(c)
        PINNrun.train()

        training_time = time.time() - training_time
        file_path = os.path.join(c.summary_out_dir, "PINNs_training_time.txt")
        with open(file_path, 'w') as file:
            file.write(str(training_time))

        # import model
        c_out, model = load_model(run, rootdir=args.rootdir+"/")

        # plot1: Model Comparison
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        plot_model_comparison(c_out, model, type="PINN", ax=ax)
        file_path = os.path.join(c.summary_out_dir, "model_comparison.png")
        plt.savefig(file_path)

        # Export params(true & learned)
        file_path = os.path.join(c.summary_out_dir, "parameters.csv")
        export_parameters(c, model, file_path)
        
        # Mse & Mae
        u_exact, u_test, u_learned = get_us(c_out, model, type="PINN")
        file_path = os.path.join(c.summary_out_dir, "metrices.csv")
        export_mse_mae(u_exact, u_test, u_learned, file_path)

        # plots
        # 2. N-l1 test loss vs training steps
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
        ax.plot(i, l1n, label=f"PINN {h} l {p} hu")
        ax.set_yscale('log')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Normalized l1 test loss')
        ax.set_title('Loss vs Training Steps')
        ax.legend()
        file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
        plt.savefig(file_path)

        # plots
        # Energy plot
        file_path = os.path.join(c.summary_out_dir, "energy_plot.png")
        export_energy_plot(c_out, model, model_type=args.model_type[0],file_path=file_path)


if __name__=="__main__":
    train_comp_model()

        