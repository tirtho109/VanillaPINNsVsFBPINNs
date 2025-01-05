import sys
import os

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import seaborn as sns
import json
import argparse
from fbpinns.domains import RectangularDomainND
from FBPINNsModel.problems import CompetitionModel, SaturatedGrowthModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters, get_x_batch, plot_loss_landscape3D, plot_loss_landscape_contour
from FBPINNsModel.subdomain_helper import get_subdomain_xsws
from FBPINNsModel.FBPINNs_loss_landscape import _loss_lanscape
from utils import plot_energy_from_params

parser = argparse.ArgumentParser(description='''
        FBPINN code for Separating longtime behavior and learning of mechanics  \n
        To find the best number of subdomain for different time limit cases'''
)

parser.add_argument('--train', help='To train the whole model (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)
parser.add_argument('--loss_landscape', help='Whether to plot loss_landscape for FBPINNs (default False)', type=bool, nargs=1, default=[False])

args = parser.parse_args()

def train_FBPINN_models():
    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{rootdir}/models/{self.run}/"
        

    numx = 100
    nCol = 200
    nTest = 500
    lambda_phy = 1e0
    lambda_data = 1e0
    lambda_param = 1e3
    sparse = True
    noise_level = 0.05
    tbegin = 0
    tend = 24
    wo = 1.9
    wi = 1.005
    tag = "DDD"
    epochs = 50000
    sampler='grid'
    nsub = 2

    time_limits = [[0,10], [10,24], [0, 24]]
    # type of problem and other fix params
    layer1 = [1, 5, 5, 5, 1]
    layer2 = [1, 5, 5, 5, 2]
    layers = [layer1, layer2, layer2]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    params_type = [1,  [0.5, 0.7, 0.3, 0.3, 0.6], [0.5, 0.3, 0.6, 0.7, 0.3]] # sg, coexistence, survival
    cases = ["sg", "coexistence", "survival"]

    parent_dir = "Different_lossWeight_Param_SameWW"
    root_dirs = [f"{case}" for case in cases]
    root_dirs_with_parent = [os.path.join(parent_dir, rd) for rd in root_dirs]

    # create fig for individual model comparison
    model_fig1, model_nn_ax = plt.subplots(1, 3, figsize=(12, 4))
    model_fig2, model_learned_ax = plt.subplots(1, 3, figsize=(12, 4))
    time_labels = ["0-10", "10-24", "0-24"]
    labels = ["u", "v"]
    colors = ['black', 'red', 'blue']
    fig_tags = ['(a)', '(b)', '(c)']
    all_fig_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    # create fig for loss landscape all together
    fig_all, ax_all = plt.subplots(3, 3, figsize=(12, 9), dpi=300) # only for contour plot
    # create fig for energy plot of coexistence and survival
    fig_energy, ax_energy = plt.subplots(2, 4, figsize=(12, 6), dpi=300)

    # create a dataframe to store the metrics
    metrics_df = pd.DataFrame(columns=['tag', 'model_type', 'time_limit', 'mse_learned', 'mse_test'])

    # create a dataframe to store parameters 
    parameters_df = pd.DataFrame(columns=['tag', 'model_type', 'time_limit', 'true_params', 'learned_params'])

    for (layer, problem, params, rootdir, name, fig_tag) in zip(layers, problem_types, params_type, root_dirs_with_parent, cases, fig_tags):
        #step 1
        domain = RectangularDomainND
        domain_init_kwargs = dict(
            xmin = np.array([tbegin,]), 
            xmax = np.array([tend,])
            )
        
        # step 2
        problem_kwargs_set = []
        if problem.__name__=="CompetitionModel":
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                problem_init_kwargs = dict(
                        params=params, u0=2, v0=1, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data, lambda_param=lambda_param,
                        sparse=[sparse], noise_level=noise_level,
                    )
                problem_kwargs_set.append((problem_init_kwargs, tl))
        else:
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                problem_init_kwargs = dict(
                        C=params, u0=0.01, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
                        sparse=[sparse], noise_level=noise_level,
                    )
                problem_kwargs_set.append((problem_init_kwargs, tl))

        # step 3
        decomposition = RectangularDecompositionND
        decomposition_kwargs_set = []   
        for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                subdomain_xs, subdomain_ws = get_subdomain_xsws(tl, tbegin, tend, nsub, wo, wi)
                decomposition_init_kwargs = dict(
                    subdomain_xs=subdomain_xs,
                    subdomain_ws=subdomain_ws,
                    unnorm=(0.,1.),
                )
                decomposition_kwargs_set.append((decomposition_init_kwargs, tl))
        
        # step 4
        network = FCN# place a fully-connected network in each subdomain
        network_init_kwargs=dict(
            layer_sizes=layer,# with 2 hidden layers
        )

        h = len(layer) -2
        p = sum(layer[1:-1])
        n=(nCol,)

        runs = []
        # fig for parameters
        # fig_params, ax_params = plt.subplots(1, 3, figsize=(12, 4), dpi=300) 
        # fig for loss landscape
        fig_ll, ax_ll = plt.subplots(1, 3, figsize=(12, 4), dpi=300) # only for contour plot
        fig_ll_3D = plt.figure(figsize=(15, 5), dpi=300)
        ax_ll_3d = [fig_ll_3D.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

        for (problem_kwargs, tl) in problem_kwargs_set:
            for (decomposition_kwargs, tld) in decomposition_kwargs_set:
                if tl==tld:
                    run = f"FBPINN_{name}_{tl[0]}_{tl[1]}"
                    runs.append(run)
                    c = ModifiedConstants(
                        run=run,
                        domain=domain,
                        domain_init_kwargs=domain_init_kwargs,
                        problem=problem,
                        problem_init_kwargs=problem_kwargs,
                        decomposition=decomposition,
                        decomposition_init_kwargs=decomposition_kwargs,
                        network=network,
                        network_init_kwargs=network_init_kwargs,
                        ns=((nCol,),),# use 200 collocation points for training
                        n_test=(nTest,),# use 500 points for testing
                        n_steps=epochs,# number of training steps
                        clear_output=True,
                        sampler=sampler,
                        show_figures=False,
                        save_figures=True,
                    )
                    if args.train[0]:
                        FBPINNrun = FBPINNTrainer(c)
                        FBPINNrun.train()
                    if args.loss_landscape[0]:
                        # export loss landscape
                        _loss_lanscape(FBPINNrun, run, rootdir+"/", 20, 2, c.summary_out_dir)
                        # plot loss landscape
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
                    c_out, model = load_model(run, rootdir=rootdir+"/")

                    ########################### 0. For comparing with SciANN ############################################################################################################
                    # model comparision
                    u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
                    x_batch = get_x_batch(c_out, model)
                    print(u_exact.shape, u_test.shape, u_learned.shape)
                    tl_str = f"{tl[0]}-{tl[1]}"
                    for i in range(0, u_exact.shape[1]):
                        model_nn_ax[cases.index(name)].plot(x_batch, u_test[:,i:i+1], label=f"{labels[i]}_nn_{tl_str}", linestyle='--' if i==0 else '-.', color=colors[time_labels.index(tl_str)])
                        model_learned_ax[cases.index(name)].plot(x_batch, u_learned[:,i:i+1], label=f"{labels[i]}_l_{tl_str}", linestyle='--' if i==0 else '-.', color=colors[time_labels.index(tl_str)])

                    # loss landscape plots
                    csv_path = c.summary_out_dir + "/loss-landscape.csv"
                    # ll together plot (3x3)
                    plot_loss_landscape_contour(csv_path, ax_all[cases.index(name) ,time_labels.index(tl_str)], title=f"{all_fig_tags[cases.index(name)*3 + time_labels.index(tl_str)]} {name}[{tl_str}]")
                    # ll individual plot (1x3)
                    plot_loss_landscape_contour(csv_path, ax_ll[time_labels.index(tl_str)], title=f"{fig_tags[time_labels.index(tl_str)]}{tl_str}")
                    plot_loss_landscape3D(csv_path, ax_ll_3d[time_labels.index(tl_str)], title=f"{fig_tags[time_labels.index(tl_str)]}{tl_str}")

                    # energy plot for coexistence and survival
                    if name in ["coexistence", "survival"]:
                        all_params = model[1]
                        true_keys = ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')
                        learned_keys = ('r', 'a1', 'a2', 'b1', 'b2')
                        true_params = [float(all_params['static']["problem"][key]) for key in true_keys]
                        learned_params = [float(all_params['trainable']["problem"][key]) for key in learned_keys]
                        if name=="coexistence":
                            if tl_str=="0-10":
                                plot_energy_from_params(true_params, model_type="coexistence", axis=ax_energy[0, 0], title=f"{all_fig_tags[time_labels.index(tl_str)]}coexistence[True]")
                            plot_energy_from_params(learned_params, model_type="coexistence", axis=ax_energy[0, time_labels.index(tl_str)+1], title=f"{all_fig_tags[time_labels.index(tl_str)+1]}coexistence_[{tl_str}]")
                        else:
                            if tl_str=="0-10":
                                plot_energy_from_params(true_params, model_type="survival", axis=ax_energy[1, 0], title=f"{all_fig_tags[time_labels.index(tl_str)+4]}survival[True]")
                            plot_energy_from_params(learned_params, model_type="survival", axis=ax_energy[1, time_labels.index(tl_str)+1], title=f"{all_fig_tags[time_labels.index(tl_str)+5]}survival_[{tl_str}]")

                    ########################### 1. For total comparision ############################################################################################################
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
                    # u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
                    file_path = os.path.join(c.summary_out_dir, "metrices.csv")
                    export_mse_mae(u_exact, u_test, u_learned, file_path)

                    # plots
                    # 2. N-l1 test loss vs training steps
                    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
                    ax.plot(i, l1n, label=f"FBPINN {h} l {p} hu {nsub} ns")
                    ax.set_yscale('log')
                    ax.set_xlabel('Training Steps')
                    ax.set_ylabel('Normalized l1 test loss')
                    ax.set_title('Loss vs Training Steps')
                    ax.legend()
                    file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
                    plt.savefig(file_path)

                    ###################################################################################

                    # export metrics all together
                    metrics_csv_pattern = os.path.join(c.summary_out_dir, "metrices.csv")
                    metrics = pd.read_csv(metrics_csv_pattern) 
                    mse_learned = metrics.loc[metrics['Metric'] == 'MSE', 'Learned'].values[0]
                    mse_test = metrics.loc[metrics['Metric'] == 'MSE', 'Test'].values[0]
                    # Create a new DataFrame for the row to be added
                    new_row_df = pd.DataFrame([{
                        'tag': 'FBPINN',
                        'model_type': name,
                        'time_limit': tl_str, 
                        'mse_learned': mse_learned,
                        'mse_test': mse_test
                    }])

                    # Concatenate the new row with the existing metrics DataFrame
                    metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)

                    # export parameters all together
                    parameters_csv_pattern = os.path.join(c.summary_out_dir, "parameters.csv")
                    parameters = pd.read_csv(parameters_csv_pattern)
                    
                    true_params = parameters['True'].tolist()
                    learned_params = parameters['Learned'].tolist()
                    if len(true_params)==1:
                        true_params = true_params[0]
                        learned_params = learned_params[0]
                    else:
                        true_params = np.array(true_params)
                        learned_params = np.array(learned_params)
                    new_row_df = pd.DataFrame([{
                        'tag': 'FBPINN',
                        'model_type': name,
                        'time_limit': tl_str, 
                        'true_params': true_params,
                        'learned_params': learned_params
                    }])
                    parameters_df = pd.concat([parameters_df, new_row_df], ignore_index=True)

        
        # save loss landscape plots
        fig_ll.tight_layout()
        fig_ll.savefig(f"{parent_dir}/loss_landscape_{name}.png")
        fig_ll_3D.subplots_adjust(wspace=0.4, hspace=0.3)
        fig_ll_3D.tight_layout(rect=[0, 0, 1, 0.95])
        fig_ll_3D.savefig(f"{parent_dir}/loss_landscape3D_{name}.png")
        # plot exact solutions
        for i in range(0, u_exact.shape[1]):
            model_nn_ax[cases.index(name)].plot(x_batch, u_exact[:,i:i+1], label=f"{labels[i]}_true", linestyle='-') #color='black')
            model_learned_ax[cases.index(name)].plot(x_batch, u_exact[:,i:i+1], label=f"{labels[i]}_true", linestyle='-') #, color='black'
            model_nn_ax[cases.index(name)].legend()
            model_learned_ax[cases.index(name)].legend()
            model_nn_ax[cases.index(name)].set_title(f"{fig_tags[fig_tags.index(fig_tag)]}{name}")
            model_learned_ax[cases.index(name)].set_title(f"{fig_tags[fig_tags.index(fig_tag)]}{name}")
            model_nn_ax[cases.index(name)].set_xlabel('Time')
            model_learned_ax[cases.index(name)].set_xlabel('Time')
            model_nn_ax[cases.index(name)].set_ylabel('Population')
            model_learned_ax[cases.index(name)].set_ylabel('Population')

        print("Done!")

    # save the fig_energy
    fig_energy.tight_layout()
    fig_energy.savefig(f"{parent_dir}/energy_plot.png")
    # loss landscape plots save
    fig_all.tight_layout()
    fig_all.savefig(f"{parent_dir}/loss_landscape_all.png")
    # save metrics
    metrics_df.to_csv(f"{parent_dir}/FBPINN_combined_metrices.csv", index=False)
    # save parameters
    parameters_df.to_csv(f"{parent_dir}/FBPINN_combined_parameters.csv", index=False)
    # save model_nn_ax and model_learned_ax
    model_fig1.tight_layout()
    model_fig1.savefig(f"{parent_dir}/model_nn.png")
    model_fig2.tight_layout()
    model_fig2.savefig(f"{parent_dir}/model_learned.png")

if __name__=="__main__":
    train_FBPINN_models()
