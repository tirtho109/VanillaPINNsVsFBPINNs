"""
    To train: python compare_with_fbpinns.py --train True -e 50000 --gpu True
    To plot: python compare_with_fbpinns.py -e 50000 --gpu True

"""

import os, sys, re, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from SciannModel.model_ode_solver import SaturatedGrowthModel, CompetitionModel, cust_semilogx, model_comparison, plot_loss_landscape_contour, plot_loss_landscape3D

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import plot_energy_from_params

import subprocess

parser = argparse.ArgumentParser(description=''' 
    FBPINN code for Separating longtime behavior and learning of mechanics  \n
    To find the mse and loss by varying noise based on different time limit cases \n
    The model's are only valid for time_limits [0,10], [10,24], [0,24] \n
    make sure to change the time_limits manually for reuse the code for other time_limits'''
)

parser.add_argument('--train', help='To train the whole model (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)
parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 50000)', type=int, nargs=1, default=[50000])

args = parser.parse_args()

parent_output_path = os.path.join(os.getcwd(), f'CompareWith_FBPINNs_{args.epochs[0]}/')
os.makedirs(parent_output_path, exist_ok=True)

def train_sciann_models():
    training_time = time.time()

    # fixed parameters
    actf = 'tanh'
    noise_level = 0.05
    epochs = args.epochs[0]
    learningrate = 0.001
    independent_networks = False
    verbose = 2
    tend = 24
    nCol = 200
    numx = 100
    nTest = 500
    sparse = True
    gpu = args.gpu[0]
    layers = [5,5,5]

    # varying parameters
    time_limits = [[0,10], [10,24], [0, 24]]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    model_types = ["sg", "coexistence", "survival"]

    # initialize the root directories
    rootdirs = []
    for model_type in model_types:
        output_path = os.path.join(parent_output_path, f'{model_type}/')
        os.makedirs(output_path, exist_ok=True)
        rootdirs.append(output_path)

    processes = []
    for (problem, rootdir, model_type) in zip(problem_types, rootdirs, model_types):
        if problem.__name__=='SaturatedGrowthModel': 
            model_args = {
                'numx': numx,
                'actf': actf,
                'noise_level': noise_level,
                'epochs': epochs,
                'learningrate': 0.001,
                'independent_networks': independent_networks,
                'verbose': verbose,
                'initial_conditions': 0.01,
                'tend': tend,
                'nCol': nCol,
                'nTest': nTest,
                'batchsize': numx + nCol, # numx + nCol
                'sparse':sparse,
                'outputpath': rootdir,
                'gpu': gpu,
                'layers': layers,
            }
        else:
            model_args = {
                'model_type': model_type,
                'numx': numx,
                'actf': actf,
                'noise_level': noise_level,
                'epochs': epochs,
                'learningrate': learningrate,
                'independent_networks': independent_networks,
                'verbose': verbose,
                'initial_conditions': [2,1],
                'tend': tend,
                'nCol': nCol,
                'nTest': nTest,
                'batchsize': numx + nCol, # numx + nCol
                'sparse': sparse,
                'outputpath': rootdir,
                'gpu': gpu,
                'layers': layers,
            }

        # loop over the varying parameters and train the models
        
        for time_limit in time_limits:
            model_args['time_limit'] = time_limit
            # Prepare the command [problem based]
            if problem.__name__=='SaturatedGrowthModel':
                command = ['python', '../SciannModel/SciANN_SaturatedGrowthModel.py']
            else:
                command = ['python', '../SciannModel/SciANN_CompetitionModel.py']

            for key, value in model_args.items():
                command.append('--' + key)
                if isinstance(value, list):
                    command.extend(map(str, value))  # add each item in the list separately
                else:
                    command.append(str(value))
            ############################# # Train the model (if train==True) #############################
            if args.train[0]:    
                # Run the command
                process = subprocess.Popen(command)
                processes.append(process)

    # Train alltogether
    if args.train[0]:
        for process in processes:
            process.wait()

    # plotting scheme
    # for model comparision
    model_fig1, model_nn_ax = plt.subplots(1, 3, figsize=(12, 4))
    model_fig2, model_learned_ax = plt.subplots(1, 3, figsize=(12, 4))
    tend = 24
    tspan = np.linspace(0, tend, 500)
    time_labels = ["0-10", "10-24", "0-24"]

    # make a list of colors for the plots mainly black, red, and blue
    colors = ['black', 'red', 'blue']
    fig_tags = ['(a)', '(b)', '(c)']

    all_fig_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    # fig for plotting all ll together
    fig_all, ax_all = plt.subplots(3, 3, figsize=(12, 9), dpi=300) # only for contour plot

    # create fig for energy plot of coexistence and survival
    fig_energy, ax_energy = plt.subplots(2, 4, figsize=(12, 6), dpi=300)

    # create a dataframe to store the metrics
    metrics_df = pd.DataFrame(columns=['tag', 'model_type', 'time_limit', 'mse_learned', 'mse_test'])

    # create a dataframe to store parameters 
    parameters_df = pd.DataFrame(columns=['tag', 'model_type', 'time_limit', 'true_params', 'learned_params'])
    
    for (problem, rootdir, model_type, fig_tag) in zip(problem_types, rootdirs, model_types, fig_tags):
        file_names = ["res_C_true", "res_C_learned"] if problem.__name__=='SaturatedGrowthModel' else ["res_params", "res_learned_params"]
        labels = ["u", "v"]
        # fig for loss landscape
        fig_ll, ax_ll = plt.subplots(1, 3, figsize=(12, 4), dpi=300) # only for contour plot
        fig_ll_3D = plt.figure(figsize=(15, 5), dpi=300)
        ax_ll_3d = [fig_ll_3D.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
        
        for sub_dir in os.listdir(rootdir):
            param_list = []
            dir_name = str(sub_dir)
            tl = re.findall(r'tl_(\d+-\d+)', dir_name)[0]
            output_path = os.path.join(rootdir, sub_dir) # add subdir as the current output path
            if os.path.isdir(output_path):
                for file_name in file_names:
                    file_path = os.path.join(output_path, f"{file_name}")
                    if os.path.isfile(file_path):
                        file_data = np.loadtxt(file_path)
                        param_list.append(file_data)
            # param_list = [true_params, learned_params]
            model = SaturatedGrowthModel if problem.__name__=='SaturatedGrowthModel' else CompetitionModel
            true_model = model(param_list[0], 0.01 if problem.__name__=='SaturatedGrowthModel' else [2,1], tend)    
            learned_model = model(param_list[1], 0.01 if problem.__name__=='SaturatedGrowthModel' else [2,1], tend)
            _, true_sol = true_model.solve_ode(true_model.initial_conditions, tspan)
            _, learned_sol = learned_model.solve_ode(learned_model.initial_conditions, tspan)
            # import file with name "_t_u_pred.csv" at the end in this subdir
            csv_pattern = os.path.join(output_path, "res_t_u_pred.csv" if problem.__name__=='SaturatedGrowthModel' else "res_t_u_v_pred.csv")
            nn_pred_solution = np.loadtxt(csv_pattern, delimiter=',', skiprows=1)
            for i in range(0,true_sol.shape[1]):
                model_learned_ax[model_types.index(model_type)].plot(tspan, learned_sol[:,i], label=f"{labels[i]}_l_{tl}", linestyle='--' if i==0 else '-.', color=colors[time_labels.index(tl)])
                model_nn_ax[model_types.index(model_type)].plot(tspan, nn_pred_solution[:,i+1], label=f"{labels[i]}_nn_{tl}", linestyle='--' if i==0 else '-.', color=colors[time_labels.index(tl)])
            
            # loss landscape plot
            loss_landscape_csv_pattern = os.path.join(output_path, "loss-landscape-history-landscape.csv")
            plot_loss_landscape_contour(loss_landscape_csv_pattern, ax_all[model_types.index(model_type) ,time_labels.index(tl)], title=f"{all_fig_tags[model_types.index(model_type)*3 + time_labels.index(tl)]} {model_type}[{tl}]")
            plot_loss_landscape_contour(loss_landscape_csv_pattern, ax_ll[time_labels.index(tl)], title=f"{fig_tags[time_labels.index(tl)]}{tl}")
            plot_loss_landscape3D(loss_landscape_csv_pattern, ax_ll_3d[time_labels.index(tl)], title=f"{fig_tags[time_labels.index(tl)]}{tl}")

            # export metrics
            metrics_csv_pattern = os.path.join(output_path, "metrices.csv")
            metrics = pd.read_csv(metrics_csv_pattern, index_col=0)
            metrics_df = metrics_df.append({'tag':'SciANN', 'model_type':model_type, 'time_limit':tl, 'mse_learned':metrics['Learned']['MSE'], 'mse_test':metrics['Test']['MSE']}, ignore_index=True)

            # export parameters
            # make each element of the param_list[1] upto 3 decimal point
            param_list[1] = np.round(param_list[1], 4)
            parameters_df = parameters_df.append({'tag':'SciANN', 'model_type':model_type, 'time_limit':tl, 'true_params':param_list[0], 'learned_params':param_list[1]}, ignore_index=True)

            # plot energy together
            if model_type == 'coexistence':
                if tl=="0-10":
                     plot_energy_from_params(param_list[0], model_type="coexistence", axis=ax_energy[0, 0], title=f"{all_fig_tags[time_labels.index(tl)]}coexistence[True]")
                plot_energy_from_params(param_list[1], model_type="coexistence", axis=ax_energy[0, time_labels.index(tl)+1], title=f"{all_fig_tags[time_labels.index(tl)+1]}coexistence_[{tl}]")
            elif model_type == 'survival':
                if tl=="0-10":
                    plot_energy_from_params(param_list[0], model_type="survival", axis=ax_energy[1, 0], title=f"{all_fig_tags[time_labels.index(tl)+4]}survival[True]")
                plot_energy_from_params(param_list[1], model_type="survival", axis=ax_energy[1, time_labels.index(tl)+1], title=f"{all_fig_tags[time_labels.index(tl)+5]}survival_[{tl}]")


        # save loss landscape plot
        fig_ll.tight_layout()
        fig_ll.savefig(os.path.join(parent_output_path, f'loss_landscape_{model_type}.png'))
        # save loss landscape 3D plot
        fig_ll_3D.subplots_adjust(wspace=0.4, hspace=0.3)
        fig_ll_3D.tight_layout(rect=[0, 0, 1, 0.95])
        fig_ll_3D.savefig(os.path.join(parent_output_path, f'loss_landscape_3D_{model_type}.png'))

        for i in range(0,true_sol.shape[1]):
            model_nn_ax[model_types.index(model_type)].plot(tspan, true_sol[:,i], label=f"{labels[i]}_true", linestyle='-')
            model_learned_ax[model_types.index(model_type)].plot(tspan, true_sol[:,i], label=f"{labels[i]}_true", linestyle='-')
            model_nn_ax[model_types.index(model_type)].legend()
            model_learned_ax[model_types.index(model_type)].legend()
            model_nn_ax[model_types.index(model_type)].set_title(f"{fig_tags[fig_tags.index(fig_tag)]}{model_type}")
            model_learned_ax[model_types.index(model_type)].set_title(f"{fig_tags[fig_tags.index(fig_tag)]}{model_type}")
            model_nn_ax[model_types.index(model_type)].set_xlabel('Time')
            model_learned_ax[model_types.index(model_type)].set_xlabel('Time')
            model_nn_ax[model_types.index(model_type)].set_ylabel('Population')
            model_learned_ax[model_types.index(model_type)].set_ylabel('Population')
        
    
        print("Done")

    # save the energy plots
    fig_energy.tight_layout()
    fig_energy.savefig(os.path.join(parent_output_path, 'energy_plots.png'))

    # save the loss landscape plots altogether
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(parent_output_path, 'loss_landscape_all.png'))
    # save the metrics dataframe
    metrics_df.to_csv(os.path.join(parent_output_path, 'SciANN_combined_metrices.csv'), index=False)

    # save the parameters dataframe
    parameters_df.to_csv(os.path.join(parent_output_path, 'SciANN_combined_params.csv'), index=False)

    model_fig1.tight_layout()
    model_fig2.tight_layout()
    # Save the plots
    model_fig1.savefig(os.path.join(parent_output_path, 'model_comparison_nn.png'))
    model_fig2.savefig(os.path.join(parent_output_path, 'model_comparison_learned.png'))

if __name__ == '__main__':
    train_sciann_models()
