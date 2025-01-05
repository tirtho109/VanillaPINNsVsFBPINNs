"""
    To train: python varying_CollocationPoints.py --train True -e 50000 --gpu True
    To plot: python varying_CollocationPoints.py -e 50000 --gpu True

"""

import os, sys, re, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from SciannModel.model_ode_solver import SaturatedGrowthModel, CompetitionModel, cust_semilogx

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 5000)', type=int, nargs=1, default=[5000])

args = parser.parse_args()

parent_output_path = os.path.join(os.getcwd(), f'VARYING_COLLOCATION_POINTS_{args.epochs[0]}/')
os.makedirs(parent_output_path, exist_ok=True) 

def plot_varying_CollocationPoints():
    training_time = time.time()

    # fixed parameters
    layers = [5]*3
    actf = 'tanh'
    noise_level = 0.05
    epochs = args.epochs[0]
    learningrate = 0.001
    independent_networks = False
    verbose = 2
    tend = 24
    numx = 100
    nTest = 500
    sparse = True
    gpu = args.gpu[0]

    # varying parameters
    nCols = np.arange(100, 501, 100)
    time_limits = [[0,10], [10,24], [0, 24]]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    model_types = ["sg", "coexistence", "survival"]

    # collector dataframes
    collcted_df = pd.DataFrame(columns=['Time Limit', 'Collocation Points', 'MSE Learned', 'MSE Test', 'Model Type', 'Learned Parameters'])

    # initialize the root directories
    rootdirs = []
    for model_type in model_types:
        output_path = os.path.join(parent_output_path, f'VARYING_COLLOCATION_POINTS_{model_type.upper()}/')
        os.makedirs(output_path, exist_ok=True)
        rootdirs.append(output_path)

    for (problem, rootdir, model_type) in zip(problem_types, rootdirs, model_types):
        if problem.__name__=='SaturatedGrowthModel':
            model_args = {
                'layers': layers,
                'actf': actf,
                'noise_level': noise_level,
                'epochs': epochs,
                'learningrate': 0.001,
                'independent_networks': independent_networks,
                'verbose': verbose,
                'initial_conditions': 0.01,
                'tend': tend,
                'numx': numx,
                'nTest': nTest,
                # 'batchsize': 25, # numx + nCol
                'sparse':sparse,
                'outputpath': rootdir,
                'gpu': gpu,
            }
        else:
            model_args = {
                'model_type': model_type,
                'layers': layers,
                'actf': actf,
                'noise_level': noise_level,
                'epochs': epochs,
                'learningrate': learningrate,
                'independent_networks': independent_networks,
                'verbose': verbose,
                'initial_conditions': [2,1],
                'tend': tend,
                'numx': numx,
                'nTest': nTest,
                # 'batchsize': 25, # numx + nCol
                'sparse': sparse,
                'outputpath': rootdir,
                'gpu': gpu,
            }
        # Loop over the varying parameters and train the models
        processes = []
        for nCol in nCols:
            for time_limit in time_limits:
                model_args['nCol'] = nCol
                model_args['time_limit'] = time_limit
                model_args['batchsize'] = numx + nCol
                
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
        if args.train[0]:
            for process in processes:
                process.wait()


        ################################################################
        ################### plot the results ###########################
        ################################################################
        fig = plt.figure(figsize=(12,10), dpi=300)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 5, 4])
        gs_lossplot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.3) 

        ax1 = fig.add_subplot(gs_lossplot[0, 0])
        ax2 = fig.add_subplot(gs_lossplot[0, 1])
        ax3 = fig.add_subplot(gs_lossplot[0, 2])
        axis_map = {'0-10': ax1, '0-24': ax2, '10-24': ax3}

        df = pd.DataFrame(columns=['Time Limit', 'Collocation Points', 'MSE Learned', 'MSE Test'])

        for dirpath, dirnames, filenames in os.walk(rootdir):
            if "res_loss" in filenames and "res_time" in filenames and "metrices.csv" in filenames:
                total_loss = np.loadtxt(os.path.join(dirpath, "res_loss"))
                
                dir_time_limit = re.search('_tl_(\d+-\d+)', dirpath)
                if dir_time_limit:
                    dir_time_limit = list(map(int, dir_time_limit.group(1).split('-')))
                    dir_time_limit_str = '-'.join(map(str, dir_time_limit))
                    # Find the corresponding axis based on the time limit
                    ax = axis_map.get(dir_time_limit_str)
                    if ax is None:
                        raise ValueError(f"Invalid time_limit key: {dir_time_limit_str}") 
                    
                    nCol = 'unknown'
                    nCol_match = re.search('_nC_(\d+)', dirpath)
                    if nCol_match:
                        nCol = nCol_match.group(1)
                cust_semilogx(ax, None, total_loss/total_loss[0], "epochs", "L/L0", label=f"nCol-{nCol}")
                ax.set_title(f'Time Limit: {dir_time_limit}')
                ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.2), loc='upper center', fontsize='small')

                ############################## [mse/mae] add to dataframe ##############################
                # Load the mse and mae values
                metrices = pd.read_csv(os.path.join(dirpath, "metrices.csv"))
                mse_learned = metrices.loc[metrices['Metric'] == 'MSE', 'Learned'].values[0]
                mse_test = metrices.loc[metrices['Metric'] == 'MSE', 'Test'].values[0]

                new_row = pd.DataFrame({'Time Limit': [dir_time_limit_str], 'Collocation Points': [nCol],
                                'MSE Learned': [mse_learned], 'MSE Test': [mse_test]})
                df = pd.concat([df, new_row], ignore_index=True)

                if model_type == 'sg':
                    # collect the file with file name 'res_C_learned'
                    if 'res_C_learned' in filenames:
                        param_learned = np.loadtxt(os.path.join(dirpath, 'res_C_learned'))
                else:
                    if 'res_learned_params' in filenames:
                        param_learned = np.loadtxt(os.path.join(dirpath, 'res_learned_params'))

                new_collcected_row = pd.DataFrame({'Time Limit': [dir_time_limit_str], 'Collocation Points': [nCol],
                                'MSE Learned': [mse_learned], 'MSE Test': [mse_test], 'Model Type': [model_type], 'Learned Parameters': [param_learned]})
                
                collcted_df = pd.concat([collcted_df, new_collcected_row], ignore_index=True)
        
        df['MSE Learned'] = pd.to_numeric(df['MSE Learned'], errors='coerce')
        df['MSE Test'] = pd.to_numeric(df['MSE Test'], errors='coerce')

        df['Collocation Points'] = df['Collocation Points'].astype(int)
        df = df.sort_values('Collocation Points')

        pivot_learned = df.pivot(index="Collocation Points", columns="Time Limit", values="MSE Learned")
        pivot_test = df.pivot(index="Collocation Points", columns="Time Limit", values="MSE Test")

        pivot_learned_log = pivot_learned.applymap(lambda x: np.log10(x + 1e-10))  # Adding a small number to avoid log(0)
        pivot_test_log = pivot_test.applymap(lambda x: np.log10(x + 1e-10))

        # heatmaps
        gs_heatmaps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3) 

        ax4 = fig.add_subplot(gs_heatmaps[0, 0])
        ax5 = fig.add_subplot(gs_heatmaps[0, 1])

        def highlight_min(data, ax, highlight_color='red'):
            for col in data.columns:
                min_row = data[col].idxmin()
                ax.add_patch(plt.Rectangle((data.columns.tolist().index(col), data.index.tolist().index(min_row)),
                                            1, 1, fill=True, facecolor=highlight_color, alpha=0.5, edgecolor=highlight_color, lw=2))

        # Heatmap for MSE 
        sns.heatmap(pivot_learned_log, annot=True, fmt=".2f", cmap='viridis', ax=ax4)
        highlight_min(pivot_learned_log, ax4)
        ax4.set_title('Log MSE Learned')
        ax4.invert_yaxis()

        sns.heatmap(pivot_test_log, annot=True, fmt=".2f", cmap='viridis', ax=ax5)
        highlight_min(pivot_test_log, ax5)
        ax5.set_title('Log MSE Test')
        ax5.invert_yaxis()

        ####################################################
        #################### Text plot #####################
        ####################################################
        ax0 = fig.add_subplot(gs[0, 0])
        lambda_phy_tex = r"$\lambda_{\mathrm{phy}}$"
        lambda_data_tex = r"$\lambda_{\mathrm{data}}$"

        params_text = (f"• noise_level: {model_args['noise_level']} " + 
               f"• nD: {model_args['numx']} " +
               f"• nT: {model_args['nTest']} " +
               f"• layer: {model_args['layers']} " +
               f"• epochs: {model_args['epochs']} " + 
               f"• learning_rate: {model_args['learningrate']} " + "\n"
               f"• batchsize: {model_args['numx']} + nCol " +
               f"• {lambda_phy_tex}: 1 " + # default: 1 #TODO: add the lambda_phy
               f"• {lambda_data_tex}: 1 " +  # default: 1 #TODO: add the lambda_data
               f"• Sparse: {model_args['sparse']} " + 
               f"• Problem: {problem.__name__ if hasattr(problem, '__name__') else problem}"
               f"({model_type})")
        
        ax0.text(0.5, 0.5, params_text, ha='center', va='center', fontsize=12)
        ax0.set_frame_on(False)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)


        plt.suptitle('SciANN MSE Value by Time Limit and Collocation Points', fontsize=14, verticalalignment='top')#, y=0.95)
        plt.subplots_adjust(hspace=0.2, top=0.88)
        plt.tight_layout()
        plt.savefig(f"{rootdir}varying__CollocationPoints__({model_type}).png")
    
    training_time = time.time() - training_time
    print(f"Total training time: {training_time} seconds")

    # save collected df
    collcted_df.to_csv(f"{parent_output_path}SciANN_collected_info_varying_CollocationPoints.csv", index=False)
    print("DONE")

if __name__=="__main__":
    plot_varying_CollocationPoints()
