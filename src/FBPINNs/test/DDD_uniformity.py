import sys
import os

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import argparse
from fbpinns.domains import RectangularDomainND
from FBPINNsModel.problems import CompetitionModel, SaturatedGrowthModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import load_model
import matplotlib.pyplot as plt
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters, get_x_batch
from FBPINNsModel.subdomain_helper import get_subdomain_xsws, get_subdomain_xs_uniform_center

parser = argparse.ArgumentParser(description='''
        FBPINN code for Separating longtime behavior and learning of mechanics  \n
        To find the mse of uniform and nonuniform DD for different time limit'''
)

parser.add_argument('--train', help='To train all models (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()

def plot_DDD_uniformity():
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
    sparse = True
    noise_level = 0.05
    nsub = 2
    tbegin = 0
    tend = 24
    wo = 1.9
    wi = 1.0005
    tag = "DDD"
    epochs = 50000
    sampler='grid'

    # varying parameters
    time_limits = [[0,10], [10,24], [0, 24]]
    subdomain_types = [get_subdomain_xsws, get_subdomain_ws]

    # type of problem and other fix params
    layer1 = [1, 5, 5, 5, 1]
    layer2 = [1, 5, 5, 5, 2]
    layers = [layer1, layer2, layer2]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    params_type = [1,  [0.5, 0.7, 0.3, 0.3, 0.6], [0.5, 0.3, 0.6, 0.7, 0.3]] # sg, coexistence, survival
    names = ["sg", "coexistence", "survival"]
    parentdir = "DDD_uniformity"
    rootdirs = ["DDD_uniformity_sg", "DDD_uniformity_coexistence", "DDD_uniformity_Survival"]
    rootdirs_with_parent = [os.path.join(parentdir, rd) for rd in rootdirs]

    for (layer, problem, params, rootdir, name) in zip(layers, problem_types, params_type, rootdirs_with_parent, names):
        
        #step 1
        domain = RectangularDomainND
        domain_init_kwargs = dict(
            xmin = np.array([tbegin,]), 
            xmax = np.array([tend,])
            )
        #step 2
        problem_kwargs_set = []
        if problem.__name__=="CompetitionModel":
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                problem_init_kwargs = dict(
                        params=params, u0=2, v0=1, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
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
        for subdomains in subdomain_types:
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                if subdomains==get_subdomain_ws:
                    subdomain_xs= get_subdomain_xs_uniform_center(tbegin, tend, nsub)
                    subdomain_ws=get_subdomain_ws(subdomain_xs, wo)
                elif subdomains==get_subdomain_xsws:
                    subdomain_xs, subdomain_ws = get_subdomain_xsws(tl, tbegin, tend, nsub, wo, wi)
                else:
                    raise ValueError("Invalid type specified.")
                decomposition_init_kwargs=dict(
                                        subdomain_xs=subdomain_xs,
                                        subdomain_ws=subdomain_ws,
                                        unnorm=(0.,1.),
                                    )
                subName = (lambda: "uniform" if subdomains == get_subdomain_ws else "nonuniform")()
                decomposition_kwargs_set.append((decomposition_init_kwargs, subName, tl))

        # step 4
        network = FCN# place a fully-connected network in each subdomain
        network_init_kwargs=dict(
            layer_sizes=layer,# with 2 hidden layers
        )

        h = len(layer) -2
        p = sum(layer[1:-1])
        n=(nCol,)

        runs = []
        for (problem_kwargs, tl) in problem_kwargs_set:
            for (decomposition_kwargs, subName, tld) in decomposition_kwargs_set:
                if tl==tld:
                    run = f"FBPINN_{tag}_{name}_{subName}_xs_{wo}_wo_{tl}_tl"
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
                        # import model
                        c_out, model = load_model(run, rootdir=rootdir+"/")
                        print(c_out)

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
                        ax.plot(i, l1n, label=f"FBPINN {h} l {p} hu {nsub} ns")
                        ax.set_yscale('log')
                        ax.set_xlabel('Training Steps')
                        ax.set_ylabel('Normalized l1 test loss')
                        ax.set_title('Loss vs Training Steps')
                        ax.legend()
                        file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
                        plt.savefig(file_path)

        #  make a fig with 2 subplot [2,2] shape to plot models
        models_fig, models_ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        for run in runs:
            c_out, model = load_model(run, rootdir=rootdir+"/")
            subname = run.split("_")[3]
            tl = c_out.problem_init_kwargs['time_limit']
            column_index = 0 if tl == [0,10] else 1 if tl == [10,24] else -1 
            if column_index != -1:
                u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
                x_batch = get_x_batch(c, model)
                labels  = ["u", "v"]    
                for i in range(0, u_exact.shape[1]):
                    models_ax[0, column_index].plot(x_batch, u_learned[:,i], '-.', label=f"{labels[i]}-{subname}")
                    models_ax[1, column_index].plot(x_batch, u_test[:,i], ':', label=f"{labels[i]}-{subname}") 

        for i in range(u_exact.shape[1]):  
            models_ax[0,0].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true") 
            models_ax[1,0].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true")  
            models_ax[0,1].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true") 
            models_ax[1,1].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true")   
        for i in range(2):  # For each row
            for j in range(2):  # For each column
                models_ax[i][j].set_title(f"{'Learned' if i == 0 else 'Test'}-tl[{'0-10' if j == 0 else '10-24'}]")
                models_ax[i][j].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
        models_fig.tight_layout()
        # models_fig.suptitle(f"Model Comparison for {name}", fontsize=14, verticalalignment='top')# , y=0.95)
        file_path = f"{parentdir}/Models_plot({name}).png"
        models_fig.savefig(file_path)

        results = {
            "0-10": {"uniform": [], "nonuniform": []},
            "10-24": {"uniform": [], "nonuniform": []},
            "0-24": {"uniform": [], "nonuniform": []}
        }

        for run in runs:
            c_out, model = load_model(run, rootdir=rootdir+"/")

            parts = c_out.run.split("_")
            subName_index = 3 
            subName = parts[subName_index]
            
            tl = c_out.problem_init_kwargs['time_limit']
            tl_key = f"{tl[0]}-{tl[1]}"
            
            u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
            mse_test = np.mean((u_exact - u_test)**2)
            mse_learned = np.mean((u_exact - u_learned)**2)

            results[tl_key][subName].append((mse_test, mse_learned))

        time_limits_col = ["0-10", "10-24", "0-24"]
        subNames = ['uniform', 'nonuniform']
        mse_types = ['mse_learned', 'mse_test']

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05})

        lambda_phy_tex = r"$\lambda_{\mathrm{phy}}$"
        lambda_data_tex = r"$\lambda_{\mathrm{data}}$"

        params_text = (f"• nD: {numx} " +
                    f"• nC: {nCol} " +
                    f"• nT: {nTest} " +
                    f"• layer: {layer} " +
                    f"• epochs: {epochs} " + "\n"
                    f"• {lambda_phy_tex}: {lambda_phy} " +
                    f"• {lambda_data_tex}: {lambda_data} " + 
                    f"• Sparse: {sparse} " +
                    f"• optimiser: {c_out.optimiser.__name__} " +
                    f"• lr: {c_out.optimiser_kwargs["learning_rate"]} " + "\n"
                    f"• DD: Nonuniform " +
                    f"• Noise: {noise_level} " +
                    f"• nSub: {nsub} " + 
                    f"• wo: {wo} " +
                    f"• wi: {wi} " +
                    f"• Problem: {problem.__name__ if hasattr(problem, '__name__') else problem}({name})")

        ax0.text(0.5, 0.5, params_text, ha='center', va='center', fontsize=12, transform=ax0.transAxes)
        ax0.set_frame_on(True) 
        ax0.get_xaxis().set_visible(False) 
        ax0.get_yaxis().set_visible(False)  

        barWidth = 0.15
        r1 = np.arange(len(time_limits_col))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]

        # Creating the bars in the bottom subplot (ax1)
        for i, tl in enumerate(time_limits_col):
            for subName in subNames:
                mse_tests, mse_learneds = zip(*results[tl][subName])

                if subName == 'uniform':
                    ax1.bar(r1[i], np.mean(mse_learneds), color='blue', width=barWidth, edgecolor='grey', label='uniform learned' if i == 0 else "")
                    ax1.bar(r2[i], np.mean(mse_tests), color='red', width=barWidth, edgecolor='grey', label='uniform test' if i == 0 else "")
                else:
                    ax1.bar(r3[i], np.mean(mse_learneds), color='blue', width=barWidth, edgecolor='grey', label='nonuniform learned' if i == 0 else "", hatch='/', linewidth=3)
                    ax1.bar(r4[i], np.mean(mse_tests), color='red', width=barWidth, edgecolor='grey', label='nonuniform test' if i == 0 else "", hatch='/', linewidth=3)

        ax1.set_yscale('log')
        ax1.set_xlabel('Time Limit', fontweight='bold', fontsize=12)
        ax1.set_ylabel('MSE Value', fontweight='bold', fontsize=12)
        ax1.set_xticks([r + barWidth for r in range(len(time_limits_col))])
        ax1.set_xticklabels(time_limits_col)
        ax1.legend()

        plt.suptitle('MSE Values by Time Limit and Subdomain Type', fontsize=14, verticalalignment='top', y=0.95)
        file_path = f"{parentdir}/MSE_varying_uniformity({name}).png"
        plt.savefig(file_path)
        
        print("DONE")


if __name__=="__main__":
    plot_DDD_uniformity()