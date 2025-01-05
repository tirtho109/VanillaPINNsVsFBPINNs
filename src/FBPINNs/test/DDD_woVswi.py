import sys
import os

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import seaborn as sns
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
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters
from FBPINNsModel.subdomain_helper import get_subdomain_xsws

parser = argparse.ArgumentParser(description='''
        FBPINN code for Separating longtime behavior and learning of mechanics  \n
        To find the best number of subdomain for different time limit cases'''
)

parser.add_argument('--train', help='To train the whole model (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()


def plot_DDD_varying_overlap():
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
    tbegin = 0
    tend = 24
    # wo = 1.9
    nsub=2
    # wi = 1.0005
    tag = "DDD"
    epochs = 50000
    sampler='grid'

    # Varying parameters
    # wos =[1.1, 1.5, 1.9, 2.3, 2.7]
    wos = np.linspace(1.001, 1.01, 4)
    wis = np.linspace(1.001, 1.01, 4)
    time_limits = [[0, 10], [10,24]]

    # type of problem and other fix params
    layer1 = [1, 5, 5, 5, 1]
    layer2 = [1, 5, 5, 5, 2]
    layers = [layer1, layer2, layer2]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    params_type = [1,  [0.5, 0.7, 0.3, 0.3, 0.6], [0.5, 0.3, 0.6, 0.7, 0.3]] # sg, coexistence, survival
    cases = ["sg", "coexistence", "survival"]

    parentdir = "DDD_woVswi"
    rootdirs = ["DDD_woVswi_sg", "DDD_woVswi_coexistence", "DDD_woVswi_survival"]
    rootdirs_with_parent = [os.path.join(parentdir, rd) for rd in rootdirs]

    # to store num of collocation points in each overlap
    col_df = pd.DataFrame(columns=["model_type", "time_window","wo", "wi", "col_points"])

    for (layer, problem, params, rootdir, name) in zip(layers, problem_types, params_type, rootdirs_with_parent, cases):
        
        runs=[]

        for tl in time_limits:

            # step 1:
            domain = RectangularDomainND
            domain_init_kwargs = dict(
                xmin = np.array([tbegin,]), 
                xmax = np.array([tend,])
                )
            
            # step 2:
            if problem.__name__=="CompetitionModel":
                problem_init_kwargs = dict(
                        params=params, u0=2, v0=1, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
                        sparse=[sparse], noise_level=noise_level,
                    )
            else:
                problem_init_kwargs = dict(
                        C=params, u0=0.01, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
                        sparse=[sparse], noise_level=noise_level,
                    )
                
            # step 3
            decomposition = RectangularDecompositionND
            decomposition_kwargs_set = []
            for wo in wos:
                for wi in wis:
                    subdomain_xs, subdomain_ws = get_subdomain_xsws(tl, tbegin, tend, nsub, wo, wi)
                    decomposition_init_kwargs = dict(
                        subdomain_xs=subdomain_xs,
                        subdomain_ws=subdomain_ws,
                        unnorm=(0.,1.),
                    )
                    decomposition_kwargs_set.append((decomposition_init_kwargs, wo, wi))

            # step 4
            network = FCN# place a fully-connected network in each subdomain
            network_init_kwargs=dict(
                layer_sizes=layer,# with 2 hidden layers
            )

            h = len(layer) -2
            p = sum(layer[1:-1])
            n=(nCol,)

            for (decomposition_kwargs, wo, wi) in decomposition_kwargs_set:
                run = f"FBPINN_{problem.__name__}_{tag}_{nsub}_nsub_{wo}_wo_{wi}_wi_{tl}_tl"
                runs.append(run)
                c = ModifiedConstants(
                    run = run,
                    domain= domain,
                    domain_init_kwargs=domain_init_kwargs,
                    problem=problem,
                    problem_init_kwargs=problem_init_kwargs,
                    decomposition=decomposition,
                    decomposition_init_kwargs=decomposition_kwargs,
                    network=network,
                    network_init_kwargs=network_init_kwargs,
                    ns=((nCol,),),
                    n_test=(nTest,),
                    n_steps=epochs,
                    clear_output=True,
                    sampler=sampler,
                    show_figures=False,
                    save_figures=True,
                )

                # collocation points in the overlaps between subdomains
                subdomain_xs, subdomain_ws = get_subdomain_xsws(tl, tbegin, tend, nsub, wo, wi)
                collocation_points = np.linspace(domain_init_kwargs['xmin'][0], domain_init_kwargs['xmax'][0], nCol)
                spans = [[subdomain_xs[0][i] - subdomain_ws[0][i]/2 , subdomain_xs[0][i] + subdomain_ws[0][i]/2] for i in range(len(subdomain_xs[0]))]

                overlaps = []
                for i in range(len(spans) - 1):
                    # Overlap is the intersection of the current span with the next one
                    overlap_start = max(spans[i][0], spans[i+1][0])
                    overlap_end = min(spans[i][1], spans[i+1][1])
                    if overlap_start < overlap_end:
                        overlaps.append([overlap_start, overlap_end])

                print(f"Overlaps: {overlaps}")

                # Count collocation points within each overlap
                overlap_collocation_counts = []
                for overlap in overlaps:
                    count = np.sum((collocation_points >= overlap[0]) & (collocation_points <= overlap[1]))
                    overlap_collocation_counts.append(count)

                print(f"Collocation points in each overlap: {overlap_collocation_counts}")
                
                # add rows into col_df using name, tl, wo, wi, overlap_collocation_counts
                new_row = pd.DataFrame({'model_type': [name], 'time_window': [tl], 'wo': [wo], 'wi': [wi], 'col_points': [overlap_collocation_counts[0] if nsub == 2 else overlap_collocation_counts]}) 
                col_df = pd.concat([col_df, new_row], ignore_index=True)

                if args.train[0]:
                        FBPINNrun = FBPINNTrainer(c)
                        FBPINNrun.train()
                        
                        # import model 
                        c_out, model = load_model(run, rootdir=rootdir+"/")

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

        # Plot
        fig = plt.figure(figsize=(12,14), dpi=300)
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 5, 5, 4])
        gs_lossplot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.3) 
        ax1 = fig.add_subplot(gs_lossplot[0, 0])
        ax2 = fig.add_subplot(gs_lossplot[0, 1])

        # Create Dataset for plotting
        df1 = pd.DataFrame(columns=['Time Limit[0,10]', 'Window Overlap', 'Window NoDataRegion','MSE Learned', 'MSE Test'])
        df2 = pd.DataFrame(columns=['Time Limit[10,24]', 'Window Overlap', 'Window NoDataRegion','MSE Learned', 'MSE Test'])
        for run in runs:
            c_out, model = load_model(run, rootdir=rootdir+"/")

            # run = f"FBPINN_{problem.__name__}_{tag}_{nsub}_nsub_{wo}_wo_{wi}_wi_{tl}_tl"
            tl = c_out.problem_init_kwargs['time_limit']
            tl_key = f"{tl[0]}-{tl[1]}"

            parts = c_out.run.split("_")
            wo_index = 5
            wi_index = 7
            wo = parts[wo_index]
            wi = parts[wi_index]

            ###################### plot loss###########################
            i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]

            axis_map = {'0-10': ax1, '10-24': ax2}
            ax = axis_map.get(tl_key)
            if ax is None:
                raise ValueError(f"Invalid time_limit key: {tl_key}")
            
            # Now plot on the determined axis
            ax.plot(i, l1n, label=f"{wo}-{wi}")
            ax.set_yscale('log')
            ax.set_title(f'Time Limit: {tl_key}[wo Vs. wi]')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Normalized l1 test loss')
            ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.2), 
                      loc='upper center', fontsize='small')
            #################################

            u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
            mse_test = np.mean((u_exact - u_test)**2)
            mse_learned = np.mean((u_exact - u_learned)**2)

            if tl_key=="0-10":
                new_row = pd.DataFrame({'Time Limit[0,10]': [tl_key], 'Window Overlap': [wo], 
                                    'Window NoDataRegion' :[wi], 'MSE Learned': [mse_learned], 
                                    'MSE Test': [mse_test]})  
                df1 = pd.concat([df1, new_row], ignore_index=True)
            else:
                new_row = pd.DataFrame({'Time Limit[10,24]': [tl_key], 'Window Overlap': [wo], 
                                    'Window NoDataRegion' :[wi], 'MSE Learned': [mse_learned], 
                                    'MSE Test': [mse_test]})  
                df2 = pd.concat([df2, new_row], ignore_index=True)


        df1['MSE Learned'] = pd.to_numeric(df1['MSE Learned'], errors='coerce')
        df1['MSE Test'] = pd.to_numeric(df1['MSE Test'], errors='coerce')
        df2['MSE Learned'] = pd.to_numeric(df2['MSE Learned'], errors='coerce')
        df2['MSE Test'] = pd.to_numeric(df2['MSE Test'], errors='coerce')

        # [0,10]
        pivot_learned1 = df1.pivot(index="Window Overlap", columns="Window NoDataRegion", values="MSE Learned")
        pivot_test1 = df1.pivot(index="Window Overlap", columns="Window NoDataRegion", values="MSE Test")
        pivot_learned_log1 = pivot_learned1.map(lambda x: np.log10(x + 1e-10))  # Adding a small number to avoid log(0)
        pivot_test_log1 = pivot_test1.map(lambda x: np.log10(x + 1e-10))

        # [10,24]
        pivot_learned2 = df2.pivot(index="Window Overlap", columns="Window NoDataRegion", values="MSE Learned")
        pivot_test2 = df2.pivot(index="Window Overlap", columns="Window NoDataRegion", values="MSE Test")
        pivot_learned_log2 = pivot_learned2.map(lambda x: np.log10(x + 1e-10))  # Adding a small number to avoid log(0)
        pivot_test_log2 = pivot_test2.map(lambda x: np.log10(x + 1e-10))

        ax0 = fig.add_subplot(gs[0, 0])
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
                    f"• lr: {c_out.optimiser_kwargs["learning_rate"]} " +
                    f"• Noise: {noise_level} " + "\n"
                    f"• DD: Nonuniform " +
                    f"• nsub: {nsub} " +
                    f"• Problem: {problem.__name__ if hasattr(problem, '__name__') else problem}({name})") 

        ax0.text(0.5, 0.5, params_text, ha='center', va='center', fontsize=12)
        ax0.set_frame_on(False)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        # highlight_text = r"$\mathbf{MAX}$"
        # ax0.text(0.47, -0.2, highlight_text, ha='center', va='center', fontsize=12,
        #         bbox=dict(facecolor='red', alpha=0.5))

        # heatmaps for 0-10 and 10-24
        gs_heatmaps1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3) 
        gs_heatmaps2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.3) 

        ax4 = fig.add_subplot(gs_heatmaps1[0, 0])
        ax5 = fig.add_subplot(gs_heatmaps1[0, 1])

        ax6 = fig.add_subplot(gs_heatmaps2[0, 0])
        ax7 = fig.add_subplot(gs_heatmaps2[0, 1])

        def highlight_max(data, ax, highlight_color='red'):
            for col in data.columns:
                max_row = data[col].idxmax()
                ax.add_patch(plt.Rectangle((data.columns.tolist().index(col), data.index.tolist().index(max_row)),
                                            1, 1, fill=True, facecolor=highlight_color, alpha=0.5, edgecolor=highlight_color, lw=2))

        # Heatmap for MSE  Learned
        # [0,10]
        sns.heatmap(pivot_learned_log1, annot=True, fmt=".2f", cmap='viridis', ax=ax4)
        # highlight_max(pivot_learned_log1, ax4)
        ax4.set_title('Log MSE Learned[0,10]')
        ax4.invert_yaxis()
        # [10,24]
        sns.heatmap(pivot_learned_log2, annot=True, fmt=".2f", cmap='viridis', ax=ax5)
        # highlight_max(pivot_learned_log2, ax5)
        ax5.set_title('Log MSE Learned[10,24]')
        ax5.invert_yaxis()

        # Heatmap for MSE  Test
        # [0,10]
        sns.heatmap(pivot_test_log1, annot=True, fmt=".2f", cmap='viridis', ax=ax6)
        # highlight_max(pivot_test_log1, ax6)
        ax6.set_title('Log MSE Test[0,10]')
        ax6.invert_yaxis()
    
        #[10,24]
        sns.heatmap(pivot_test_log2, annot=True, fmt=".2f", cmap='viridis', ax=ax7)
        # highlight_max(pivot_test_log2, ax7)
        ax7.set_title('Log MSE Test[10,24]')
        ax7.invert_yaxis()

        plt.suptitle('MSE Value by WO Vs. WI', fontsize=14, verticalalignment='top')#, y=0.95)
        plt.subplots_adjust(hspace=0.2, top=0.88)
        plt.tight_layout()
        file_path = f"{parentdir}/MSE_WoVsWi_({name}).png"
        plt.savefig(file_path)
        print("DONE")

    # export col_df
    col_df.to_csv(f"{parentdir}/col_df.csv", index=False)


if __name__=="__main__":
    plot_DDD_varying_overlap()