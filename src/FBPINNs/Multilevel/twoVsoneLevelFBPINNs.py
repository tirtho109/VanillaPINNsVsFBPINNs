import sys
import os
from itertools import product

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import argparse
from fbpinns.domains import RectangularDomainND
from FBPINNsModel.problems import CompetitionModel, SaturatedGrowthModel
from fbpinns.decompositions import MultilevelRectangularDecompositionND, RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.analysis import load_model
import matplotlib.pyplot as plt
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters, export_energy_plot, plot_loss_landscape3D, plot_loss_landscape_contour
from FBPINNsModel.subdomain_helper import get_subdomain_xsws
from FBPINNsModel.FBPINNs_loss_landscape import _loss_lanscape


parser = argparse.ArgumentParser(description='''
        FBPINN code for Separating longtime behavior and learning of mechanics  \n
        To find the best number of subdomain for different time limit cases'''
)

parser.add_argument('--train', help='To train the whole model (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-ll', '--loss_landscape', help='To plot loss landscape (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='To plot (default True)', action='store_false')


args = parser.parse_args()


def submit(runs):
    # for run in run print(run)
    for run_config, run_type, rootdir in runs:
        print(run_config.run)
    if args.train[0]:
        for run_config, run_type, rootdir in runs:
            if run_type == "FBPINN":
                FBPINNrun = FBPINNTrainer(run_config)
                FBPINNrun.train()
                if args.loss_landscape[0]:
                    # export loss landscape
                    _loss_lanscape(FBPINNrun, run_config.run, rootdir+"/", 20, 2, run_config.summary_out_dir)
                    # plot loss landscape
                    csv_path = run_config.summary_out_dir + "/loss-landscape.csv"
                    fig = plt.figure(figsize=(12, 4), dpi=300) 
                    ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
                    ax2 = fig.add_subplot(1, 2, 2)
                    plot_loss_landscape3D(csv_path, ax1)
                    plot_loss_landscape_contour(csv_path, ax2)
                    fig.suptitle("Loss Landscape", fontsize=16, fontweight='bold')
                    plt.subplots_adjust(wspace=0.5) 
                    file_path = os.path.join(run_config.summary_out_dir, " _viz.png")
                    plt.savefig(file_path, bbox_inches='tight') 
            elif run_type == "PINN":
                run = PINNTrainer(run_config)
                all_params = run.train()
            else:
                raise ValueError(f"Unknown run type: {run_type}")
    if args.plot:
        df = pd.DataFrame(columns=["Time Limit", "Model Type", "MSE Learned", "MSE Test", "Level", "Hidden Layers and Units", "run", "rootdir"])
        for run_config, run_type, rootdir in runs:
            if run_type == "FBPINN":
                # 0. load model
                c_out, model = load_model(run_config.run,  rootdir=rootdir+"/")

                # 1. model comparision
                fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                plot_model_comparison(c_out, model, type="FBPINN", ax=ax)
                file_path = os.path.join(run_config.summary_out_dir, "model_comparison.png")
                plt.savefig(file_path)

                # Export parameters
                file_path = os.path.join(run_config.summary_out_dir, "parameters.csv")
                export_parameters(run_config, model, file_path)
                
                # Export MSE and MAE
                u_exact, u_test, u_learned = get_us(c_out, model, type=run_type)
                mse_test = np.mean((u_exact - u_test)**2)
                mse_learned = np.mean((u_exact - u_learned)**2)

                fifth_segment = run_config.run.split("_")[4]
                levels = fifth_segment[fifth_segment.find("[")+1 : fifth_segment.find("]")]
                level_count = len(levels.split(',')) 
                new_row = pd.DataFrame({
                                        "Time Limit": str(run_config.problem_init_kwargs["time_limit"]), 
                                        "Model Type": run_config.run.split("_")[1], 
                                        "MSE Learned": mse_learned, 
                                        "MSE Test": mse_test,
                                        "Level": level_count,
                                        "Hidden Layers and Units": str(run_config.run.split("_")[6].split("-")[0]) + "X" + str(run_config.run.split("_")[7].split("-")[0]),
                                        "run": run_config.run,
                                        "rootdir": rootdir
                                    }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)
                file_path = os.path.join(run_config.summary_out_dir, "matrices.csv")
                export_mse_mae(u_exact, u_test, u_learned, file_path)

                # 2. N-l1 test loss vs training steps
                fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                i, t, l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
                p = run_config.network_init_kwargs["layer_sizes"][1]
                h = len(run_config.network_init_kwargs["layer_sizes"]) -2 
                level = len(run_config.decomposition_init_kwargs["subdomain_xss"]) if run_config.decomposition.__name__ == "MultilevelRectangularDecompositionND" else 1
                ax.plot(i, l1n, label=f"{run_type}-{h}h-{p}u-{level}-l")
                ax.set_yscale("log") 
                ax.set_xlabel("Training steps")
                ax.set_ylabel("N-l1 test loss")
                ax.set_title('Loss vs Training Steps')
                ax.legend()
                file_path = os.path.join(run_config.summary_out_dir, "normalized_test_loss.png")
                plt.savefig(file_path)
                
                if run_config.problem.__name__ == "CompetitionModel":
                    model_type = "survival" if run_config.run.split("_")[1] == "Surv" else "coexistence"
                    # 3. Energy plot
                    file_path = os.path.join(run_config.summary_out_dir, "energy_plot.png")
                    export_energy_plot(c_out, model, model_type, file_path)
                
            # elif run_type == "PINN":
            #     run = PINNTrainer(run_config)
            #     run.plot()
            else:
                raise ValueError(f"Unknown run type: {run_type}")
        print(df.head(100))
        # export the df to csv
        file_path = ("SLvsML_simple/summary.csv")
        df.to_csv(file_path, index=False)


def pscan(*pss):
    "scan from fixed point"
    assert all(isinstance(ps, list) for ps in pss)
    return list(product(*pss))

def run_FBPINN(tag, rootdir):
    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{rootdir}/models/{self.run}/"
        
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-lev_{w}-ww_{h}-hl_{p}-hu_{n[0]}-n_{time_limit}-tl"
    c = ModifiedConstants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=MultilevelRectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xss=subdomain_xss,
                    subdomain_wss=subdomain_wss,
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=dict(
            layer_sizes=layer_sizes,
            ),
        n_steps=n_steps,
        ns=(n,),
        n_test=n_test,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        clear_output=True,
        save_figures=True,
        show_figures=False,
        )
    return c, "FBPINN", rootdir

def run_FBPINN_single_level(tag, rootdir):
    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{rootdir}/models/{self.run}/"
        
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_[1]-lev_{w}-ww_{h}-hl_{p}-hu_{n[0]}-n_{time_limit}-tl"
    c = ModifiedConstants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xs=subdomain_xs,
                    subdomain_ws=subdomain_ws,
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=dict(
            layer_sizes=layer_sizes,
            ),
        n_steps=n_steps,
        ns=(n,),
        n_test=n_test,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        clear_output=True,
        save_figures=True,
        show_figures=False,
        )
    return c, "FBPINN", rootdir


runs = []

# Fixed parameters
network = FCN
domain = RectangularDomainND
domain_init_kwargs = dict(xmin = np.array([0.]),
                          xmax = np.array([24.]),
                          )

C = 1
survival_params = [0.5, 0.3, 0.6, 0.7, 0.3]
coexistence_params = [0.5, 0.7, 0.3, 0.3, 0.6]

# training parameters
numx = 100
sparse = True
noise_level = 0.05
unnorm = (0, 1.)

# network parameters
test_freq = 1000
model_save_freq = 10000
n_steps = 50000
n = (200,)
n_test = (500,)

# variable parameters
time_limits = [[0,10], [10, 24], [0,24]]
hs = [3]
ls = [1, 2]
ws = [1.9]
ps = [5]
p0 = [2, 1.9, 5]
nsub = 2 # number of subdomains for single level test

parentdir = ""
## Test 1: simple SG model
tag = "SG"
parentdir = "twoVsoneLevelFBPINNs/SGMultilevel"

for time_limit in time_limits:
    for l_, w, p in pscan(ls, ws, ps):
        for h in hs:
            problem = SaturatedGrowthModel
            problem_init_kwargs = dict(C=1, u0=0.01, sd=0.1, 
                                time_limit=time_limit, numx=numx,
                                lambda_phy=1e0, lambda_data=1e0,
                                sparse=sparse, noise_level=noise_level)
            rootdir = f"{time_limit[0]}-{time_limit[1]}"
            rootdir = os.path.join(parentdir, rootdir)
            # multilevel scaling
            l = [2**i for i in range(l_)]
            subdomain_xss = [[np.array([12.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[0] for subs in l[1:]]
            subdomain_wss = [[np.array([24.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[1] for subs in l[1:]]
            layer_sizes = [1,] + [p,]*h + [1,]
            runs.append(run_FBPINN(tag, rootdir))
            # single level with 2 subdomain
            subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin=0, 
                                                    t_end=24, num_subdomain=nsub,
                                                    ww = w, w_noDataRegion=1.005)
            runs.append(run_FBPINN_single_level(tag, rootdir))



####################################################################################################################
############################################### Test 2: Competition Model(survival)################################
####################################################################################################################
tag = "Surv"

parentdir = "twoVsoneLevelFBPINNs/SurvMultilevel"
# Competition Model

for time_limit in time_limits:
    for l_, w, p in pscan(ls, ws, ps):
        for h in hs:
            problem = CompetitionModel
            problem_init_kwargs = dict(params=survival_params, u0=2, v0=1, sd=0.1, 
                                time_limit=time_limit, numx=numx,
                                lambda_phy=1e0, lambda_data=1e0,
                                sparse=sparse , noise_level=noise_level)
            rootdir = f"{time_limit[0]}-{time_limit[1]}"
            rootdir = os.path.join(parentdir, rootdir)
            # multilevel scaling
            l = [2**i for i in range(l_)]
            subdomain_xss = [[np.array([12.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[0] for subs in l[1:]]
            subdomain_wss = [[np.array([24.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[1] for subs in l[1:]]
            layer_sizes = [1,] + [p,]*h + [2,]
            runs.append(run_FBPINN(tag, rootdir))
            # single level with 2 subdomain
            subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin=0, 
                                                    t_end=24, num_subdomain=nsub,
                                                    ww = w, w_noDataRegion=1.005)
            runs.append(run_FBPINN_single_level(tag, rootdir))

####################################################################################################################
######################################### # Test 3: Competition Model(coexistence) #################################
####################################################################################################################

tag = "Coex"
parentdir = "twoVsoneLevelFBPINNs/CoexMultilevel"

for time_limit in time_limits:
    for l_, w, p in pscan(ls, ws, ps):
        for h in hs:
            problem = CompetitionModel
            problem_init_kwargs = dict(params=coexistence_params, u0=2, v0=1, sd=0.1, 
                                time_limit=time_limit, numx=numx,
                                lambda_phy=1e0, lambda_data=1e0,
                                sparse=sparse , noise_level=noise_level)
            rootdir = f"{time_limit[0]}-{time_limit[1]}"
            rootdir = os.path.join(parentdir, rootdir)
            # multilevel scaling
            l = [2**i for i in range(l_)]
            subdomain_xss = [[np.array([12.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[0] for subs in l[1:]]
            subdomain_wss = [[np.array([24.]),]] + [get_subdomain_xsws(time_limit=time_limit, t_begin=0, t_end=24, num_subdomain=subs, ww=w, w_noDataRegion=1.005)[1] for subs in l[1:]]
            layer_sizes = [1,] + [p,]*h + [2,]
            runs.append(run_FBPINN(tag, rootdir))
            # single level with 2 subdomain
            subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin=0, 
                                                    t_end=24, num_subdomain=nsub,
                                                    ww = w, w_noDataRegion=1.005)
            runs.append(run_FBPINN_single_level(tag, rootdir))

if __name__ == "__main__":

        # parallel submit all runs
        # submit is modified for running locally
        submit(runs)
        print("All runs submitted.")