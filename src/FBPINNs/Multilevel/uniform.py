import sys
import os
from itertools import product
"""
python uniform.py --train True
python nonuniform.py --train True
"""
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
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
                    file_path = os.path.join(run_config.summary_out_dir, "loss_landscape_viz.png")
                    plt.savefig(file_path, bbox_inches='tight') 

            elif run_type == "PINN":
                run = PINNTrainer(run_config)
                all_params = run.train()
            else:
                raise ValueError(f"Unknown run type: {run_type}")
    if args.plot:
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
                file_path = os.path.join(run_config.summary_out_dir, "matrices.csv")
                export_mse_mae(u_exact, u_test, u_learned, file_path)

                # 2. N-l1 test loss vs training steps
                fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                i, t, l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
                p = run_config.network_init_kwargs["layer_sizes"][1]
                h = len(run_config.network_init_kwargs["layer_sizes"]) -2 
                level = len(run_config.decomposition_init_kwargs["subdomain_xss"])
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
time_limits = [[0,10], [10, 24]]
hs = [3, 5]
ls = [2]
ws = [1.9]
ps = [5, 10, 20]
p0 = [2, 1.9, 5]

parentdir = ""
## Test 1: simple SG model
tag = "SG"
parentdir = "MLUniform/SGMultilevel"

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
            subdomain_xss = [[np.array([12.]),]] + [[np.linspace(0,24,n_),] for n_ in l[1:]]
            subdomain_wss = [[np.array([w*24.]),]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
            layer_sizes = [1,] + [p,]*h + [1,]
            runs.append(run_FBPINN(tag, rootdir))

####################################################################################################################
############################################### Test 2: Competition Model(survival)################################
####################################################################################################################
tag = "Surv"

parentdir = "MLUniform/SurvMultilevel"
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
            subdomain_xss = [[np.array([12.]),]] +  [[np.linspace(0,24,n_),] for n_ in l[1:]]
            subdomain_wss = [[np.array([w*24.]),]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
            layer_sizes = [1,] + [p,]*h + [2,]
            runs.append(run_FBPINN(tag, rootdir))

####################################################################################################################
############################################### # Test 3: Competition Model(coexistence) ################################
####################################################################################################################

tag = "Coex"
parentdir = "MLUniform/CoexMultilevel"

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
            subdomain_xss = [[np.array([12.]),]] +  [[np.linspace(0,24,n_),] for n_ in l[1:]]
            subdomain_wss = [[np.array([w*24.]),]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
            layer_sizes = [1,] + [p,]*h + [2,]
            runs.append(run_FBPINN(tag, rootdir))

if __name__ == "__main__":

        # parallel submit all runs
        # submit is modified for running locally
        submit(runs)
        print("All runs submitted.")