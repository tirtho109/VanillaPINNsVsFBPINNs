import subprocess
from FBPINNs_CompetitionModel import get_parser
import pandas as pd
import matplotlib.pyplot as plt

parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '10', '24', "--rootdir", "CSsubdomain_test_10to24", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [10, 24] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
rootdir = "CSsubdomain_test_10to24"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{defaults["model_type"][0]}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)
# plt.show()

################ time_limit=[0, 24]###################################################################
parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '0', '24', "--rootdir", "CSsubdomain_test_0to24", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [0, 24] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
rootdir = "CSsubdomain_test_0to24"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{defaults["model_type"][0]}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)
# plt.show()

################ time_limit=[0, 10]###################################################################
parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '0', '10', "--rootdir", "CSsubdomain_test_0to10", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [0, 10] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
rootdir = "CSsubdomain_test_0to10"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{defaults["model_type"][0]}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)


##########################################################
############ For Co-existence Model ######################
##########################################################
parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '10', '24', "--rootdir", "CCsubdomain_test_10to24", "--model_type", "coexistence", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [10, 24] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
model_type = "coexistence"
rootdir = "CCsubdomain_test_10to24"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{model_type}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)
# plt.show()

################ time_limit=[0, 24]###################################################################
parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '0', '24', "--rootdir", "CCsubdomain_test_0to24", "--model_type", "coexistence", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [0, 24] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
model_type = "coexistence"
rootdir = "CCsubdomain_test_0to24"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{model_type}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)
# plt.show()

################ time_limit=[0, 10]###################################################################
parser = get_parser()
defaults = {action.dest: action.default for action in parser._actions}
print(defaults)
print(type(defaults['tag']))
base_command = ['python', 'FBPINNs_CompetitionModel.py', '-tl', '0', '10', "--rootdir", "CCsubdomain_test_0to10", "--model_type", "coexistence", "--sparse", "True", "-nl", "0.05"]
defaults["time_limit"] = [0, 10] 
defaults["sparse"] = [True]
defaults["noise_level"]=0.05
processes = []
runs = []
problem_name = "CompetitionModel"
network_name = "FCN"
model_type = "coexistence"
rootdir = "CCsubdomain_test_0to10"
h = len(defaults["layers"]) - 2  # Number of hidden layers
p = sum(defaults["layers"][1:-1]) 
num_subdomains = range(2,11)

for num_subdomain in num_subdomains:  
    command = base_command + ['--num_subdomain', str(num_subdomain)]
    run = f"FBPINN_{defaults['tag']}_{problem_name}_{model_type}_{network_name}_{num_subdomain}-ns_{defaults['window_overlap']}-ol_{h}-l_{p}-h_{defaults["num_collocation"]}-nC_"
    run += f"{defaults["epochs"]}-e_{defaults["numx"]}-nD_{defaults["time_limit"]}-tl_{defaults["num_test"]}-nT_{defaults["initial_conditions"]}-ic_{defaults["sparse"]}-sp_{defaults["noise_level"]}-nl_"
    runs.append(run)
    process = subprocess.Popen(command)
    processes.append(process)

for process in processes:
    process.wait()

mse_test = []
mse_learned = []

for run, num_subdomain in zip(runs,num_subdomains):
    path = f"{rootdir}/summaries/{run}/metrices.csv"
    metrices = pd.read_csv(path)  
    mse_test.append(metrices[metrices['Metric'] == 'MSE']['Test'].values[0])
    mse_learned.append(metrices[metrices['Metric'] == 'MSE']['Learned'].values[0])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_subdomains, mse_test, 'o-', label="Test", marker='o')
plt.plot(num_subdomains, mse_learned, 'x-', label="Learned", marker='x')
plt.xlabel('# of Subdomains')
plt.ylabel('MSE')
plt.title('MSE over Subdomains')
plt.legend()
plt.grid(True)
plt.yscale('log')
save_path = f"{rootdir}/summaries/MSEvsSubdomains.png"
plt.savefig(save_path)