from fbpinns.trainers import _common_train_initialisation
from fbpinns.trainers import FBPINN_loss
from jax import random
import jax.numpy as jnp
import numpy as np
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import load_model
from copy import deepcopy

"""
The idea of Loss Landscape is inspired by the following paper: https://arxiv.org/pdf/1712.09913
And the coding strategy is inspired by the following code: https://github.com/sciann/sciann/blob/master/sciann/utils/callbacks.py
"""

def get_loss(fbpinn_trainer, c, all_params, active, all_opt_states, i):
    key = random.PRNGKey(c.seed)
    domain, problem, decomposition, network = c.domain, c.problem, c.decomposition, c.network
    (_, all_opt_states, _, loss_fn, key, constraints_global, x_batch_global, 
        constraint_offsets_global, constraint_fs_global, 
        jmapss, _, _) = _common_train_initialisation(c, key, all_params, problem, domain)
    
    (active, _, _, 
        active_params, fixed_params, static_params, takess, 
        constraints, _) = fbpinn_trainer._get_update_inputs(i, active , all_params, all_opt_states ,
                                                        x_batch_global, constraints_global, constraint_fs_global, 
                                                        constraint_offsets_global, decomposition, problem)                          
    # required_ujss = [constraint_[-1] for constraint_ in constraints_global]
    model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn, problem.constraining_fn)
    lossval = FBPINN_loss(active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn)
    return lossval

def _collect_weights(all_params):
    network_params = all_params["trainable"]["network"]["subdomain"]["layers"]
    problem_params = all_params["trainable"]["problem"]
    x_values = []
    for w, b in network_params:
        for subdomain_weights in w:
            flattened_weights = jnp.array(subdomain_weights).flatten()
            x_values.append(jnp.array(flattened_weights))
        for subdomain_biases in b:
            flattened_biases = jnp.array(subdomain_biases).flatten()
            x_values.append(jnp.array(flattened_biases))
    # param_values = jnp.array([float(problem_params[key]) for key in problem_params])
    # x_values.append(jnp.array(param_values))
    return x_values

def _update_weights(all_params, x_values):
    idx = 0
    for layer_idx, layer_params in enumerate(all_params["trainable"]["network"]["subdomain"]["layers"]):
        w, b = layer_params 
        w_shapes = [subdomain_w.shape for subdomain_w in w]
        b_shapes = [subdomain_b.shape for subdomain_b in b]
        new_w_subdomain = []
        for shape in w_shapes:
            subdomain_size = jnp.prod(jnp.array(shape))
            new_w_flat = x_values[idx][:subdomain_size].reshape(shape)
            new_w_subdomain.append(new_w_flat)
            idx += 1
        new_w =jnp.stack(new_w_subdomain, axis=0)

        new_b_subdomains = []
        for shape in b_shapes:
            subdomain_size = jnp.prod(jnp.array(shape))
            new_b_flat = x_values[idx][:subdomain_size].reshape(shape)
            new_b_subdomains.append(new_b_flat)
            idx += 1
        new_b = jnp.stack(new_b_subdomains, axis=0)

        all_params["trainable"]["network"]["subdomain"]["layers"][layer_idx] = (new_w, new_b)

        # # set problem_params
        # problem_keys = list(all_params["trainable"]["problem"].keys())
        # for i, key in enumerate(problem_keys):
        #     all_params["trainable"]["problem"][key] = jnp.array(x_values[idx][i], dtype=jnp.float32)
        return all_params
    

def on_epoch_end(fbpinn_trainer, c, all_params, active, all_opt_states, i, _weight_norm, _loss_value, logs={}):
    x_trained = _collect_weights(all_params)
    norm = 2
    _norm = lambda xs: np.linalg.norm(xs, norm)
    _weight_norm.append(_norm(np.concatenate(x_trained)))
    _loss_value.append(get_loss(fbpinn_trainer, c, all_params, active, all_opt_states, i))
    logs['norm-loss-weights'] = _weight_norm[-1]



def on_train_end(fbpinn_trainer, c, all_params, active, all_opt_states, i,
                 _weight_norm, _resolution, path, logs={}, delta_weights=None):
    norm = 2
    _norm = lambda xs: np.linalg.norm(xs, norm)

    x_trained = _collect_weights(all_params)
    x_sizes = [x.size for x in x_trained]
    num_param = sum(x_sizes)
    n0 = np.split(np.random.standard_normal(num_param), np.cumsum(x_sizes))
    n1 = np.split(np.random.standard_normal(num_param), np.cumsum(x_sizes))
    
    n0_norm, n1_norm = [_norm(np.concatenate(ni)) for ni in [n0, n1]]
    n0 = [ni/n0_norm for ni in n0]
    n1 = [ni/n1_norm for ni in n1]

    #TODO check this
    if delta_weights is None:
        delta_weights = 2.0 * abs(_weight_norm[-1] - _weight_norm[0])
    else:
        delta_weights = delta_weights

    print("delta_weights",delta_weights)
    loss_values = np.zeros((_resolution**2, 3))

    k = 0
    for i, l0 in enumerate(np.linspace(-delta_weights, delta_weights, _resolution)):
        for j, l1 in enumerate(np.linspace(-delta_weights, delta_weights, _resolution)):
            test_weights = [xi + n0i*l0 + n1i*l1 for xi,n0i,n1i in zip(x_trained, n0, n1)]
            _update_weights(all_params, test_weights)
            loss_values[k, :] = [l0, l1, get_loss(fbpinn_trainer, c, all_params, active, all_opt_states, i)]
            print(f"loss values at is {loss_values[k, :]}")
            k += 1
            # save the calculations
            np.savetxt(
                path + "loss-landscape.csv",
                loss_values,
                delimiter=','
            )
    _update_weights(all_params, x_trained)

def _loss_lanscape(fbpinn_trainer, run, rootdir, resolution = 20, norm=2, path=None, delta_weights=None):
    _norm = lambda xs: np.linalg.norm(xs, ord=norm)
    # initial model, weight_norm & loss
    c_0, model_0 = load_model(run=run, i=0, rootdir=rootdir)
    i_0, all_params_0, active_0, all_opt_states_0 = model_0[0], model_0[1], model_0[3], model_0[2]
    _weight_norm = [_norm(np.concatenate(_collect_weights(all_params_0)))]
    _loss_value  = [get_loss(fbpinn_trainer, c_0, all_params_0, active_0, all_opt_states_0, i_0)]

    # final model
    c, model = load_model(run=run, rootdir=rootdir)
    i, all_params, active, all_opt_states = model[0], model[1], model[3], model[2]
    copied_params = deepcopy(all_params)
    on_epoch_end(fbpinn_trainer, c, copied_params, active, all_opt_states, i, _weight_norm, _loss_value, logs={})

    # export loss_lanscape
    on_train_end(fbpinn_trainer, c, copied_params, active, all_opt_states, i, _weight_norm, resolution, path, delta_weights=delta_weights)



