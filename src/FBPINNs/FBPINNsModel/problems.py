import sys
import os
from scipy.integrate import odeint
from scipy.interpolate import interp1d

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import jax
import jax.numpy as jnp
from fbpinns.problems import Problem
import matplotlib.pyplot as plt


class SaturatedGrowthModel(Problem):
    """
    u' = u(C-u)
    I.C.
    u(0) = u_0 = 0.01
    We have to pass: "C_ture":C, "u_0":u0,"sd":sd,
            "time_limit":time_limit, "numx":numx,
    """

    @staticmethod
    def init_params(C=1, u0=0.01, sd=0.1, 
                    time_limit=[0, 24], numx=50,
                    lambda_phy=1e0, lambda_data=1e0,
                    sparse=False, noise_level=0.0):
        
        static_params = {
            "dims":(1,1),
            "C_true":C,
            "u_0":u0,
            "sd":sd,
            "time_limit":time_limit,
            "numx":numx,
            "lambda_phy": lambda_phy,
            "lambda_data": lambda_data,
            "sparse":sparse,
            "noise_level":noise_level,
        }
        trainable_params = {
            "C":jnp.array(0.5), # learn C from constraints
        }
        
        return static_params, trainable_params
    
    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        u0 = all_params["static"]["problem"]["u_0"]
        C = all_params["static"]["problem"]["C_true"]

        exp = jnp.exp(-C*x_batch[:,0:1])
        u = C / (1 + ((C - u0) / u0) * exp)
        return u
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),
            (0, (0,)),
        )
        time_limit = all_params["static"]["problem"]["time_limit"]
        numx = all_params["static"]["problem"]["numx"]
        # Data Loss
        if all_params['static']['problem']['sparse']:
            x_batch_data = jnp.sort(jax.random.uniform(key=key, shape=(numx,1), minval=time_limit[0], maxval=time_limit[1]), axis=0)
        else:
            x_batch_data = jnp.linspace(time_limit[0],time_limit[1],numx).astype(float).reshape((numx,1)) 
        noise = jax.random.normal(key, shape=(numx,1))  * all_params['static']['problem']['noise_level']
        u_data = SaturatedGrowthModel.exact_solution(all_params, x_batch_data) + noise
        required_ujs_data = (
            (0, ()),
        )
        return [[x_batch_phys, required_ujs_phys],
                [x_batch_data, u_data, required_ujs_data]]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        u0 = all_params["static"]["problem"]["u_0"]

        x, tanh = x_batch[:,0:1], jnp.tanh

        u = u0 + tanh(x/sd) * u
        
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        C = all_params["trainable"]["problem"]["C"]
        lambda_phy = all_params["static"]["problem"]["lambda_phy"]
        lambda_data = all_params["static"]["problem"]["lambda_data"]
        # physics loss
        _, u, ut = constraints[0]
        phys = lambda_phy*jnp.mean((ut - u*(C-u))**2)

        # data loss
        _, uc, u = constraints[1]
        data = lambda_data*jnp.mean((u-uc)**2)

        return phys + data
    
    @staticmethod
    def model(u, t, C):
        """Defines the ODE to be solved: du/dt = u * (C - u)."""
        return u * (C - u)
    
    @staticmethod
    def learned_solution(all_params, x_batch):
        """Solves the ODE for given initial conditions and  learned parameters."""
        # Extracting parameters and initial condition
        C = all_params['trainable']["problem"]["C"]
        u0 = all_params["static"]["problem"]["u_0"]  
              
        exp = jnp.exp(-C*x_batch)
        u = C / (1 + ((C - u0) / u0) * exp) # solution for C_learned
        return u.reshape(-1,1)

class CompetitionModel(Problem):

    @staticmethod
    def init_params(params=[0.5, 0.7, 0.3, 0.3, 0.6], 
                    u0=2, v0=1, sd=0.1, time_limit=[0,24], 
                    numx=50, lambda_phy=1e0, lambda_data=1e0, lambda_param=1e6,
                    sparse=False, noise_level=0.00):
        
        r, a1, a2, b1, b2 = params 
        static_params = {
            "dims":(2,1),   # dims of solution and problem
            "r_true":r,
            "a1_true":a1,
            "a2_true":a2,
            "b1_true":b1,
            "b2_true":b2,
            "u0":u0,
            "v0":v0,
            "sd":sd,
            "time_limit":time_limit,
            "numx":numx,
            "lambda_phy": lambda_phy,
            "lambda_data": lambda_data,
            "lambda_param": lambda_param,
            "sparse":sparse,
            "noise_level":noise_level,
        }
        trainable_params = {
            "r":jnp.array(0.5),
            "a1":jnp.array(0.5),
            "a2":jnp.array(0.5),
            "b1":jnp.array(0.5),
            "b2":jnp.array(0.5),
        }
        return static_params, trainable_params
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),  
            (1, ()),  
            (0, (0,)), 
            (1, (0,)),  
        )

        # Data Loss
        time_limit = all_params["static"]["problem"]["time_limit"]
        numx = all_params["static"]["problem"]["numx"]
        if all_params['static']['problem']['sparse']:
            x_batch_data = jnp.sort(jax.random.uniform(key=key, shape=(numx,1), minval=time_limit[0], maxval=time_limit[1]), axis=0)
        else:
            x_batch_data = jnp.linspace(time_limit[0],time_limit[1],numx).astype(float).reshape((numx,1)) 
        noise = jax.random.normal(key, shape=(numx,2))  * all_params['static']['problem']['noise_level']
        solution = CompetitionModel.exact_solution(all_params, x_batch_data)
        solution = solution + noise
        required_ujs_data = (
            (0, ()), 
            (1, ()),  
        )
        return [[x_batch_phys, required_ujs_phys],
                [x_batch_data, solution, required_ujs_data]]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, solution):
        sd = all_params["static"]["problem"]["sd"]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]

        x, tanh = x_batch[:,0:1], jnp.tanh

        u = solution[:, 0:1] * tanh(x/sd) + u0 # Hard constraining
        v = solution[:, 1:2] * tanh(x/sd)  + v0

        return jnp.concatenate([u, v], axis=1)
    
    @staticmethod
    def loss_fn(all_params, constraints):
        
        r, a1, a2, b1, b2 = [all_params['trainable']["problem"][key] for key in ('r', 'a1', 'a2', 'b1', 'b2')]
        lambda_phy = all_params["static"]["problem"]["lambda_phy"]
        lambda_data = all_params["static"]["problem"]["lambda_data"]
        # Physics loss
        _, u, v, ut, vt = constraints[0]
        phys1 = jnp.mean((ut - u + a1*u**2 + a2*u*v)**2)
        phys2 = jnp.mean((vt - r*v + r*b1*u*v + r*b2*v**2)**2)
        phys = lambda_phy*(phys1 + phys2)

        # Data Loss
        _, sol, u, v = constraints[1]
        ud = sol[:, 0:1]
        vd = sol[:, 1:2]
        data = lambda_data*(jnp.mean((u-ud)**2) + lambda_data*jnp.mean((v-vd)**2))

        # Penalty for negative parameters
        penalty_factor = all_params["static"]["problem"]["lambda_param"]
        penalty_terms = [r, a1, a2, b1, b2]
        penalties = sum(jnp.where(param < 0, penalty_factor * (param ** 2), 0) for param in penalty_terms) #TODO: Add small value to avoid zero penalty (Need to varify)
        
        return phys + data + penalties
    
    @staticmethod
    def model(y, t, params):
        """
        Compute the derivatives of the system at time t.
        
        :param y: Current state of the system [u, v].
        :param t: Current time.
        :param params: Parameters of the model (a1, a2, b1, b2, r).
        :return: Derivatives [du/dt, dv/dt].
        """
        u, v = y  
        r, a1, a2, b1, b2 = params  
        
        # Define the equations
        du_dt = u * (1 - a1 * u - a2 * v)
        dv_dt = r * v * (1 - b1 * u - b2 * v)
        
        return [du_dt, dv_dt]
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        r, a1, a2, b1, b2 = [all_params['static']["problem"][key] for key in ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [r, a1, a2, b1, b2]
        
        t = jnp.arange(0, 25.02, 0.02)  
        
        # Solve the system 
        solution = odeint(CompetitionModel.model, [u0, v0], t, args=(params,))
        
        # Interpolation 
        u_interp = interp1d(t, solution[:, 0], kind='cubic')
        v_interp = interp1d(t, solution[:, 1], kind='cubic')
        
        u_data = u_interp(x_batch.flatten())
        v_data = v_interp(x_batch.flatten())
        
        # Combine 
        combined_solution = jnp.vstack((u_data, v_data)).T
        if batch_shape:
            combined_solution = combined_solution.reshape(batch_shape + (2,))
        
        return combined_solution
    
    @staticmethod
    def learned_solution(all_params, x_batch):
        r, a1, a2, b1, b2 = [all_params['trainable']["problem"][key] for key in ('r', 'a1', 'a2', 'b1', 'b2')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [r, a1, a2, b1, b2]

        solution = odeint(CompetitionModel.model, [u0, v0], x_batch, args=(params,))

        return solution
    

if __name__ == "__main__":
    from fbpinns.domains import RectangularDomainND
    import numpy as np

    np.random.seed(0)  

    ps_ = CompetitionModel.init_params()
    problem_static, problem_trainable = ps_

    domain = RectangularDomainND
    xmin, xmax = jnp.array([0,]), jnp.array([24,])
    domain_static, domain_trainable = domain.init_params(xmin, xmax)
    all_params = {"static": {"problem": problem_static, "domain": domain_static}, 
                  "trainable": {"problem": problem_trainable, "domain": domain_trainable}}

    print(all_params)
    # set jax random key to seed 0
    key = jax.random.PRNGKey(0)

    batch_shapes = ((200,),)
    _, sol, ujs = CompetitionModel.sample_constraints(all_params, domain, key, 'grid', batch_shapes)[1]
    ud = sol[:, 0:1]
    vd = sol[:, 1:2]   
    print(ud.shape, vd.shape, ujs)

    x_batch = np.linspace(0, 24, 200).reshape(-1, 1)
    sol = CompetitionModel.exact_solution(all_params, x_batch)
    print(sol.shape)
    
    t = np.linspace(0, 24, 50)
    # plot u and v over time
    plt.figure()
    plt.scatter(t, ud, label='u')
    plt.scatter(t, vd, label='v')
    plt.show()


    

    
