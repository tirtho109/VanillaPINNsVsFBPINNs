# FBPINNs Implementation

## Steps for FBPINN Approach

To implement the FBPINN approach, follow these steps:

### 1. Define Domain
Specify the temporal boundaries of the problem:
- $t_{\text{min}} = t_{\text{begin}}$
- $t_{\text{max}} = t_{\text{end}}$

### 2. Define Problem
Initialize problem parameters, physical and data constraints, constraining operator, loss function, and exact solution (if available).

#### Parameter Initialization

- **Saturated Growth Model:** Set parameters such as `dims`, $C_{\text{true}}$, \(u_0\), `sd`, `time_limit`, `sparse`, `numx`, $\lambda_{\text{phy}}$, $\lambda_{\text{data}}$, `noise_level`, and the trainable parameter \(C\).
- **Competition Model:** Set parameters such as `dims`, $r_{\text{true}}$, $a_{1_{\text{true}}}$, $a_{2_{\text{true}}}$, $b_{1_{\text{true}}}$, $b_{2_{\text{true}}}$, $u_0$, $v_0$, `sd`, `time_limit`, `sparse`, `numx`, $\lambda_{\text{phy}}$, $\lambda_{\text{data}}$, $\lambda_{\text{params}}$, `noise_level`, and trainable parameters $r$, $a_1$, $a_2$, $b_1$, $b_2$.


#### Physics and Data Constraints
- Generate collocation points $t_{\text{batch phy}}$.
- Set derivatives (0th and 1st order) of the network outputs with respect to collocation points.
- Create training data points $t_{\text{batch data}}$ with noisy solutions.

#### Constraining Operator
- **Saturated Growth Model:** Update network output to satisfy initial condition: $\tilde{u} = u \cdot \tanh(t/\text{sd}) + u_0$.
- **Competition Model:** Update network outputs to satisfy initial conditions: $\tilde{u} = u \cdot \tanh(t/\text{sd}) + u_0$, $\tilde{v} = v \cdot \tanh(t/\text{sd}) + v_0$.

#### Loss Functions
- **Saturated Growth Model:** Compute physics loss $\lambda_{\text{phy}} \cdot \text{MSE}(\tilde{u}' - \tilde{u}(C - \tilde{u}))$ and data loss $\lambda_{\text{data}} \cdot \text{MSE}(\tilde{u} - u_{\text{true}})$.
- **Competition Model:** Compute physics losses:
  - $\lambda_{\text{phy}} \cdot \text{MSE}\left( \tilde{u}' - \tilde{u} + a_1 \tilde{u}^2 + a_2 \tilde{u} \tilde{v} \right)$
  - $\lambda_{\text{phy}} \cdot \text{MSE}\left( \tilde{v}' - r \tilde{v} + r b_1 \tilde{u} \tilde{v} + r b_2 \tilde{v}^2 \right)$
  and data losses:
  - $\lambda_{\text{data}} \cdot \text{MSE}(\tilde{u} - u_{\text{true}})$
  - $\lambda_{\text{data}} \cdot \text{MSE}(\tilde{v} - v_{\text{true}})$

  In addition, compute a parameter penalty:
  - $\lambda_{\text{params}} \times (\text{param})^2, \quad \text{if } \text{param} \leq 0$

  Finally, sum all loss terms for the final loss.


#### Exact Solution
- **Saturated Growth Model:** Exact solution: $u^{\text{exact}} = \frac{C}{1 + \left( \frac{C - u_0}{u_0} \right) \cdot e^{-Ct}}$.
- **Competition Model:** Approximate solution using `odeint` from `scipy.integrate`.

### 3. Domain Decomposition

Apply domain decomposition strategy:

- **Scenario 1:** Data from the entire time domain. Divide the time domain \(t\) by `nsub`. Calculate the center of each subdomain as `subdomain_xs`, then determine each subdomain width with the window overlap `wo` to calculate `subdomain_ws`.

- **Scenario 2:** Data from a part of the time domain. Separate into `Data region` and `No Data region`. Assign one subdomain for the `No Data region`, calculate its center, and determine the subdomain width using the window overlap `wi`. For the `Data region`, divide the domain into \((nsub-1)\) subdomains, calculate the center of each subdomain, and determine their widths using the window overlap `wo`.


### 4. Define Subdomain Neural Networks
For each subdomain, define a fully connected neural network (FCN) with appropriate output neurons.

### 5. Define Model Constants
Create a `Constants` object containing all problem information, domain decomposition, network settings, collocation points, test points, epochs, callback arguments, etc.

### 6. Train the Model
Train the networks using the `FBPINNTrainer` or `PINNTrainer` class, passing the `Constants` instance.

### 7. Postprocessing
- Plot loss landscape and model comparisons.
- Export MSE and MAE.
- Plot normalized $\ell_1$ test loss and energy using the Lyapunov function.

---

We use `t` and `x` interchangeably in the code, where `t` is the network input parameter, but `x` is used more generally to represent input features.
