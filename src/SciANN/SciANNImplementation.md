### SciANN Implementation

To implement the models using SciANN, follow these steps:

- **Set Variable:** Define `t` as a `Variable`, which serves as input to the neural network (NN). For both models, `t` is considered as the input to the network.

- **Set Functional:** Establish $u^{\text{PINN}}$ for the Saturated Growth model, and $u^{\text{PINN}}$ and $v^{\text{PINN}}$ for the Competition model as `Functional`, representing the NN output.

- **Constraining Operator:** Apply a constraining operator to the network output to relate with the differential equation defining the model.
  - **Saturated Growth Model:** $\tilde{u}^{\text{PINN}} = u_0 + \tanh(t/0.1) \cdot u^{\text{PINN}}$
  - **Competition Model:** $\tilde{u}^{\text{PINN}} = u_0 + \tanh(t/0.1) \cdot u^{\text{PINN}}$ and $\tilde{v}^{\text{PINN}} = v_0 + \tanh(t/0.1) \cdot v^{\text{PINN}}$

- **Set Trainable Parameter:** Define `C` as a `Parameter` for the Saturated Growth model, representing trainable parameters other than network weights and bias ($\boldsymbol{\Theta}$). For the Competition model, set $r$, $a_1$, $a_2$, $b_1$, and $b_2$ as trainable parameters.

- **Set Gradients:** Set the gradients required for the ODE/PDE residual.
  - **Saturated Growth Model:** $\dot{\tilde{u}}^{\text{PINN}} = \frac{d\tilde{u}^{\text{PINN}}}{dt}$
  - **Competition Model:** $\dot{\tilde{u}}^{\text{PINN}} = \frac{d\tilde{u}^{\text{PINN}}}{dt}$ and $\dot{\tilde{v}}^{\text{PINN}} = \frac{d\tilde{v}}{dt}$

- **Assign Data Constraints:** Specify the data constraints that the model must adhere to during training, which are the constrained network outputs of the specific model.
  - **Saturated Growth Model:** $\tilde{u}^{\text{PINN}}$
  - **Competition Model:** $\tilde{u}^{\text{PINN}}$ and $\tilde{v}^{\text{PINN}}$

  Use the prior noisy solution of the problem as a label for the input, and apply data constraints using these noisy solutions.

- **Assign ODE Constraints:** Impose constraints by using the `Tie` constraint class from the `SciANN` package to the differential equations, ensuring model fidelity.
  - **Saturated Growth Model:** $(\dot{\tilde{u}}^{\text{PINN}}, \tilde{u}^{\text{PINN}} \cdot (C - \tilde{u}^{\text{PINN}}))$
  - **Competition Model:** $(\dot{\tilde{u}}^{\text{PINN}}, \tilde{u}^{\text{PINN}} \cdot (1 - a_1 \cdot \tilde{u}^{\text{PINN}} - a_2 \cdot \tilde{v}^{\text{PINN}}))$ and $(\dot{\tilde{v}}^{\text{PINN}}, r \cdot \tilde{v}^{\text{PINN}} \cdot (1 - b_1 \cdot \tilde{u}^{\text{PINN}} - b_2 \cdot \tilde{v}^{\text{PINN}}))$

- **Create CustomSciModel:** Create subclass `CustomSciModel` from the `SciModel` class to customize loss weights for various terms in the loss function.
  - **Saturated Growth Model:** Two custom loss weights: $\lambda_{\text{phy}}$ and $\lambda_{\text{data}}$ for the ODE loss and data loss, respectively.
  - **Competition Model:** Four custom loss weights: $\lambda_{\text{phy1}}$, $\lambda_{\text{phy2}}$, $\lambda_{\text{data1}}$, and $\lambda_{\text{data2}}$. $1,2$ represents $u,v$. For simplicity, we use $\lambda_{\text{phy}}$ and $\lambda_{\text{data}}$ as loss weights, keeping the weights the same for all ODEs and data.

- **Setup the Model:** Initialize the model with inputs, targets, and loss functions, and configure training parameters. Both models have `t` as input.
  - **Saturated Growth Model:** One ODE constraint and one data constraint.
  - **Competition Model:** Two ODE constraints and two data constraints.

  Set mean squared error (MSE) as the loss function for all constraints.

- **Generate Training Data:** Create synthetic supervised training data reflecting the expected behavior of the model under study.

- **Generate Collocation Points:** Establish collocation points for evaluating the differential equation constraints. Create structured grid temporal points for both models.

- **Assign IDs to Data Points:** Concatenate and sort collocation points and data points. Uniquely label each data point by assigning an ID to manage and track during training.

- **Set Constraint Values for Targets:** Define each target's specific constraint expected values. For the ODE constraints, set it to $0$, as the left-hand side and right-hand side of the ODE have to be equal. Set the synthetically generated data as the target data for the data constraints.

- **Train the Model:** Conduct the training. Pass model-specific inputs, targets, epochs, batch size, optimizer, learning rate, etc. Optionally, pass any callback arguments (e.g., loss landscape) to investigate different objectives.

- **Postprocessing:** Handle outputs after training, such as saving model weights, evaluating model performance, plotting, etc.
