# VanillaPINNsVsFBPINNs

## Title
Towards Model Discovery Using Domain Decomposition and PINNs.

## Introduction
Physics-Informed Neural Networks (PINNs) offer a powerful approach to solving differential equations by embedding physics constraints directly into neural network training. In this study, we compare the traditional PINN approach with domain decomposition-based FBPINNs to evaluate their performance in learning the dynamics of two ordinary differential equation (ODE) models:

- **Saturated Growth Model** 
  $$u' = u(C - u)$$ 
  with $u_0 >0$, and $C$ is a positive parameter.
  
- **Competition Model with coexistence and survival** 
  $$u' = u(1 - a_1u - a_2v)$$ 
  $$v' = rv(1 - b_1u - b_2v)$$
  with $u_0>0 \quad \text{and}\quad v_0>0$, and $r, a_1, a_2, b_1, b_2$ are all positive parameters. 

## Methods
- **[SciANN](https://github.com/ehsanhaghighat/sciann)** for Vanilla PINNs. [Link](https://github.com/tirtho109/VanillaPINNsVsFBPINNs/tree/main/src/SciANN)
- **[FBPINNs](https://github.com/benmoseley/FBPINNs)** for domain decomposition-based PINNs. [Link](https://github.com/tirtho109/VanillaPINNsVsFBPINNs/tree/main/src/FBPINNs)

## Implementation
- [SciANN Implementation](https://github.com/tirtho109/VanillaPINNsVsFBPINNs/blob/main/src/SciANN/SciANNImplementation.md)
- [FBPINNs Implementation](https://github.com/tirtho109/VanillaPINNsVsFBPINNs/blob/main/src/FBPINNs/FBPINNsImplementation.md)

## Results
Our results indicate that FBPINNs outperform vanilla PINNs, particularly in cases with data from only a quasi-stationary time domain with few dynamics, and maintain robustness even with noisy datasets.
[View results.](https://github.com/tirtho109/VanillaPINNsVsFBPINNs/tree/main/src/PaperPlots)
