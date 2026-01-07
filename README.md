# PINNs for a Congestion-Driven PDE with Transported Capacity
Based on arXiv:[TBC]
This repository contains a **Physics-Informed Neural Network (PINN)** implementation in PyTorch for a congestion-type PDE system derived analytically. The aim is to improve our understanding of the qualitative properties of this system without resorting to traditional methods (e.g. finite difference) which do not scale well to higher dimensions.

The implementation uses a **two-stage optimisation strategy (Adam → L-BFGS)** and enforces initial conditions through **hard conditioning** built directly into the neural network architecture.

---

## PDE model

The model solved by the PINN is the 1D system
$$
\begin{cases}
\partial_t \rho + \partial_x(\rho u) = 0, \\\\
\partial_t(\rho u) + \partial_x(\rho u^2)
- \partial_x\!\left( \dfrac{\mu}{1-\rho}\,\partial_x u \right)
+ \partial_x\!\left( \left(\dfrac{\rho}{\rho^\*}\right)^\gamma \right)
= \rho f, \\\\
\partial_t \rho^\* + u\,\partial_x \rho^\* = 0.
\end{cases}
$$

### Variables and parameters
- $\rho(x,t)$ : density  
- $u(x,t)$ : velocity  
- $\rho^\*(x,t)$ : transported congestion threshold (capacity)  
- $\mu > 0$ : viscosity parameter (singular as $\rho \to 1$)  
- $\gamma > 0$ : congestion exponent  
- $f$ : external forcing (constant in this code)

This system combines:
- hyperbolic transport,
- singular / degenerate viscosity,
- a **moving congestion constraint** through $\rho^\*$.

## Network architecture

The PINN is a fully connected multilayer perceptron (MLP):

- **Input:** $(x,t)$  
- **Output (raw):** $(\delta \rho,\ \delta u,\ \delta \rho^\*)$  
- **Hidden layers:** configurable (default: 5 layers of width 256)  
- **Activation:** `tanh` (configurable)

### Hard initial-condition conditioning

The final network outputs are constructed as
- $\rho(x,t) = \rho_0(x)\,\exp\!\big(A(t)\,\delta\rho\big)$  
- $u(x,t) = u_0(x) + A(t)\,\delta u$  
- $\rho^\*(x,t) = \rho^\*_0(x)\,\exp\!\big(A(t)\,\delta\rho^\*\big)$  

where $A(t)=t/T$ is a time-scheduling factor that ensures **exact satisfaction of the initial conditions at $t=0$**.

---

## Loss function

The total loss is given by
$$
\mathcal{L}
= \mathcal{L}_{\text{PDE}}
+ \mathcal{L}_{\text{IC}}
+ \mathcal{L}_{\text{BC}}
+ \mathcal{L}_{\text{IC-derivatives}}.
$$

Here:
- $\mathcal{L}_{\text{PDE}}$ enforces the three PDE equations,
- $\mathcal{L}_{\text{IC}}$ enforces the initial data,
- $\mathcal{L}_{\text{BC}}$ enforces boundary consistency,
- $\mathcal{L}_{\text{IC-derivatives}}$ (optional) penalises spatial derivatives at $t=0$ to regularise initial gradients.

All loss weights are fully configurable.

---

## Repository contents

This repository is intentionally lightweight and self-contained:

├── pinn_vcs.py # Main PINN implementation 
└── README.md # This documentation




