# PINNs for a Congestion-Driven PDE with Transported Capacity
Based on arXiv:[TBC]
This repository contains a **Physics-Informed Neural Network (PINN)** implementation in PyTorch for a congestion-type PDE system derived analytically. The aim is to improve our understanding of the qualitative properties of this system without resorting to traditional methods (e.g. finite difference) which do not scale well to higher dimensions.

---

## PDE model

We consider the following 1D system:

$$
\begin{cases}
\partial_t \rho + \partial_x(\rho u) = 0, \\
\partial_t(\rho u) + \partial_x(\rho u^2)
- \partial_x\left( \dfrac{\mu}{1-\rho}\,\partial_x u \right)
+ \partial_x\left( \left(\dfrac{\rho}{\rho^\*}\right)^\gamma \right)
= \rho f, \\
\partial_t \rho^\* + u\,\partial_x \rho^\* = 0.
\end{cases}
$$

### Variables and parameters

- $\rho(t,x)$: density  
- $u(t,x)$: velocity  
- $\rho^\*(t,x)$: transported congestion threshold (capacity)  
- $\mu > 0$: viscosity parameter (note the singular factor $1/(1-\rho)$)  
- $\gamma > 0$: congestion exponent  
- $f(t,x)$: forcing term  

This system combines:
- hyperbolic transport,
- singular / degenerate viscosity as $\rho \to 1$,
- a **transported unilateral constraint** through $\rho^\*$.

---
