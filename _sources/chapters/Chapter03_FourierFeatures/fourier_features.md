![image.png](fourier_features_files/image.png)
code: 

# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

**Authors**: Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng  
**Institutions**: UC Berkeley, Google Research, UC San Diego  
**Venue**: arXiv preprint, submitted June 2020  
**Link**: https://arxiv.org/abs/2006.10739

---

## Introduction

In recent years, **coordinate-based multilayer perceptrons (MLPs)** have emerged as a powerful representation for continuous signals in **computer vision** and **graphics**. These MLPs take spatial coordinates (e.g., 2D $(x, y)$ or 3D $(x, y, z)$) as inputs and output corresponding signal values, such as color, density, or occupancy. This method offers compactness and is well-suited for gradient-based learning, leading to impressive applications like **NeRF** (Mildenhall et al., 2020) and **Occupancy Networks** (Mescheder et al., 2019).

However, a key limitation of standard MLPs is their **spectral bias**: they favor learning low-frequency functions and struggle to capture high-frequency details, a challenge described in several studies such as Rahaman et al. (2019) and Basri et al. (2020).

To address this, the authors introduce **Fourier feature mappings**, inspired by **random Fourier features** from kernel theory (Rahimi & Recht, 2007). By transforming the input coordinates via sinusoidal functions before passing them to the MLP, they effectively modify the network's **Neural Tangent Kernel (NTK)** to become **stationary** and **tunable**, allowing the network to learn high-frequency components more efficiently.

### Key Equation: Fourier Feature Mapping

The core transformation is:

$$
\gamma(\mathbf{v}) = \left[
a_1 \cos(2\pi \mathbf{b}_1^T \mathbf{v}),\ a_1 \sin(2\pi \mathbf{b}_1^T \mathbf{v}),\ \dots,\ a_m \cos(2\pi \mathbf{b}_m^T \mathbf{v}),\ a_m \sin(2\pi \mathbf{b}_m^T \mathbf{v})
\right]^T
$$

This mapping allows an MLP to represent functions with **higher frequency content**, and its success depends more on the **scale** of frequency sampling than the precise distribution shape.

In this blog, we will explore:

1. **Background** on coordinate-based MLPs and spectral bias  
2. **The Core Idea** of Fourier feature mappings and their theoretical basis via NTK  
3. **The contributions** of this paper to the graphics and vision community

---



```python

```
