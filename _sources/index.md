# Awesome Computational Imaging

Welcome to **Awesome Computational Imaging**, a curated collection of classic papers and hands-on code implementations across multiple foundational topics, including **computer graphics**, **inverse problems**, and **scientific imaging**.

## Overview of Blog Structure

This blog starts with **Implicit Neural Representations (INRs)**, including **SIREN (Sinusoidal Representation Networks)**, **NeRF (Neural Radiance Fields)**, and **FFN (Fourier Feature Networks)**. These models move beyond traditional mesh-based representations by treating neural networks as continuous functions, which helps overcome the limitations of storing fine details and learning high-frequency information.

Next is **Gaussian Splatting (GS)**, applicable in both 2D and 3D settings. Unlike the previous INR methods, GS eliminates neural networks entirely, significantly accelerating inference while still maintaining high visual fidelity through continuous Gaussian primitives.

Chapters 7 to 9 focus on applying INR techniques to three different **scientific imaging** problems, showing their flexibility in domains beyond graphics.

Then come **Diffusion Models**, which learn data priors through Markov chains. When the forward model (likelihood) is known, they can approximate complex distributions and solve inverse problems in a probabilistic framework.

Finally, **Plug-and-Play (PnP)** methods are introduced—an innovative approach that integrates neural networks into classical optimization algorithms, bridging traditional priors with modern learning-based denoisers.


I hope this resource helps you quickly build intuition and gain practical knowledge in modern computational imaging.

If you notice any mistakes or have suggestions, please let me know right away.



#### TODO: Hide certain cells. Update code in My repo.

