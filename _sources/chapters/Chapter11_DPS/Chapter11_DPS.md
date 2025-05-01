### From Paper to Code: Understanding and Reproducing "DIFFUSION POSTERIOR SAMPLING FOR GENERAL NOISY INVERSE PROBLEMS"
![image.png](Chapter11_DPS_files/image.png)
Published in International Conference on Learning Representations (ICLR), 2023  
Code: [GitHub Repository](https://github.com/DPS2022/diffusion-posterior-sampling), 
Source Code in My Repo: ../../../../code/Inv/diffusion-posterior-sampling-main/sample_condition.py

# Paper Reading Notes

## 1. Highlights

Proposed a guided diffusion sampling method tailored for solving general noisy inverse problems.

## 2. Background

Diffusion models (DMs) have recently shown great success in high-quality image generation, thanks to their ability to model complex data distributions through a sequence of denoising steps. One representative framework is Denoising Diffusion Probabilistic Models (DDPMs) [1], which learn to reverse a fixed forward noising process.

However, these models are typically trained unconditionally or conditionally using label information or prompts (e.g., text). When applied to inverse problems like deblurring, inpainting, or super-resolution, these models face the challenge that:
- The model is not aware of how the measurements (e.g., blurry or masked images) were generated.
- There is no direct way to sample from the posterior distribution $p(x|y)$, where $y$ is the observed measurement.

Existing methods either retrain models with known measurement models (which is expensive and specific), or apply test-time optimization (which is slow and sometimes unstable).
![image.png](Chapter11_DPS_files/image.png)


## 3. Method Overview

This paper introduces Diffusion Posterior Sampling (DPS), a general framework that combines the strength of diffusion models with classical Bayesian inference to solve arbitrary noisy inverse problems, without requiring retraining.

### 3.1 Goal

We want to sample from the posterior distribution $p(x|y)$, where:
- $x$ is the clean data we want to recover,
- $y$ is the observed measurement (e.g., blurred, compressed, masked image),
- and $y = A(x) + n$, where $A$ is a known forward operator and $n$ is noise.

### 3.2 Posterior sampling formulation

We apply Bayes' theorem:
```math
p(x|y) \propto p(y|x) \cdot p(x)
```
Here:
- $p(x)$ is the learned prior from the diffusion model.
- $p(y|x)$ comes from the measurement model $A$.

In the standard diffusion setup, the reverse process is modeled as:

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
```

But to condition this process on $y$, DPS adds a correction based on the measurement log-likelihood $\nabla_{x_t} \log p(y|x_t)$.

### 3.3 DPS Sampling Rule

At each step $t$, given a sample $x_t$, we update:
```math
x_t \leftarrow x_t + \eta_t \nabla_{x_t} \log p(y|x_t)
```
where $\eta_t$ is a step-size schedule.

This gradient is computed by defining a likelihood (based on the known forward model $A$):
```math
\log p(y|x) = -\| A(x) - y \|^2 / (2 \sigma^2)
```

So the final update is a guided sampling:
```math
x_t \leftarrow x_t + \eta_t \nabla_{x_t} \left[ -\| A(x_t) - y \|^2 \right]
```

This allows us to plug in arbitrary forward models $A$, such as:
- masking (for inpainting),
- blurring (for deblurring),
- downsampling (for super-resolution),
without retraining the diffusion model.



## References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*  
[2] Song, Y., & Ermon, S. (2020). Score-based generative modeling through stochastic differential equations. *ICLR*  
[3] Song, J., Meng, C., & Ermon, S. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. *ICLR*



# Code Reproduction with Explanation: Inference via DPS using a Pretrained DPM

This function implements sampling using a pretrained diffusion model, but with a key twist: it performs posterior-guided inference by incorporating measurement consistency.

> In contrast to standard diffusion models, which generate samples unconditionally (or conditionally on high-level information like text), DPS (Diffusion Posterior Sampling) adds a *gradient-based correction* at each denoising step to ensure that the output is consistent with an observed measurement $y = A(x) + n$.

This is particularly useful for solving inverse problems, where we don’t just want to generate “realistic” images — we want to generate images that match what we observed under a known forward model $A$.



### 🔧 Function Breakdown: `p_sample_loop()`

```python
def p_sample_loop(self, model, x_start, measurement, measurement_cond_fn, record, save_root):
```

This function starts from pure noise `x_start` and iteratively denoises it to obtain a final reconstruction `img`, guided by both the diffusion model and the observed measurement `y`.



#### Main Loop: Reverse-Time Sampling

```python
pbar = tqdm(list(range(self.num_timesteps))[::-1])
for idx in pbar:
    time = torch.tensor([idx] * img.shape[0], device=device)
    img = img.requires_grad_()
```

- Loop through all timesteps from $T$ down to 1.
- Like in standard DDPM, the input at each step `x_t` requires gradients to allow gradient-based correction.



#### 1. Standard Diffusion Step

```python
out = self.p_sample(x=img, t=time, model=model)
```

This step computes the standard reverse diffusion prediction, producing:
- `out['sample']`: the denoised sample $x_{t-1}$
- `out['pred_xstart']`: estimated clean image $x_0$

> This is identical to what a regular diffusion model does.



#### 2. Add Measurement-Based Guidance (Key Difference!)

```python
noisy_measurement = self.q_sample(measurement, t=time)
img, distance = measurement_cond_fn(
    x_t=out['sample'],
    measurement=measurement,
    noisy_measurement=noisy_measurement,
    x_prev=img,
    x_0_hat=out['pred_xstart']
)
```

This is where DPS differs from standard diffusion:
- `measurement_cond_fn` uses the forward model $A$ and the current output to compute a gradient correction that pulls `img` closer to satisfying $A(x) ≈ y$.
- The update may look like:
  ```python
  x_t ← x_t + η_t ∇ₓ log p(y | x)
  ```
- The `distance` tracks how far the current estimate is from matching the measurement.



#### 3. Detach and Proceed

```python
img = img.detach_()
```

Reset gradients before the next step. This makes sure each sampling step is independent.



#### 4. Optional Visualization

```python
if record and idx % 100 == 0:
    # Save and show intermediate reconstruction
```

Every 100 steps, the current result is visualized and saved.


```python
return img
```

This is the final output after all denoising and measurement-based corrections.



```python
from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import sys
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
```

    /home/xqgao/anaconda3/envs/inr/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
def main():
    sys.argv = ['']
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_config', type=str, default='/home/xqgao/2025/MIT/code/Inv/diffusion-posterior-sampling-main/configs/model_config.yaml', help='Path to the model configuration file')
    parser.add_argument('--diffusion_config', type=str, default='/home/xqgao/2025/MIT/code/Inv/diffusion-posterior-sampling-main/configs/diffusion_config.yaml', help='Path to the diffusion configuration file')
    parser.add_argument('--task_config', type=str, default='/home/xqgao/2025/MIT/code/Inv/diffusion-posterior-sampling-main/configs/super_resolution_config.yaml', help='Path to the task configuration file (default: {TASK-CONFIG})')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (default: 0)')
    parser.add_argument('--save_dir', type=str, default='/home/xqgao/2025/MIT/code/Inv/diffusion-posterior-sampling-main/results', help='Directory to save results (default: ./results)')

    args = parser.parse_args()

    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, measure_config['operator'])
    noiser = get_noise(measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           measure_config['mask_opt']
        )
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

if __name__ == '__main__':
    main()

```

    2025-04-20 16:39:31,573 [DPS] >> Device set to cuda:0.
    /home/xqgao/2025/MIT/code/Inv/diffusion-posterior-sampling-main/guided_diffusion/unet.py:88: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      model.load_state_dict(th.load(model_path, map_location='cpu'))
    2025-04-20 16:39:34,294 [DPS] >> Operation: super_resolution / Noise: gaussian
    2025-04-20 16:39:34,296 [DPS] >> Conditioning method : ps
    2025-04-20 16:39:34,338 [DPS] >> Inference for image 0
     10%|▉         | 97/1000 [00:06<00:41, 21.99it/s, distance=22.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_1.png)
    


     20%|█▉        | 198/1000 [00:10<00:36, 22.17it/s, distance=17.6]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_3.png)
    


     30%|██▉       | 297/1000 [00:15<00:31, 22.25it/s, distance=16]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_5.png)
    


     40%|███▉      | 399/1000 [00:20<00:27, 21.94it/s, distance=14.2]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_7.png)
    


     50%|████▉     | 498/1000 [00:24<00:23, 21.76it/s, distance=13.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_9.png)
    


     60%|█████▉    | 597/1000 [00:29<00:18, 22.08it/s, distance=12.5]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_11.png)
    


     70%|██████▉   | 699/1000 [00:34<00:13, 22.06it/s, distance=12]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_13.png)
    


     80%|███████▉  | 798/1000 [00:38<00:09, 22.17it/s, distance=11.6]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_15.png)
    


     90%|████████▉ | 899/1000 [00:43<00:04, 22.15it/s, distance=11.1]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_17.png)
    


    100%|█████████▉| 997/1000 [00:48<00:00, 21.94it/s, distance=10.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_19.png)
    


    100%|██████████| 1000/1000 [00:48<00:00, 20.56it/s, distance=10.3]
    2025-04-20 16:40:23,072 [DPS] >> Inference for image 1
     10%|▉         | 99/1000 [00:04<00:40, 22.18it/s, distance=25.9]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_21.png)
    


     20%|█▉        | 198/1000 [00:09<00:36, 21.95it/s, distance=19.7]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_23.png)
    


     30%|██▉       | 297/1000 [00:13<00:31, 22.10it/s, distance=16.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_25.png)
    


     40%|███▉      | 399/1000 [00:18<00:27, 22.03it/s, distance=14.9]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_27.png)
    


     50%|████▉     | 498/1000 [00:23<00:22, 22.08it/s, distance=14]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_29.png)
    


     60%|█████▉    | 598/1000 [00:27<00:18, 21.57it/s, distance=13.2]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_31.png)
    


     70%|██████▉   | 698/1000 [00:34<00:20, 14.50it/s, distance=13]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_33.png)
    


     80%|███████▉  | 798/1000 [00:41<00:13, 15.14it/s, distance=12.6]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_35.png)
    


     90%|████████▉ | 898/1000 [00:48<00:06, 14.97it/s, distance=12.1]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_37.png)
    


    100%|█████████▉| 998/1000 [00:55<00:00, 14.23it/s, distance=11.4]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_39.png)
    


    100%|██████████| 1000/1000 [00:55<00:00, 17.96it/s, distance=11.4]
    2025-04-20 16:41:18,868 [DPS] >> Inference for image 2
     10%|▉         | 98/1000 [00:06<01:02, 14.33it/s, distance=25.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_41.png)
    


     20%|█▉        | 198/1000 [00:13<00:53, 15.07it/s, distance=20.5]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_43.png)
    


     30%|██▉       | 298/1000 [00:20<00:44, 15.95it/s, distance=18.2]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_45.png)
    


     40%|███▉      | 398/1000 [00:26<00:40, 14.74it/s, distance=15]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_47.png)
    


     50%|████▉     | 498/1000 [00:33<00:25, 20.05it/s, distance=13.8]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_49.png)
    


     60%|█████▉    | 598/1000 [00:40<00:28, 13.91it/s, distance=13]  


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_51.png)
    


     70%|██████▉   | 698/1000 [00:47<00:21, 13.79it/s, distance=12.6]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_53.png)
    


     80%|███████▉  | 798/1000 [00:54<00:14, 14.26it/s, distance=11.9]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_55.png)
    


     90%|████████▉ | 898/1000 [01:02<00:07, 14.11it/s, distance=11.3]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_57.png)
    


    100%|█████████▉| 998/1000 [01:09<00:00, 13.86it/s, distance=10.6]


    
![png](Chapter11_DPS_files/Chapter11_DPS_5_59.png)
    


    100%|██████████| 1000/1000 [01:09<00:00, 14.41it/s, distance=10.6]

