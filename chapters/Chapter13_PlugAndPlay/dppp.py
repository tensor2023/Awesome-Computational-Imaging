import torch
import torch.nn.functional as F
from typing import Callable, Iterable
import functools
import matplotlib.pyplot as plt
import torch
import re
from typing import Tuple, Callable, Literal
TensorLike = torch.Tensor
ResizeFnType = Callable[[TensorLike, int, bool], TensorLike]
import matplotlib.pyplot as plt
import os
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_unet import UNetRes as net

lpips_model_alex = None
# 定义一个函数_rfft2d，用于对输入的二维张量x进行实数域快速傅里叶变换
def _rfft2d(x):
    return torch.fft.rfft2(x, dim=(-2, -1))

def _irfft2d(x, like):
    return torch.fft.irfft2(x, s=like.shape[-2:], dim=(-2, -1))

def upscale_no_interpolation(x, scale_factor):
    # Assume input is [B, H, W, C] → convert to [B, C, H, W]
    x = x.permute(0, 3, 1, 2)
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up.permute(0, 2, 3, 1)  # back to [B, H, W, C]

def _splits(x, sf, w):
    B, C, H, W = x.shape
    x = x.view(B, C, H // sf, sf, W // sf, sf)
    return x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H // sf, W // sf, -1)

def _pre_calculate(x, k, scale_factor):
    B, C, H, W = x.shape
    k_h, k_w = k.shape[-2:]

    # use grayscale kernel first
    k_single_channel = torch.flip(k[0, 0], dims=[0, 1]).unsqueeze(0).unsqueeze(0)

    pad_h = int(H * scale_factor - k_h)
    pad_w = int(W * scale_factor - k_w)
    k_padded = F.pad(k_single_channel, (0, pad_w, 0, pad_h))
    k_padded = torch.roll(k_padded, shifts=(-k_h // 2, -k_w // 2), dims=(-2, -1))

    FB = _rfft2d(k_padded)

    n_ops = k_h * k_w * np.log2(k_h * k_w)
    threshold = n_ops * 2.22e-16
    FB = torch.where(
        torch.abs(FB.imag) >= threshold,
        FB,
        FB.real.to(torch.complex64)
    )

    FBC = torch.conj(FB)
    F2B = (FB.real2 + FB.imag2).to(torch.complex64)

    Sty = upscale_no_interpolation(x, scale_factor)
    FBFy = FBC * _rfft2d(Sty)

    return FB, FBC, F2B, FBFy

def _get_rho_sigma(sigma=0.01, iter_num=15, model_sigma1=0.2, model_sigma2=0.01, w=1, lambd=0.23):
    logspace = torch.logspace(np.log10(model_sigma1), np.log10(model_sigma2), iter_num)
    linspace = torch.linspace(model_sigma1, model_sigma2, iter_num)
    sigmas = logspace * w + linspace * (1 - w)
    rhos = lambd * (sigma  2) / (sigmas  2)
    return rhos.to(torch.float32), sigmas.to(torch.float32)

def _data_solution_closed_form(x, alpha, FB, FBC, F2B, FBFy, scale_factor):
    alpha_complex = torch.tensor(alpha, dtype=torch.complex64, device=x.device)
    x_cf = x  # shape [B, C, H, W]

    FR = FBFy + _rfft2d(alpha * x_cf)

    B, C, H, W = x_cf.shape
    FBR = _splits(FB * FR, scale_factor, W).mean(dim=-1)
    invW = _splits(F2B, scale_factor, W).mean(dim=-1)

    invWBR = FBR / (invW + alpha_complex)
    invWBR_tiled = torch.tile(invWBR, (1, 1, scale_factor, scale_factor))[..., :FBC.shape[-1]]

    FCBinvWBR = FBC * invWBR_tiled
    FX = (FR - FCBinvWBR) / alpha_complex
    x_est = _irfft2d(FX, like=x_cf)

    return x_est


def resize_fn(x, scale_factor, upsample=True, mode='bicubic'):
    """
    Resize function for HQS / iterative data consistency step.

    Args:
        x (torch.Tensor): Image tensor of shape [B, C, H, W]
        scale_factor (int): Upscale factor
        upsample (bool): 
            - True: upsample (e.g., LR -> HR)
            - False: downsample (e.g., HR -> LR)
        mode (str): interpolation mode (e.g., 'bicubic', 'bilinear', 'nearest')

    Returns:
        torch.Tensor: resized tensor
    """
    if upsample:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    else:
        B, C, H, W = x.shape
        new_H, new_W = H // scale_factor, W // scale_factor
        return F.interpolate(x, size=(new_H, new_W), mode=mode, align_corners=False)

# def _data_solution_iterative(
#     x, alpha, degraded, resize_fn, scale_factor, num_steps=30, step_size=1.5
# ):
#     for _ in range(num_steps):
#         data_err = degraded - resize_fn(x, scale_factor, False)
#         x = x - step_size * resize_fn(-data_err, scale_factor, True)
#     return x

def _data_solution_iterative(
    x, alpha, degraded, resize_fn, scale_factor, num_steps=30, step_size=1.5, mode='bicubic'
):
    """
    Iterative data consistency solver. If resize_fn is None, fallback to internal F.interpolate.
    
    Args:
        x (Tensor): [B, C, H, W] high-res estimate
        alpha (float): not used
        degraded (Tensor): [B, C, h, w] low-res observed image
        resize_fn (callable or None): resizing function or None
        scale_factor (int): upsampling/downsampling factor
        num_steps (int): number of iterations
        step_size (float): update step size
        mode (str): fallback interpolation mode when resize_fn is None
        
    Returns:
        Tensor: updated high-res estimate
    """
    for _ in range(num_steps):
        # Downsample
        if resize_fn is not None:
            x_down = resize_fn(x, scale_factor, upsample=False)
        else:
            B, C, H, W = x.shape
            h, w = H // scale_factor, W // scale_factor
            x_down = F.interpolate(x, size=(h, w), mode=mode, align_corners=False)

        # Residual
        data_err = degraded - x_down

        # Upsample
        if resize_fn is not None:
            data_err_up = resize_fn(data_err, scale_factor, upsample=True)
        else:
            B, C, H, W = x.shape
            data_err_up = F.interpolate(data_err, size=(H, W), mode=mode, align_corners=False)

        # Update
        x = x + step_size * data_err_up

    return x

def hqs_super_resolve(
    degraded: TensorLike,
    image: TensorLike,
    sr_factor: int,
    denoiser: Callable[[TensorLike, float], TensorLike],
    max_denoiser_stddev: float,
    kernel: TensorLike = None,
    resize_fn: ResizeFnType = None,
    noise_stddev: float = 0,
    num_steps: int = 60,
    num_steps_data: int = 5,
    step_size_data: float = 1.5,
    callbacks: Iterable[Callable[[TensorLike, int], None]] = None,
):
    if callbacks is None:
        callbacks = []

    B, H_lr, W_lr, C = degraded.shape
    H, W = H_lr * sr_factor, W_lr * sr_factor

    rhos, sigmas = _get_rho_sigma(
        sigma=max(noise_stddev, 0.001),
        iter_num=num_steps,
        model_sigma1=max_denoiser_stddev,
        model_sigma2=max(sr_factor / 255.0, noise_stddev)
    )

    if kernel is not None:
        print('Closed Form')
        FB, FBC, F2B, FBFy = _pre_calculate(degraded, kernel, sr_factor)
        data_solution = functools.partial(
            _data_solution_closed_form,
            FB=FB, FBC=FBC, F2B=F2B, FBFy=FBFy,
            scale_factor=sr_factor,
        )
    else:
        print('Iterative')
        data_solution = functools.partial(
            _data_solution_iterative,
            degraded=degraded,
            resize_fn=resize_fn,
            scale_factor=sr_factor,
            num_steps=num_steps_data,
            step_size=step_size_data,
        )

    if sr_factor == 1:
        x = degraded
    else:
        if resize_fn is not None:
            x = resize_fn(degraded, sr_factor, True)
        else:
            x = F.interpolate(degraded.permute(0, 3, 1, 2), size=(H, W), mode='bicubic', align_corners=True)
            # x = x.permute(0, 2, 3, 1)  # back to [B, H, W, C]
    denoiser_requires_noise_map=1#改


    for step in tqdm(range(num_steps), desc="HQS steps"):
        
        x = data_solution(x, rhos[step])

        if denoiser_requires_noise_map:
            B, C, H, W = x.shape
            sigma_map = torch.full((B, 1, H, W), sigmas[step], dtype=x.dtype, device=x.device)
            x_input = torch.cat([x, sigma_map], dim=1)
            x = denoiser(x_input)
        else:
            x = denoiser(x)

        for callback in callbacks:
            callback(x, step)

        # 👇 每跑完 num_steps // 10 的整数倍就 show 一次
        if (step + 1) % (num_steps // 10) == 0:
            reconstructed=x.clamp(0,1)
            
            psnr_val = psnr(image, reconstructed).item()
            lpips_val = lpips_alex(image, reconstructed).item()
            print(f"PSNR: {psnr_val:.2f}, LPIPS: {lpips_val:.4f}")

            visualize_images([
                image.squeeze(),
                degraded.squeeze(),
                reconstructed.squeeze()
            ], titles=["Original", "Degraded", "Reconstructed"])



    return x

def conv2D_filter_rgb(kernel: torch.Tensor):
    """
    Args:
        kernel: shape [H, W]
    Returns:
        RGB kernel: shape [3, 3, H, W]
    """
    kernel = torch.from_numpy(kernel).float()
    zeros = torch.zeros_like(kernel)
    filter_r = torch.stack([kernel, zeros, zeros], dim=0)
    filter_g = torch.stack([zeros, kernel, zeros], dim=0)
    filter_b = torch.stack([zeros, zeros, kernel], dim=0)
    full_filter = torch.stack([filter_r, filter_g, filter_b], dim=0)  # shape: [3, 3, H, W]
    return full_filter


def blur(image, kernel, noise_stddev=0.0,
         mode: Literal["constant", "wrap"] = "constant", clip_final=True):
    """
    Args:
        image: [B, C, H, W], float32, range [0, 1]
        kernel: [3, 3, kH, kW]
        noise_stddev: float, standard deviation of Gaussian noise
        mode: "constant" or "wrap"
    Returns:
        blurred image [B, C, H, W]
    """
    B, C, H, W = image.shape
    kH, kW = kernel.shape[-2:]

    # padding
    pad_h, pad_w = kH // 2, kW // 2
    padding_mode = "circular" if mode == "wrap" else "constant"
    padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)

    # convolution
    kernel = kernel.to(image.device)  # [3, 3, kH, kW]
    out = torch.zeros_like(image)
    for c in range(3):
        k = kernel[c][c].unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
        x = padded[:, c:c+1]
        out[:, c:c+1] = F.conv2d(x, k, padding=0)

    # add noise
    if noise_stddev > 0:
        noise = torch.randn_like(out) * noise_stddev
        out = out + noise

    if clip_final:
        out = torch.clamp(out, 0.0, 1.0)

    return out


def load_denoiser(device) -> Tuple[Callable, Tuple[float, float]]:
    """
    Args:
        path_to_model: Path to the model. The filename should include the noise level range, e.g., model_0.01-0.04.pt

    Returns:
        (denoiser_fn, (min_sigma, max_sigma))

    """
    # match = re.match(r".*?(\d\.\d+)?-?(\d\.\d+)\.pth", path_to_model)
    # if match[1]:
    #     min_sigma, max_sigma = float(match[1]), float(match[2])
    # else:
    #     min_sigma = max_sigma = float(match[2])

    model = net(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose').to(device)  # define network
    path_to_model='../../../code/PnP/DPIR-master/model_zoo/drunet_color.pth'#'../../../code/PnP/DPIR-master/model_zoo/drunet_deblocking_color.pth'#


    model.load_state_dict(torch.load(path_to_model), strict=True)
    model.eval()

    def denoiser(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    return denoiser


def visualize_images(images, titles=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in [1, 3]:  # [C, H, W]
                img = img.permute(1, 2, 0).cpu().detach().numpy()  # [H, W, C]
            elif img.dim() == 2:  # [H, W]
                img = img.cpu().detach().numpy()
        axs[i].imshow(img, cmap=None if img.ndim == 3 else "gray")
        axs[i].axis("off")
        if titles:
            axs[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()

base_dir = os.path.abspath(".")  
kernel_path = os.path.join(base_dir, "../../../code/PnP/DPIR-master/kernels/Levin09.mat")

with h5py.File(
    os.path.normpath(kernel_path), "r"#os.path.join(__file__, "..", "kernels", "Levin09.mat")
) as f:
    NB_DEBLURRING_LEVIN_KERNELS = [
        f[k_ref[0]][()].astype("float32") for k_ref in f["kernels"][()]
    ]

import lpips  # pip install lpips

def init_lpips_model_alex():
    """
    Initialize the LPIPS model using AlexNet (PyTorch version).
    Automatically downloads model weights if not present.
    """
    global lpips_model_alex
    lpips_model_alex = lpips.LPIPS(net='alex')  # use AlexNet backbone
    lpips_model_alex.eval()



def psnr(imgs_a: torch.Tensor, imgs_b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two batches of images.

    Args:
        imgs_a, imgs_b: tensors of shape [B, C, H, W] with values in [0, 1]
        max_val: the maximum possible pixel value

    Returns:
        PSNR for each image in the batch, shape [B]
    """
    mse = torch.mean((imgs_a - imgs_b)  2, dim=[1, 2, 3])
    psnr_vals = 10 * torch.log10((max_val  2) / (mse + 1e-8))
    return psnr_vals


def lpips_alex(imgs_a: torch.Tensor, imgs_b: torch.Tensor) -> torch.Tensor:
    """
    Compute LPIPS perceptual similarity between two batches of images.

    Args:
        imgs_a, imgs_b: tensors of shape [B, C, H, W], normalized to [-1, 1]

    Returns:
        Tensor of shape [B] with LPIPS distances
    """
    global lpips_model_alex
    if lpips_model_alex is None:
        init_lpips_model_alex()
    return lpips_model_alex(imgs_a, imgs_b).squeeze()
