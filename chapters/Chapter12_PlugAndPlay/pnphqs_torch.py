import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import dppp  # Your custom module (should include blur, psnr, lpips_alex, hqs_super_resolve, etc.)
import matplotlib.pyplot as plt

# Automatically select device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("Using device:", device)

# Dataset class to load images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# Smart image display function
def smart_imshow(tensor, title=None):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        if tensor.shape[0] in [1, 3]:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0).numpy()
        else:  # Already in [H, W, C]
            tensor = tensor.numpy()
    elif tensor.dim() == 2:
        tensor = tensor.numpy()
    if tensor.ndim == 2 or (tensor.ndim == 3 and tensor.shape[2] == 1):
        plt.imshow(tensor.squeeze(), cmap='gray')
    else:
        plt.imshow(tensor)
    if title:
        plt.title(title)
    plt.axis("off")

# Path to image dataset
data_root = "/home/xqgao/2025/MIT/Datasets/CBSD68/original"
dataset = ImageFolderDataset(data_root)

# Load one image and move it to device
image = dataset[0].unsqueeze(0).to(device)  # Shape: [1, 3, H, W]

# Prepare a blur kernel
kernel = dppp.conv2D_filter_rgb(dppp.NB_DEBLURRING_LEVIN_KERNELS[0]).to(torch.float32).to(device)

# Simulate degraded image (blur + Gaussian noise)
noise_stddev = 0.04
degraded = dppp.blur(image, kernel, noise_stddev=noise_stddev)

# Load denoiser model (ensure it runs on the same device)
denoiser = dppp.load_denoiser(device=device)

# Deconvolution via HQS (Plug-and-Play image restoration)
reconstructed = dppp.hqs_super_resolve(
    degraded=degraded,
    image=image,
    sr_factor=1,
    denoiser=denoiser,
    max_denoiser_stddev=0.2,
    kernel=None,           # Set to None to use iterative solver
    callbacks=[],          # Optional hooks for visualization or logging
)
