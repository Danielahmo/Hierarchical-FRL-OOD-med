import torch
import torch.nn.functional as F

def get_gaussian_kernel_3d(size=5, sigma=1.0):
    assert size % 2 == 1, "Kernel size must be odd"

    ax = torch.arange(size) - size // 2
    gauss = torch.exp(-0.5 * (ax / sigma)**2)
    gauss = gauss / gauss.sum()

    kernel_1d = gauss
    kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)
    kernel_3d = kernel_3d / kernel_3d.sum()  # normalize
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # shape [1, 1, D, H, W]
    return kernel_3d

def gaussian_blur_3d(x, k, stride=1, padding=1):
    C = x.shape[1]
    k = k.to(x.device)
    k = k.repeat(C, 1, 1, 1, 1)  # shape: [C, 1, D, H, W]
    padding = (k.shape[-1] // 2, k.shape[-2] // 2, k.shape[-3] // 2)
    x = F.pad(x, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode='replicate')
    return F.conv3d(x, k, stride=stride, padding=0, groups=C)

def find_voxel_high_freq(mri, gauss_kernel_3d):
    """Extract high-frequency 3D component from MRI (grayscale or single-channel)."""
    padding = (gauss_kernel_3d.shape[-1] - 1) // 2
    blurred = gaussian_blur_3d(mri, gauss_kernel_3d, padding=padding)
    high_freq = mri - blurred
    return high_freq, blurred
    
def normalize(x, eps=1e-8):
    min_val = x.amin(dim=[2, 3, 4], keepdim=True)
    max_val = x.amax(dim=[2, 3, 4], keepdim=True)
    return (x - min_val) / (max_val - min_val + eps)

def process_freq(x,gauss_size):
    gauss_kernel = get_gaussian_kernel_3d(size=gauss_size)
    hf, lf = find_voxel_high_freq(x, gauss_kernel)
    #hf = (hf + 1)/2
    hf = normalize(hf)
    lf= normalize(lf)

    return hf, lf