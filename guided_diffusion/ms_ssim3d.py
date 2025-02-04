import torch
import torch.nn.functional as F

def gaussian_kernel(size: int, sigma: float):
    """Creates a 3D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return kernel

def _ssim_3d(img1, img2, kernel, size_average=True):
    """Calculates the SSIM for 3D images."""
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.conv3d(img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1))
    mu2 = F.conv3d(img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, kernel, padding=kernel.size(2)//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel.size(2)//2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map

def ms_ssim_3d(img1, img2, levels=5, size_average=True):
    """Calculates the Multi-Scale SSIM for 3D images."""
    kernel = gaussian_kernel(size=11, sigma=1.5).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(img1.device)
    
    mssim = []
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)

    for _ in range(levels):
        ssim = _ssim_3d(img1, img2, kernel, size_average)
        mssim.append(ssim)

        img1 = F.avg_pool3d(img1, kernel_size=2)
        img2 = F.avg_pool3d(img2, kernel_size=2)

    mssim = torch.stack(mssim)
    return (mssim * weights).sum() if size_average else mssim

# Example usage
# Assuming `image1` and `image2` are 3D images with shape (N, C, D, H, W)
# image1, image2 = ...
