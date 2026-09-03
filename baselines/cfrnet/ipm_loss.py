"""
ipm_loss.py — IPM (Integral Probability Metric) losses cho CFRNet
===================================================================
Cung cấp 2 loại IPM để đo khoảng cách phân phối giữa
biểu diễn của nhóm treated và control:

  - MMD (Maximum Mean Discrepancy) với Gaussian kernel
    -> compute_mmd(z_t, z_c, sigma=1.0)

  - Wasserstein distance (Sinkhorn approximation)
    -> compute_wasserstein(z_t, z_c, p=1, n_iter=100, reg=0.1)

  - Giao diện thống nhất:
    -> compute_ipm(z, t, mode='mmd' | 'wass', **kwargs)

Tham chiếu:
  - Shalit et al., "Estimating individual treatment effect:
    generalization bounds and algorithms" (ICML 2017)
    https://arxiv.org/abs/1606.03976
"""

import torch


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian Kernel MMD
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Tính Gaussian kernel matrix K(x, y).

    Tham số
    -------
    x     : (n, d)
    y     : (m, d)
    sigma : bandwidth

    Trả về
    ------
    K : (n, m)  với K_ij = exp(−||x_i − y_j||² / (2σ²))
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)  # (n, m, d)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)  # (n, m, d)

    sq_dist = (tiled_x - tiled_y).pow(2).sum(2)           # (n, m)
    return torch.exp(-sq_dist / (2.0 * sigma ** 2))


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) với Gaussian kernel:
        MMD²(x, y) = E[k(x,x)] + E[k(y,y)] - 2·E[k(x,y)]

    Tham số
    -------
    x     : (n, d) — biểu diễn nhóm treated
    y     : (m, d) — biểu diễn nhóm control
    sigma : bandwidth của Gaussian kernel

    Trả về
    ------
    mmd : scalar Tensor
    """
    k_xx = _gaussian_kernel(x, x, sigma)
    k_yy = _gaussian_kernel(y, y, sigma)
    k_xy = _gaussian_kernel(x, y, sigma)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Sinkhorn–Wasserstein
# ══════════════════════════════════════════════════════════════════════════════

def compute_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    p: int = 1,
    n_iter: int = 100,
    reg: float = 0.1,
) -> torch.Tensor:
    """
    Sinkhorn approximation của Wasserstein-p distance (differentiable).
    Dùng làm IPM penalty trong CFR-WASS.

    Tham số
    -------
    x     : (n, d)
    y     : (m, d)
    p     : bậc của cost (1 = Earth Mover's Distance, 2 = Wasserstein-2)
    n_iter: số vòng lặp Sinkhorn
    reg   : regularisation (entropy strength)

    Trả về
    ------
    W : scalar Tensor — Sinkhorn–Wasserstein distance
    """
    n = x.size(0)
    m = y.size(0)

    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)

    # Cost matrix C_ij = ||x_i − y_j||^p
    C = torch.cdist(x, y, p=2).pow(p)           # (n, m)

    # Sinkhorn iterations (log-domain stabilized)
    K = torch.exp(-C / reg)                      # (n, m)
    u = torch.ones(n, 1, device=x.device) / n   # (n, 1)

    for _ in range(n_iter):
        v = (1.0 / m) / (K.t() @ u + 1e-8)     # (m, 1)
        u = (1.0 / n) / (K @ v + 1e-8)          # (n, 1)

    T = u * K * v.t()                            # (n, m) transport plan
    return (T * C).sum()


# ══════════════════════════════════════════════════════════════════════════════
# Unified interface
# ══════════════════════════════════════════════════════════════════════════════

def compute_ipm(
    z: torch.Tensor,
    t: torch.Tensor,
    mode: str = "mmd",
    **kwargs,
) -> torch.Tensor:
    """
    Tính IPM penalty giữa phân phối biểu diễn của treated và control.

    Tham số
    -------
    z    : (n, d) — shared representation từ CFRNet
    t    : (n,) hoặc (n, 1) — treatment indicator (0/1)
    mode : 'mmd'  → compute_mmd(z_t, z_c, sigma=...)
           'wass' → compute_wasserstein(z_t, z_c, p=..., n_iter=..., reg=...)
    **kwargs : các tham số tùy chọn truyền vào từng hàm IPM tương ứng.

    Trả về
    ------
    ipm : scalar Tensor — 0 nếu một trong hai nhóm rỗng.
    """
    if mode not in ("mmd", "wass"):
        raise ValueError(f"Unknown IPM mode '{mode}'. Choose from: 'mmd', 'wass'.")

    t_flat = t.view(-1).float()
    z_t = z[t_flat == 1]   # treated representations
    z_c = z[t_flat == 0]   # control representations

    if z_t.size(0) == 0 or z_c.size(0) == 0:
        return torch.tensor(0.0, device=z.device)

    if mode == "mmd":
        sigma = kwargs.get("sigma", 1.0)
        return compute_mmd(z_t, z_c, sigma=sigma)
    else:  # 'wass'
        p = kwargs.get("p", 1)
        n_iter = kwargs.get("n_iter", 100)
        reg = kwargs.get("reg", 0.1)
        return compute_wasserstein(z_t, z_c, p=p, n_iter=n_iter, reg=reg)
