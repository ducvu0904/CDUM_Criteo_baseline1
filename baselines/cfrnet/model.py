"""
model.py — CFRNet (Counterfactual Regression Network) cho Criteo Uplift v2.1
=============================================================================
Kiến trúc:
    Input (x)
       │
       ▼
  Shared Representation φ(x)  (3 lớp Linear + ReLU — theo Shalit 2017)
       │
       ├──► Head 0 (Control):     2 lớp ẩn + 1 Linear → y0_logit
       │
       └──► Head 1 (Treatment):   2 lớp ẩn + 1 Linear → y1_logit

  Loss:
      L_factual  = BCE(ŷ_factual, y)
      L_ipm      = IPM(φ(x)[T=1], φ(x)[T=0])   via MMD hoặc Wasserstein
      L_total    = L_factual + lambda_ipm * L_ipm

Sự khác biệt với TARNET:
  - Thêm IPM regulariser để cân bằng phân phối biểu diễn giữa 2 nhóm,
    giúp giảm confounding bias trong ước lượng uplift.
  - Hỗ trợ 2 loại IPM: 'mmd' (Maximum Mean Discrepancy)
                        'wass' (Sinkhorn–Wasserstein approximation)

Tham chiếu:
  Shalit et al., "Estimating individual treatment effect:
  generalization bounds and algorithms." ICML 2017.
  https://arxiv.org/abs/1606.03976
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ipm_loss import compute_ipm


class CFRNetModel(nn.Module):
    """
    CFRNet: TARNet + IPM regularisation.

    Tham số
    -------
    input_dim   : số chiều feature đầu vào (12 với Criteo).
    shared_dim  : số neuron trong shared representation.
    head_dim    : số neuron trong mỗi outcome head.
    mode        : loại IPM — 'mmd' (MMD Gaussian kernel) hoặc 'wass' (Sinkhorn Wasserstein).
    lambda_ipm  : trọng số của IPM loss trong tổng loss.
    """

    def __init__(
        self,
        input_dim: int = 12,
        shared_dim: int = 200,
        head_dim: int = 100,
        mode: str = "wass",
        lambda_ipm: float = 1.0,
    ):
        super().__init__()

        assert mode in ("mmd", "wass"), f"Unknown IPM mode '{mode}'. Choose from: 'mmd', 'wass'."
        self.mode = mode
        self.lambda_ipm = lambda_ipm

        # ── 1. Shared Representation (3 lớp, theo Shalit 2017) ──
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim), nn.ReLU(),
        )

        # ── 2. Head 0 - Control
        self.head0 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        # ── 3. Head 1 - Treatment
        self.head1 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Trả về
        ------
        phi      : (n, shared_dim) — shared representation
        y0_logit : (n, 1)
        y1_logit : (n, 1)
        y0_prob  : (n, 1)
        y1_prob  : (n, 1)
        """
        phi = self.shared_net(x)

        y0_logit = self.head0(phi)
        y1_logit = self.head1(phi)

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)

        return phi, y0_logit, y1_logit, y0_prob, y1_prob

    def compute_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        **ipm_kwargs,
    ) -> torch.Tensor:
        """
        CFRNet Loss:
            L = BCE(ŷ_factual, y) + lambda_ipm * IPM(φ[T=1], φ[T=0])

        Tham số
        -------
        x          : (n, d) features
        t          : (n,) treatment indicator (0/1)
        y          : (n,) observed outcome (0/1)
        **ipm_kwargs: tham số tùy chọn cho IPM (sigma / p / n_iter / reg)

        Trả về
        ------
        loss : scalar Tensor
        """
        phi, y0_logit, y1_logit, _, _ = self.forward(x)

        t_col = t.float().view(-1, 1)
        y_col = y.float().view(-1, 1)

        # Factual BCE loss
        y_logit = t_col * y1_logit + (1.0 - t_col) * y0_logit
        l_factual = F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), y.float())

        # IPM loss (mode và kwargs được quyết định lúc khởi tạo / gọi)
        l_ipm = compute_ipm(phi, t, mode=self.mode, **ipm_kwargs)

        return l_factual + self.lambda_ipm * l_ipm

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor):
        """
        Dự đoán uplift: tau = y1_prob - y0_prob.

        Trả về
        ------
        y0_prob : (n,)
        y1_prob : (n,)
        """
        _, _, _, y0_prob, y1_prob = self.forward(x)
        return y0_prob.squeeze(-1), y1_prob.squeeze(-1)

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lấy shared representation φ(x).

        Trả về
        ------
        phi : (n, shared_dim)
        """
        return self.shared_net(x)
