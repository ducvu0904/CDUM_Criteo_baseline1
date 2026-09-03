"""
model.py — TARNet (Treatment-Agnostic Representation Network)
=============================================================
Kiến trúc:
    Input (x) 
       │
       ▼
  Shared Representation (2 lớp Linear + ReLU)
       │
       ├──► Head 0 (Control):   2 lớp ẩn (Linear + ReLU) + 1 Linear output → y0_logit
       │
       └──► Head 1 (Treatment): 2 lớp ẩn (Linear + ReLU) + 1 Linear output → y1_logit
"""

import torch
import torch.nn as nn


class TARNetModel(nn.Module):
    """
    TARNet:
    - Shared representation: 2 lớp Linear 
    - 2 Outcome heads: mỗi head gồm 2 lớp ẩn Linear và 1 lớp Linear ra logit
    """
    def __init__(self, input_dim: int = 12, shared_dim: int = 64, head_dim: int = 32):
        super().__init__()

        # ── 1. Shared Representation 
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )

        # ── 2. Head 0 - Control 
        self.head0 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )

        # ── 3. Head 1 - Treatment
        self.head1 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Trả về:
            y0_logit : (n, 1)
            y1_logit : (n, 1)
            y0_prob  : (n, 1)
            y1_prob  : (n, 1)
        """
        # Đi qua 2 lớp shared representation
        phi = self.shared_net(x)

        # Đi qua 2 outcome heads
        y0_logit = self.head0(phi)
        y1_logit = self.head1(phi)

        # Xác suất dự đoán
        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)

        return y0_logit, y1_logit, y0_prob, y1_prob

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Factual BCE Loss:
        - Mẫu t=0 tính loss trên head0
        - Mẫu t=1 tính loss trên head1
        """
        y0_logit, y1_logit, _, _ = self.forward(x)

        # Chọn logit tương ứng với treatment thực tế
        t = t.float().view(-1, 1)
        y_logit = t * y1_logit + (1.0 - t) * y0_logit

        return nn.functional.binary_cross_entropy_with_logits(y_logit.squeeze(-1), y.float())

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor):
        """
        Dự đoán uplift: tau = y1_prob - y0_prob
        """
        _, _, y0_prob, y1_prob = self.forward(x)
        return y0_prob.squeeze(-1), y1_prob.squeeze(-1)
