"""
model.py — S-Learner (Single Learner for Uplift Modeling)
=========================================================
Kiến trúc:
    Input (x, dim=12) ──┐
                        ├──► Concatenate [x, t] (dim=13)
    Treatment (t, 0/1) ──┘
                             │
                             ▼
                 Shared Representation (2 lớp Linear + ReLU)
                             │
                             ▼
                 Outcome Head: 2 lớp ẩn (Linear + ReLU) + 1 Linear output → y_logit

Cơ chế:
    - S-Learner sử dụng một mô hình duy nhất ước lượng E[Y | X, T].
    - Biến can thiệp T được ghép vào vector đặc trưng X như một feature bổ sung.
    - Dự đoán Uplift (CATE): tau(x) = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0)
"""

import torch
import torch.nn as nn
from typing import Tuple


class SLearnerModel(nn.Module):
    """
    S-Learner (Single Learner) Neural Network:
    - Đầu vào: Vector đặc trưng x (input_dim) ghép với treatment indicator t (0 hoặc 1).
    - Shared representation: 2 lớp Linear + ReLU.
    - Outcome head: 2 lớp ẩn Linear + ReLU và 1 lớp Linear ra logit.
    """

    def __init__(
        self,
        input_dim: int = 12,
        shared_dim: int = 64,
        head_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.head_dim = head_dim

        # ── 1. Shared Representation (nhận input_dim + 1 treatment feature)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + 1, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
        )

        # ── 2. Outcome Head
        self.head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass cho cặp (x, t).

        Tham số
        -------
        x : torch.Tensor of shape (batch_size, input_dim)
        t : torch.Tensor of shape (batch_size,) hoặc (batch_size, 1)

        Trả về
        ------
        y_logit : torch.Tensor of shape (batch_size, 1)
        y_prob  : torch.Tensor of shape (batch_size, 1)
        """
        t = t.float().view(-1, 1)
        xt = torch.cat([x, t], dim=-1)
        phi = self.shared_net(xt)
        y_logit = self.head(phi)
        y_prob = torch.sigmoid(y_logit)
        return y_logit, y_prob

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Factual Binary Cross-Entropy Loss:
        Tính loss giữa dự đoán outcome theo treatment thực tế t và nhãn y.
        """
        y_logit, _ = self.forward(x, t)
        return nn.functional.binary_cross_entropy_with_logits(
            y_logit.squeeze(-1), y.float()
        )

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dự đoán potential outcomes counterfactuals:
            y0_prob = P(Y=1 | X=x, T=0)
            y1_prob = P(Y=1 | X=x, T=1)

        Trả về
        ------
        y0_prob : torch.Tensor of shape (batch_size,)
        y1_prob : torch.Tensor of shape (batch_size,)
        """
        batch_size = x.size(0)
        t0 = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        t1 = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        _, y0_prob = self.forward(x, t0)
        _, y1_prob = self.forward(x, t1)

        return y0_prob.squeeze(-1), y1_prob.squeeze(-1)
