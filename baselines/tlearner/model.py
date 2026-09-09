"""
model.py — T-Learner (Two Learners for Uplift Modeling)
=========================================================
Kiến trúc:
    Input (x, dim=12)
        │
        ├──► Net 0 (Control Model):   Linear(input_dim→shared_dim) → ReLU
        │                             → Linear(shared_dim→shared_dim) → ReLU
        │                             → Linear(shared_dim→head_dim) → ReLU
        │                             → Linear(head_dim→head_dim) → ReLU
        │                             → Linear(head_dim→1) → y0_logit
        │
        └──► Net 1 (Treated Model):   Linear(input_dim→shared_dim) → ReLU
                                      → Linear(shared_dim→shared_dim) → ReLU
                                      → Linear(shared_dim→head_dim) → ReLU
                                      → Linear(head_dim→head_dim) → ReLU
                                      → Linear(head_dim→1) → y1_logit

Cơ chế:
    - T-Learner huấn luyện 2 mô hình hoàn toàn độc lập (không chia sẻ trọng số):
        + Net 0 ước lượng E[Y(0) | X] trên nhóm đối chứng (T=0)
        + Net 1 ước lượng E[Y(1) | X] trên nhóm can thiệp (T=1)
    - Trong mỗi batch huấn luyện:
        + Mẫu t=0 chỉ cập nhật gradient cho Net 0
        + Mẫu t=1 chỉ cập nhật gradient cho Net 1
    - Dự đoán Uplift (CATE): tau(x) = y1_prob - y0_prob
"""

import torch
import torch.nn as nn
from typing import Tuple


class TLearnerModel(nn.Module):
    """
    T-Learner (Two Learners) Neural Network:
    - Gồm 2 mạng neural hoàn toàn độc lập: net0 (cho nhóm control) và net1 (cho nhóm treatment).
    - Không chia sẻ trọng số (unshared parameters), tránh hiện tượng treatment bias bị triệt tiêu.
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

        # ── 1. Mạng Net 0 - Dành riêng cho nhóm Control (T=0)
        self.net0 = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        # ── 2. Mạng Net 1 - Dành riêng cho nhóm Treatment (T=1)
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass qua cả 2 nhánh net0 và net1.

        Tham số
        -------
        x : torch.Tensor of shape (batch_size, input_dim)

        Trả về
        ------
        y0_logit : torch.Tensor of shape (batch_size, 1)
        y1_logit : torch.Tensor of shape (batch_size, 1)
        y0_prob  : torch.Tensor of shape (batch_size, 1)
        y1_prob  : torch.Tensor of shape (batch_size, 1)
        """
        y0_logit = self.net0(x)
        y1_logit = self.net1(x)

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)

        return y0_logit, y1_logit, y0_prob, y1_prob

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Factual BCE Loss:
        - Mẫu t=0 cập nhật loss cho net0
        - Mẫu t=1 cập nhật loss cho net1
        """
        y0_logit, y1_logit, _, _ = self.forward(x)

        # Chọn logit tương ứng với treatment thực tế
        t = t.float().view(-1, 1)
        y_logit = t * y1_logit + (1.0 - t) * y0_logit

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
        _, _, y0_prob, y1_prob = self.forward(x)
        return y0_prob.squeeze(-1), y1_prob.squeeze(-1)
