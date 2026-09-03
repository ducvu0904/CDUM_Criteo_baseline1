"""
model.py — DragonNet (Criteo Uplift v2.1)
=========================================
Kiến trúc:
    Input (x)
       │
       ▼
  Shared Representation (2 lớp Linear + ReLU)
       │
       ├──► Head 0 (Control):     2 lớp ẩn (Linear + ReLU) + 1 Linear → y0_logit
       │
       ├──► Head 1 (Treatment):   2 lớp ẩn (Linear + ReLU) + 1 Linear → y1_logit
       │
       └──► Propensity Head:      1 Linear → t_logit  (P(T=1|X))

Sự khác biệt với TARNet:
- Thêm propensity head: ước lượng P(T=1|X) từ shared representation.
- Thêm parameter epsilon cho targeted regularisation (TMLE-style).
- Loss = Factual BCE + propensity BCE + alpha * targeted regularisation term.

Tham khảo:
    Shi et al., "Adapting Neural Networks for the Estimation of Treatment Effects" (2019)
    https://arxiv.org/abs/1906.02120
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNetModel(nn.Module):
    """
    DragonNet: TARNet + propensity head + epsilon cho targeted regularisation.

    Tham số
    -------
    input_dim   : số chiều input feature (mặc định 12 cho Criteo).
    shared_dim  : số neuron trong shared representation.
    head_dim    : số neuron trong mỗi outcome head.
    """

    def __init__(
        self,
        input_dim: int = 12,
        shared_dim: int = 200,
        head_dim: int = 100,
    ):
        super().__init__()

        # ── 1. Shared Representation 
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim), nn.ReLU(),
        )

        # ── 2. Head 0 - Control: 2 lớp ẩn + 1 Linear ra logit ──
        self.head0 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        # ── 3. Head 1 - Treatment: 2 lớp ẩn + 1 Linear ra logit ──
        self.head1 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        # ── 4. Propensity Head: P(T=1 | X) từ shared representation ──
        self.propensity_head = nn.Linear(shared_dim, 1)

        # ── 5. Epsilon: scalar parameter cho targeted regularisation ──
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        Forward pass qua toàn bộ DragonNet.

        Trả về
        ------
        y0_logit        : (n, 1) — logit outcome dưới control
        y1_logit        : (n, 1) — logit outcome dưới treatment
        y0_prob         : (n, 1) — xác suất outcome dưới control
        y1_prob         : (n, 1) — xác suất outcome dưới treatment
        propensity_logit: (n, 1) — logit propensity score P(T=1|X)
        propensity_prob : (n, 1) — xác suất P(T=1|X) = sigmoid(propensity_logit)
        epsilon         : (n,)   — epsilon scalar (expand theo batch size)
        """
        phi = self.shared_net(x)

        y0_logit = self.head0(phi)
        y1_logit = self.head1(phi)

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)

        propensity_logit = self.propensity_head(phi)
        propensity_prob = torch.sigmoid(propensity_logit)

        eps = self.epsilon.expand(x.size(0))

        return y0_logit, y1_logit, y0_prob, y1_prob, propensity_logit, propensity_prob, eps

    def compute_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        DragonNet Loss:
            L_factual    = BCE(y_hat_factual, y)
            L_propensity = BCE(t_hat, t)
            L_targeted   = mean((y - y_hat + eps * (t - t_hat))^2)  (targeted reg)

            L_total = L_factual + alpha * L_propensity + beta * L_targeted

        Tham số
        -------
        alpha : trọng số của propensity loss.
        beta  : trọng số của targeted regularisation loss.
        """
        y0_logit, y1_logit, y0_prob, y1_prob, prop_logit, prop_prob, eps = self.forward(x)

        t = t.float().view(-1, 1)
        y = y.float().view(-1, 1)

        # ── Factual BCE loss ──────────────────────────────────────────────────
        y_logit = t * y1_logit + (1.0 - t) * y0_logit
        l_factual = F.binary_cross_entropy_with_logits(y_logit, y)

        # ── Propensity BCE loss ───────────────────────────────────────────────
        l_propensity = F.binary_cross_entropy_with_logits(prop_logit, t)

        # ── Targeted Regularisation (TMLE-style) ─────────────────────────────
        # y_hat: predicted outcome theo factual treatment
        y_hat = t * y1_prob + (1.0 - t) * y0_prob
        # eps được expand thành (n, 1) để broadcast
        eps_col = eps.view(-1, 1)
        targeted_residual = y - y_hat + eps_col * (t - prop_prob)
        l_targeted = targeted_residual.pow(2).mean()

        return l_factual + alpha * l_propensity + beta * l_targeted

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor):
        """
        Dự đoán uplift: tau = y1_prob - y0_prob.

        Trả về
        ------
        y0_prob : (n,)
        y1_prob : (n,)
        """
        _, _, y0_prob, y1_prob, _, _, _ = self.forward(x)
        return y0_prob.squeeze(-1), y1_prob.squeeze(-1)

    @torch.no_grad()
    def predict_propensity(self, x: torch.Tensor):
        """
        Dự đoán propensity score P(T=1|X).

        Trả về
        ------
        propensity_prob : (n,)
        """
        _, _, _, _, _, propensity_prob, _ = self.forward(x)
        return propensity_prob.squeeze(-1)
