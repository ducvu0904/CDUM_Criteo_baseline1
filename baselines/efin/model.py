"""
model.py — EFIN (Explicit Feature Interaction-aware Uplift Network)
===================================================================
Triển khai mô hình EFIN cho bài toán Uplift Modeling (Individual Treatment
Effect Estimation) theo bài báo KDD 2023:
    Liu et al., "Explicit Feature Interaction-aware Uplift Network
    for Online Marketing", KDD 2023.
    https://doi.org/10.1145/3580305.3599820

Kiến trúc EFIN gồm 4 module chính:
1. Feature Encoder Module:
   - Mã hóa đặc trưng người dùng/ngữ cảnh x (d_x chiều) thành embedding không gian K_d:
       e^x_{ij} = W_j * x_{ij} + b_j  (với đặc trưng liên tục)
   - Mã hóa đặc trưng treatment t thành embedding e^t (K_d chiều).

2. Self-Interaction Module:
   - Sử dụng Self-Attention trên tập đặc trưng non-treatment để học phản hồi tự nhiên (natural response):
       Q = K = V = [e^x_{i0}; e^x_{i1}; ...; e^x_{id_x}]
       e_tilde^x_i = softmax(QK^T / sqrt(K_d)) * V
   - Nối (concat) e_tilde^x_i và đưa qua MLP để dự đoán ŷ_i(0):
       ŷ_i(0) = MLP_s(concat(e_tilde^x_i))
   - Loss giám sát trên nhóm control: L_S = L(ŷ_i(0), y_i(0))

3. Treatment-aware Interaction Module:
   - Mô hình hóa tương tác giữa treatment embedding e^t_i và từng feature e^x_{ij} qua attention:
       α_j^i = Softmax_j(W_t0^T * ReLU(W_t1 * e^t_i + W_t2 * e^x_{ij} + b_t2))
       e^{xt}_i = sum_{j=1}^{d_x} α_j^i * e^x_{ij}
   - Ước lượng ITE từ biểu diễn tương tác:
       τ̂_k(x_i) = MLP_t(e^{xt}_i)
   - Dự đoán phản hồi khi có treatment:
       ŷ_i(k) = ŷ_i(0) + τ̂_k(x_i)
   - Loss giám sát trên nhóm treatment: L_T = L(ŷ_i(k), y_i(k))

4. Intervention Constraint Module:
   - Dự đoán group label từ e^{xt}_i và huấn luyện với nhãn đảo (inverse label):
       t̂_i0 = MLP_c(e^{xt}_i)
       L_C = L(t̂_i0, t̄_i0), với t̄_i0 = 1 - t_i0 (trường hợp binary)
   - Giúp cân bằng phân phối ITE giữa control và treatment, chống phân rã do non-random assignment.

Mục tiêu tối ưu hóa toàn cục:
   min_θ L_EFIN = L_S + L_T + λ_c * L_C + weight_decay * ||θ||^2
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousFeatureEncoder(nn.Module):
    """
    Bộ mã hóa đặc trưng liên tục sang không gian embedding K_d theo Eq. (3):
        e^x_{ij} = W_j * x_{ij} + b_j
    với W_j, b_j là tham số học riêng cho từng feature.
    """

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Khởi tạo tham số chiếu riêng cho từng feature: shape (1, num_features, embed_dim)
        self.weight = nn.Parameter(torch.empty(1, num_features, embed_dim))
        self.bias = nn.Parameter(torch.empty(1, num_features, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số
        -------
        x : (batch_size, num_features) — Tensor đặc trưng liên tục.

        Trả về
        ------
        e_x : (batch_size, num_features, embed_dim)
        """
        # x: (B, num_features, 1)
        x_exp = x.unsqueeze(-1)
        # Broadcasting: (B, num_features, 1) * (1, num_features, embed_dim) + (1, num_features, embed_dim)
        e_x = x_exp * self.weight + self.bias
        return e_x


class SelfInteractionModule(nn.Module):
    """
    Self-Interaction Module (Eq. 4, 5, 6):
    - Dùng self-attention để mô hình hóa tương tác giữa các đặc trưng non-treatment:
        Q = K = V = e^x
        e_tilde^x = softmax(Q K^T / sqrt(K_d)) V
    - Trải phẳng concat(e_tilde^x) và đưa qua MLP để dự đoán natural response ŷ(0).
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        shared_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

        # Multi-layer perceptron dự đoán ŷ(0)
        concat_dim = num_features * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, e_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tham số
        -------
        e_x : (batch_size, num_features, embed_dim)

        Trả về
        ------
        y0_logit : (batch_size, 1) — raw logit phản hồi tự nhiên khi t=0
        e_tilde  : (batch_size, num_features, embed_dim) — biểu diễn sau self-attention
        """
        # Q = K = V = e_x: (B, num_features, embed_dim)
        # Q @ K^T: (B, num_features, embed_dim) @ (B, embed_dim, num_features) -> (B, num_features, num_features)
        scores = torch.bmm(e_x, e_x.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # e_tilde: (B, num_features, embed_dim)
        e_tilde = torch.bmm(attn_weights, e_x)

        # Concat toàn bộ features: (B, num_features * embed_dim)
        e_concat = e_tilde.reshape(e_tilde.size(0), -1)

        # y0 logit: (B, 1)
        y0_logit = self.mlp(e_concat)
        return y0_logit, e_tilde


class TreatmentAwareInteractionModule(nn.Module):
    """
    Treatment-Aware Interaction Module (Eq. 8, 9, 10):
    - Tính attention weights giữa treatment embedding e^t và từng non-treatment feature e^x_j:
        α_j = Softmax_j(W_t0^T * ReLU(W_t1 * e^t + W_t2 * e^x_j + b_t2))
    - Tổng hợp thông tin nhạy cảm với treatment:
        e^{xt} = sum_{j=1}^{d_x} α_j * e^x_j
    - Dự đoán ITE (uplift score) qua MLP:
        τ̂_k(x) = MLP_t(e^{xt})
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dim: int = 64,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim

        # W_t1 chiếu e^t
        self.w_t1 = nn.Linear(embed_dim, attn_dim, bias=False)
        # W_t2 chiếu e^x_j kèm bias b_t2
        self.w_t2 = nn.Linear(embed_dim, attn_dim, bias=True)
        # W_t0 chiếu ra scalar score
        self.w_t0 = nn.Linear(attn_dim, 1, bias=False)

        # MLP_t dự đoán ITE τ̂_k
        self.mlp_tau = nn.Sequential(
            nn.Linear(embed_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self, e_x: torch.Tensor, e_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tham số
        -------
        e_x : (batch_size, num_features, embed_dim) — biểu diễn đặc trưng non-treatment
        e_t : (batch_size, embed_dim) — biểu diễn treatment

        Trả về
        ------
        tau_hat : (batch_size, 1) — ITE ước lượng
        e_xt    : (batch_size, embed_dim) — biểu diễn tổng hợp có tương tác treatment
        alpha   : (batch_size, num_features) — trọng số attention của từng feature
        """
        # W_t1 * e^t: (B, 1, attn_dim)
        t_proj = self.w_t1(e_t).unsqueeze(1)

        # W_t2 * e^x: (B, num_features, attn_dim)
        x_proj = self.w_t2(e_x)

        # ReLU(W_t1 * e^t + W_t2 * e^x + b_t2) -> (B, num_features, attn_dim)
        interaction = F.relu(t_proj + x_proj)

        # W_t0^T -> (B, num_features, 1) -> squeeze ra (B, num_features)
        energy = self.w_t0(interaction).squeeze(-1)
        alpha = F.softmax(energy, dim=-1)  # (B, num_features)

        # e^{xt} = sum_j alpha_j * e^x_j: (B, 1, num_features) @ (B, num_features, embed_dim) -> (B, embed_dim)
        e_xt = torch.bmm(alpha.unsqueeze(1), e_x).squeeze(1)

        # ITE logit / score
        tau_hat = self.mlp_tau(e_xt)
        return tau_hat, e_xt, alpha


class InterventionConstraintModule(nn.Module):
    """
    Intervention Constraint Module (Eq. 13, 14):
    - Nhận biểu diễn e^{xt} (chứa thông tin tương tác nhạy cảm với treatment).
    - Dự đoán group membership t̂_i0 = MLP_c(e^{xt}).
    - Được huấn luyện với nhãn đảo t̄_i0 = 1 - t_i0 nhằm tạo perturbation đối nghịch,
      giúp cân bằng phân phối ITE giữa control và treatment.
    """

    def __init__(self, embed_dim: int, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, e_xt: torch.Tensor) -> torch.Tensor:
        """
        Trả về
        ------
        t_logit : (batch_size, 1) — logit dự đoán group membership
        """
        return self.mlp_c(e_xt)


class EFINModel(nn.Module):
    """
    EFIN (Explicit Feature Interaction-aware Uplift Network) hoàn chỉnh.

    Tham số
    -------
    input_dim       : int   — số đặc trưng non-treatment (mặc định 12 cho Criteo).
    embed_dim       : int   — số chiều embedding K_d (mặc định 64, tương ứng Table 2: 2^5, 2^6, 2^7).
    num_treatments  : int   — số nhóm treatment (mặc định 2 cho binary treatment: 0=control, 1=treatment).
    shared_dim      : int   — số neuron lớp ẩn cho Self-Interaction MLP (mặc định 128).
    head_dim        : int   — số neuron lớp ẩn cho các head ITE và Constraint (mặc định 64).
    attn_dim        : int   — số chiều ẩn cho treatment-aware attention (mặc định 64).
    dropout         : float — dropout rate.
    detach_y0       : bool  — nếu True, detach ŷ(0) khi tính ŷ(k) = ŷ(0).detach() + τ̂(x)
                              để cô lập gradient natural response từ nhóm treatment.
    """

    def __init__(
        self,
        input_dim: int = 12,
        embed_dim: int = 64,
        num_treatments: int = 2,
        shared_dim: int = 128,
        head_dim: int = 64,
        attn_dim: int = 64,
        dropout: float = 0.0,
        detach_y0: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_treatments = num_treatments
        self.detach_y0 = detach_y0

        # ── 1. Feature Encoder Module ─────────────────────────────────────────
        self.feature_encoder = ContinuousFeatureEncoder(
            num_features=input_dim, embed_dim=embed_dim
        )
        self.treatment_embedding = nn.Embedding(num_treatments, embed_dim)

        # ── 2. Self-Interaction Module ────────────────────────────────────────
        self.self_interaction = SelfInteractionModule(
            num_features=input_dim,
            embed_dim=embed_dim,
            shared_dim=shared_dim,
            head_dim=head_dim,
            dropout=dropout,
        )

        # ── 3. Treatment-aware Interaction Module ──────────────────────────────
        self.treatment_interaction = TreatmentAwareInteractionModule(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            head_dim=head_dim,
            dropout=dropout,
        )

        # ── 4. Intervention Constraint Module ─────────────────────────────────
        self.intervention_constraint = InterventionConstraintModule(
            embed_dim=embed_dim,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass của mô hình EFIN.

        Tham số
        -------
        x : (batch_size, input_dim) — Tensor đặc trưng người dùng/ngữ cảnh.
        t : Optional (batch_size,)  — Nhãn treatment thực tế (nếu None, mặc định giả định t=1 để tính uplift).

        Trả về
        ------
        y0_logit : (batch_size, 1) — logit dự đoán outcome nhóm control ŷ(0)
        y1_logit : (batch_size, 1) — logit dự đoán outcome nhóm treatment ŷ(1)
        tau_hat  : (batch_size, 1) — ước lượng ITE τ̂(x)
        t_logit  : (batch_size, 1) — logit dự đoán group membership của constraint module
        alpha    : (batch_size, input_dim) — attention weights nhạy cảm treatment
        """
        B = x.size(0)

        # 1. Feature Encoding
        e_x = self.feature_encoder(x)  # (B, input_dim, embed_dim)

        # 2. Self-interaction: phản hồi tự nhiên ŷ(0)
        y0_logit, _ = self.self_interaction(e_x)  # (B, 1)

        # 3. Treatment encoding cho nhánh treatment (k=1)
        # Để ước lượng hiệu ứng can thiệp (uplift của treatment k=1 so với control 0),
        # ta mã hóa treatment=1
        t_target = torch.ones(B, dtype=torch.long, device=x.device)
        e_t1 = self.treatment_embedding(t_target)  # (B, embed_dim)

        # 4. Treatment-aware interaction: ước lượng τ̂_1(x)
        tau_hat, e_xt, alpha = self.treatment_interaction(e_x, e_t1)

        # 5. Kết hợp dự đoán treatment response (Eq. 11): ŷ(1) = ŷ(0) + τ̂(x)
        if self.detach_y0:
            y1_logit = y0_logit.detach() + tau_hat
        else:
            y1_logit = y0_logit + tau_hat

        # 6. Intervention constraint prediction (Eq. 13)
        t_logit = self.intervention_constraint(e_xt)

        return y0_logit, y1_logit, tau_hat, t_logit, alpha

    def compute_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        lambda_c: float = 0.01,
        loss_type: str = "bce",
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Tính toán hàm mất mát EFIN (Eq. 2, 7, 12, 14):
            L_EFIN = L_S + L_T + λ_c * L_C

        Tham số
        -------
        x                 : (batch_size, input_dim)
        t                 : (batch_size,) hoặc (batch_size, 1)
        y                 : (batch_size,) hoặc (batch_size, 1)
        lambda_c          : float — trọng số mất mát của Intervention Constraint Module.
        loss_type         : str — 'bce' (Binary Cross-Entropy với logits) hoặc 'mse'.
        return_components : bool — nếu True, trả về (total_loss, components_dict).
        """
        t_flat = t.view(-1).long()
        y_col = y.float().view(-1, 1)
        t_col = t.float().view(-1, 1)

        y0_logit, y1_logit, tau_hat, t_logit, _ = self.forward(x, t=t_flat)

        ctrl_mask = (t_flat == 0)
        treat_mask = (t_flat == 1)

        # ── 1. Loss tự tương tác L_S trên nhóm control (Eq. 7) ────────────────
        if ctrl_mask.any():
            if loss_type == "bce":
                l_s = F.binary_cross_entropy_with_logits(y0_logit[ctrl_mask], y_col[ctrl_mask])
            else:
                l_s = F.mse_loss(y0_logit[ctrl_mask], y_col[ctrl_mask])
        else:
            l_s = torch.tensor(0.0, device=x.device, requires_grad=True)

        # ── 2. Loss tương tác treatment L_T trên nhóm treatment (Eq. 12) ──────
        if treat_mask.any():
            if loss_type == "bce":
                l_t = F.binary_cross_entropy_with_logits(y1_logit[treat_mask], y_col[treat_mask])
            else:
                l_t = F.mse_loss(y1_logit[treat_mask], y_col[treat_mask])
        else:
            l_t = torch.tensor(0.0, device=x.device, requires_grad=True)

        # ── 3. Intervention constraint loss L_C với nhãn đảo (Eq. 14) ────────
        # Nhãn đảo t̄ = 1 - t (đối kháng group discrimination)
        inverse_t = (1.0 - t_col).to(x.device)
        if loss_type == "bce":
            l_c = F.binary_cross_entropy_with_logits(t_logit, inverse_t)
        else:
            l_c = F.mse_loss(torch.sigmoid(t_logit), inverse_t)

        total_loss = l_s + l_t + lambda_c * l_c

        if return_components:
            return total_loss, {
                "loss_s": l_s,
                "loss_t": l_t,
                "loss_c": l_c,
                "total_loss": total_loss,
            }
        return total_loss

    @torch.no_grad()
    def predict_uplift(
        self, x: torch.Tensor, loss_type: str = "bce"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dự đoán xác suất outcome cho cả 2 nhánh:
            y0_prob = P(Y=1 | T=0, X)
            y1_prob = P(Y=1 | T=1, X)
        Uplift score sau đó được tính là: tau = y1_prob - y0_prob.

        Trả về
        ------
        y0_prob : (batch_size,)
        y1_prob : (batch_size,)
        """
        y0_logit, y1_logit, _, _, _ = self.forward(x)
        if loss_type == "bce":
            y0_prob = torch.sigmoid(y0_logit).squeeze(-1)
            y1_prob = torch.sigmoid(y1_logit).squeeze(-1)
        else:
            y0_prob = torch.clamp(y0_logit, 0.0, 1.0).squeeze(-1)
            y1_prob = torch.clamp(y1_logit, 0.0, 1.0).squeeze(-1)
        return y0_prob, y1_prob

    @torch.no_grad()
    def predict_tau(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất trực tiếp ước lượng ITE τ̂(x) từ Treatment-Aware Module theo Eq. (10).
        """
        _, _, tau_hat, _, _ = self.forward(x)
        return tau_hat.squeeze(-1)

    @torch.no_grad()
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lấy attention weights α nhạy cảm với treatment của từng feature (Eq. 8).
        """
        _, _, _, _, alpha = self.forward(x)
        return alpha
