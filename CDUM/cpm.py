from typing import Optional
import torch
import torch.nn as nn
from .experts import UserExpert, GuidanceGate
from .encoder import FeatureEncoder
from .treatment_refine import TreatmentRefine

class TreatmentTower(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 32,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super(TreatmentTower, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU() if activation.lower() == "relu" else nn.Identity()
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.layer2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, mixed: torch.Tensor, e_ind):
        h = self.layer1(mixed)
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = h * e_ind   
        
        y_hat = self.layer2(h) 
        return  y_hat  

class CPM(nn.Module):
    def __init__(self, 
                 num_features: int,
                 num_bins: int,
                 embedding_dim: int = 32,
                 
                 refine_hidden_dim: int = 64,
                 refine_dim: int = 32,
                 
                 num_experts: int = 3,
                 expert_hidden_dim: int = 128,
                 expert_dim: int = 64,
                 tower_hidden_dim: Optional[int] = None,
                 activation: str = "relu",
                 dropout_rate: float = 0.0,
                 use_bn: bool = False):
        super(CPM, self).__init__()
        
        if tower_hidden_dim is None:
            tower_hidden_dim = refine_dim  # default 32 to match indicator embedding e_ind
        
        self.encoder = FeatureEncoder(num_features=num_features, num_bins=num_bins, embedding_dim=embedding_dim)
        self.treatment_refine = TreatmentRefine(treatment_dim=embedding_dim, hidden_dim=refine_hidden_dim, output_dim=refine_dim)
        self.user_experts = UserExpert(
            num_experts=num_experts,
            input_dim=num_features * embedding_dim,
            hidden_dim=expert_hidden_dim,
            expert_dim=expert_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )
        
        self.control_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        self.treatment_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        
        self.control_tower = TreatmentTower(
            input_dim=expert_dim,
            hidden_dim=tower_hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )
        self.treatment_tower = TreatmentTower(
            input_dim=expert_dim,
            hidden_dim=tower_hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )
        
    def _forward_treatment(self, 
                           expert_outputs,
                           treatment_id):
        batch_size = expert_outputs.shape[0]
        
        t = torch.full(
            (batch_size,), 
            treatment_id,
            dtype=torch.long,
            device = expert_outputs.device
        )
        
        t_emb = self.encoder.encode_treatment(t)
        
        # Avg pooling for treatment embedding
        if t_emb.dim() == 3:
            t_emb = t_emb.mean(dim=1)
        
        e_guidance, e_indicator = self.treatment_refine(t_emb)
        
        if treatment_id == 0:
            gate_weights = self.control_gate(e_guidance)
        else:
            gate_weights = self.treatment_gate(e_guidance)
        
        gate_weights = gate_weights.unsqueeze(-1)
        weighted_experts = (expert_outputs * gate_weights).sum(dim=1)
        
        if treatment_id == 0:
            y_hat = self.control_tower(weighted_experts, e_indicator)
        else:
            y_hat = self.treatment_tower(weighted_experts, e_indicator)
        
        return y_hat
    
    def forward(self, x_ids, t):
        x_emb = self.encoder.encode_features(x_ids)
        x_emb = x_emb.flatten(start_dim=1)
        
        expert_outputs = self.user_experts(x_emb)
        
        y0_hat = self._forward_treatment(expert_outputs, treatment_id=0)
        y1_hat = self._forward_treatment(expert_outputs, treatment_id=1)
        
        t_float = t.float().view(-1,1)
        y_factual = t_float * y1_hat + (1 - t_float) * y0_hat
        uplift = y1_hat - y0_hat
        return {
            "y_factual": y_factual,
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "uplift": uplift
        }
        
        
        