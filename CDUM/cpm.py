import torch
import torch.nn as nn
from .experts import UserExpert, GuidanceGate
from .encoder import FeatureEncoder
from .treatment_refine import TreatmentRefine

class TreatmentTower(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
    ):
        super(TreatmentTower, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, mixed: torch.Tensor, e_ind):
        h = self.layer1(mixed)
        h = self.relu(h)
        
        h = h * e_ind   
        
        y_hat = self.layer2(h) 
        return  y_hat  

class CPM(nn.Module):
    def __init__(self, 
                 num_features: int,
                 num_bins: int,
                 embedding_dim: int,
                 
                 refine_hidden_dim: int,
                 refine_dim: int,
                 
                 num_experts:int,
                 expert_hidden_dim: int,
                 expert_dim: int):
        super(CPM, self).__init__()
        
        self.encoder = FeatureEncoder(num_features=num_features, num_bins=num_bins, embedding_dim=embedding_dim)
        self.treatment_refine = TreatmentRefine(treatment_dim=embedding_dim, hidden_dim = refine_hidden_dim, output_dim = refine_dim)
        self.user_experts = UserExpert(num_experts=num_experts, input_dim=embedding_dim, hidden_dim=expert_hidden_dim, expert_dim=expert_dim)
        
        self.control_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        self.treatment_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        
        self.control_tower = TreatmentTower(input_dim = expert_dim, hidden_dim = expert_hidden_dim)
        self.treatment_tower = TreatmentTower(input_dim= expert_dim, hidden_dim= expert_hidden_dim)
        
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
        
        
        