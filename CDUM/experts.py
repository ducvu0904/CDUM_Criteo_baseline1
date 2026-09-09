import torch 
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 expert_dim
                 ):
        super(Expert, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, expert_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        h = self.hidden(x)
        h = self.relu(h)
        f = self.output(h)
        return f

class UserExpert(nn.Module):
    def __init__(self,
                 num_experts,
                 input_dim,
                 hidden_dim,
                 expert_dim
                 ):
        super(UserExpert, self).__init__()
        
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, expert_dim) for _ in range(num_experts)])
        
    def forward(self, x: torch.Tensor):
        outputs = []
        
        for expert in self.experts:
            f = expert(x)
            outputs.append(f)
            
        return torch.stack(outputs, dim=1)
class GuidanceGate(nn.Module):
    def __init__(self, guidance_dim, num_experts):
        super(GuidanceGate, self).__init__()
        self.gate = nn.Linear(guidance_dim, num_experts)
        
    def forward(self, guidance_emb: torch.Tensor):
        gate_logits = self.gate(guidance_emb)
        
        gate_weights = torch.softmax(gate_logits,
                                     dim=1)
        return gate_weights
        