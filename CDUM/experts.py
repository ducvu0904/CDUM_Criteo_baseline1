import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        expert_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super(Expert, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, expert_dim)

        self.relu = nn.ReLU() if activation.lower() == "relu" else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.hidden(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        f = self.output(h)
        f = self.relu(f)
        return f


class UserExpert(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int = 128,
        expert_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super(UserExpert, self).__init__()

        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                expert_dim=expert_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                use_bn=use_bn,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        outputs = []

        for expert in self.experts:
            f = expert(x)
            outputs.append(f)

        return torch.stack(outputs, dim=1)


class GuidanceGate(nn.Module):
    def __init__(self, guidance_dim: int, num_experts: int):
        super(GuidanceGate, self).__init__()
        # Gate network hidden units: (), i.e. no hidden layers
        self.gate = nn.Linear(guidance_dim, num_experts)

    def forward(self, guidance_emb: torch.Tensor):
        gate_logits = self.gate(guidance_emb)
        gate_weights = torch.softmax(gate_logits, dim=1)
        return gate_weights

        