import torch.nn as nn
import torch

class FeatureEncoder(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_bins: int,
                 embedding_dim: int
    ):
        super(FeatureEncoder, self).__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(num_bins, embedding_dim) for _ in range(num_features)]
        )
        self.treatment_embeddings = nn.Embedding(2, embedding_dim)
    
    def forward(self, x: torch.Tensor, t:torch.Tensor):
        feature_embeddings = []

        for j in range(self.num_features):
            feat_values = x[:, j]
            feat_emb = self.feature_embeddings[j](feat_values)
            feature_embeddings.append(feat_emb)

        x_emb = torch.stack(feature_embeddings, dim=1)
        t = t.long()
        t_emb = self.treatment_embeddings(t)
        return x_emb, t_emb

class FeatureAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_emb: torch.Tensor, t_emb: torch.Tensor):
        x_emb = x_emb.mean(dim=1)
        return x_emb, t_emb