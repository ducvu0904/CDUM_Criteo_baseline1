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
    
    def encode_treatment(self, t: torch.Tensor):
        t = t.long()
        t_emb = self.treatment_embeddings(t)
        return t_emb
    def encode_features(self, x: torch.Tensor):
        feature_embeddings = []

        for j in range(self.num_features):
            feat_values = x[:, j]
            feat_emb = self.feature_embeddings[j](feat_values)
            feature_embeddings.append(feat_emb)

        x_emb = torch.stack(feature_embeddings, dim=1)
        return x_emb
    def forward(self, x: torch.Tensor, t:torch.Tensor):
        feature_embeddings = []

        for j in range(self.num_features):
            feat_values = x[:, j]
            feat_emb = self.feature_embeddings[j](feat_values)
            feature_embeddings.append(feat_emb)

        x_emb = torch.stack(feature_embeddings, dim=1)
        
        t_emb = self.encode_treatment(t)
        return x_emb, t_emb


    
class FeatureAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_emb: torch.Tensor, t_emb: torch.Tensor):
        # Concat flatten for feature embeddings
        x_emb = x_emb.flatten(start_dim=1)
        # Avg pooling for treatment embedding
        if t_emb.dim() == 3:
            t_emb = t_emb.mean(dim=1)
        return x_emb, t_emb