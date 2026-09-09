import torch 

class EquidistantBucketer:
    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.boundaries = None
        
    def fit(self, X):
        min_values = X.min(dim=0).values
        max_values = X.max(dim=0).values
        feat_range = max_values - min_values

        num_boundaries = self.num_bins - 1
        ratios = torch.arange(1, self.num_bins) / self.num_bins
        boundaries = (
            min_values.unsqueeze(0) + ratios.unsqueeze(1) * feat_range.unsqueeze(0)
        )

        self.boundaries = boundaries
        self.min_values = min_values
        self.max_values = max_values

    def transform(self, X):
        if self.boundaries is None:
            raise RuntimeError("The bucketer has not been fitted yet. Call 'fit' before 'transform'.")

        bucket_ids = []

        for j in range(X.shape[1]):
            feat_values = X[:, j]
            feat_boundaries = self.boundaries[j]

            ids = torch.bucketize(feat_values, feat_boundaries)

            bucket_ids.append(ids)

        bucket_ids = torch.stack(bucket_ids, dim=1)

        return bucket_ids.long()

