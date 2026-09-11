import torch


class EquidistantBucketer:
    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.boundaries = None
        self.min_values = None
        self.max_values = None

    def to(self, device):
        if self.boundaries is not None:
            self.boundaries = self.boundaries.to(device)
        if self.min_values is not None:
            self.min_values = self.min_values.to(device)
        if self.max_values is not None:
            self.max_values = self.max_values.to(device)
        return self

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        X = X.contiguous()
        min_values = X.min(dim=0).values
        max_values = X.max(dim=0).values
        feat_range = max_values - min_values

        num_boundaries = self.num_bins - 1
        ratios = torch.arange(1, self.num_bins, device=X.device, dtype=X.dtype) / self.num_bins

        # Shape: (num_features, num_bins - 1)
        boundaries = min_values.unsqueeze(1) + feat_range.unsqueeze(1) * ratios.unsqueeze(0)

        self.boundaries = boundaries.contiguous()
        self.min_values = min_values.contiguous()
        self.max_values = max_values.contiguous()
        return self

    def transform(self, X):
        if self.boundaries is None:
            raise RuntimeError("The bucketer has not been fitted yet. Call 'fit' before 'transform'.")

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        is_1d = X.dim() == 1
        if is_1d:
            X = X.unsqueeze(0)

        X = X.contiguous()
        num_features = X.shape[1]

        if self.boundaries.shape[0] != num_features:
            raise ValueError(
                f"Feature dimension mismatch: bucketer fitted with {self.boundaries.shape[0]} features, "
                f"but input has {num_features} features."
            )

        bucket_ids = []
        for j in range(num_features):
            # Ensure both input slice and boundaries are contiguous 1D tensors with matching device/dtype
            feat_values = X[:, j].contiguous()
            feat_boundaries = self.boundaries[j].to(device=X.device, dtype=X.dtype).contiguous()

            ids = torch.bucketize(feat_values, feat_boundaries)
            ids = torch.clamp(ids, 0, self.num_bins - 1)
            bucket_ids.append(ids)

        bucket_ids = torch.stack(bucket_ids, dim=1).contiguous()

        if is_1d:
            bucket_ids = bucket_ids.squeeze(0)

        return bucket_ids.long().contiguous()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

