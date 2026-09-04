"""
trainer.py — Trainer / Wrapper cho CEVAE (Causal Effect Variational AutoEncoder)
=================================================================================
Trainer nhận DataLoaders từ bước preprocess (preprocess/data_loader.py).
Không tạo DataLoader bên trong trainer để phân định rõ ràng trách nhiệm.
"""

import os
import sys
import logging
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from .model import CEVAE as CEVAEModel
except ImportError:
    from baselines.cevae.model import CEVAE as CEVAEModel

try:
    from metrics.uplift_metrics import uplift_auc_score1, qini_auc_score1, uplift_at_k1
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def _to_numpy(data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor], dtype=np.float32) -> np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy(dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype)
    else:
        return np.asarray(data, dtype=dtype)


class CEVAE:
    """
    Trainer / Wrapper cho mô hình CEVAE.
    Nhận DataLoader trực tiếp từ bước preprocess.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        input_dim: int = 12,
        z_dim: int = 20,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        dropout_rate: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = CEVAEModel(
                n_features=input_dim,
                z_dim=z_dim,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Huấn luyện 1 epoch và trả về mean negative ELBO loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_b, t_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            t_b = t_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.elbo(x_b, t_b, y_b)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Validation: Tính negative ELBO loss trên val_loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_b, t_b, y_b in val_loader:
                x_b = x_b.to(self.device)
                t_b = t_b.to(self.device)
                y_b = y_b.to(self.device)

                loss = self.model.elbo(x_b, t_b, y_b)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        test_loader: DataLoader,
        n_samples: int = 10,
        k: float = 0.3,
    ) -> Dict[str, float]:
        """Evaluation: Đánh giá toàn diện trên test_loader từ bước preprocess."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_uplifts, all_t, all_y = [], [], []

        with torch.no_grad():
            for x_b, t_b, y_b in test_loader:
                x_b_dev = x_b.to(self.device)
                t_b_dev = t_b.to(self.device)
                y_b_dev = y_b.to(self.device)

                loss = self.model.elbo(x_b_dev, t_b_dev, y_b_dev)
                total_loss += loss.item()
                n_batches += 1

                y0_mean, y1_mean = self.model.predict_uplift(x_b_dev, t_b_dev, y_b_dev, n_samples=n_samples)
                uplift_b = (y1_mean - y0_mean).cpu().numpy()

                all_uplifts.append(uplift_b)
                all_t.append(t_b.cpu().numpy())
                all_y.append(y_b.cpu().numpy())

        test_loss = total_loss / max(n_batches, 1)
        results = {"loss": test_loss}

        if HAS_METRICS and len(all_uplifts) > 0:
            uplift_scores = np.concatenate(all_uplifts, axis=0)
            t_arr = np.concatenate(all_t, axis=0)
            y_arr = np.concatenate(all_y, axis=0)

            try:
                results["auuc"] = float(uplift_auc_score1(y_arr, uplift_scores, t_arr))
            except Exception as e:
                logger.warning(f"Failed to compute AUUC: {e}")

            try:
                results["qini"] = float(qini_auc_score1(y_arr, uplift_scores, t_arr))
            except Exception as e:
                logger.warning(f"Failed to compute Qini: {e}")

            try:
                results[f"lift@{int(k*100)}%"] = float(
                    uplift_at_k1(y_arr, uplift_scores, t_arr, strategy="overall", k=k)
                )
            except Exception as e:
                logger.warning(f"Failed to compute Uplift@{k}: {e}")

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "cevae",
        writer: Optional[Any] = None,
        verbose: int = 1,
    ) -> Dict[str, list]:
        """Huấn luyện mô hình CEVAE bằng DataLoaders truyền vào trực tiếp."""
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            logger.info(
                f"Starting CEVAE training | Device: {self.device} | Epochs: {epochs}"
            )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)

            log_msg = f"Epoch [{epoch:02d}/{epochs:02d}]  Train Loss: {train_loss:.5f}"

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                log_msg += f" | Val Loss: {val_loss:.5f}"

                current_lr = self.optimizer.param_groups[0]["lr"]
                if writer is not None:
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    writer.add_scalar("LearningRate", current_lr, epoch)

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_dir is not None:
                        ckpt_best = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
                        self.save(ckpt_best)
                        if verbose >= 2:
                            logger.info(f"  -> Saved best checkpoint: {ckpt_best}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            logger.info(f"Early stopping triggered at epoch {epoch}!")
                        break

            if verbose:
                logger.info(log_msg)

        # Lưu checkpoint cuối cùng
        if checkpoint_dir is not None:
            ckpt_final = os.path.join(checkpoint_dir, f"{model_name}_final.pth")
            self.save(ckpt_final)
            if verbose >= 2:
                logger.info(f"  -> Saved final checkpoint: {ckpt_final}")

        if val_loader is not None and checkpoint_dir is not None:
            ckpt_best = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            if os.path.exists(ckpt_best):
                self.load(ckpt_best)

        return history

    @torch.no_grad()
    def predict_uplift(
        self,
        data: Union[DataLoader, torch.Tensor, np.ndarray, pd.DataFrame],
        t: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        n_samples: int = 10,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Dự đoán uplift score tau = y1 - y0."""
        self.model.eval()
        uplift_preds = []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0].to(self.device)
                if len(batch) >= 3:
                    t_b = batch[1].to(self.device)
                    y_b = batch[2].to(self.device)
                    y0_mean, y1_mean = self.model.predict_uplift(x_b, t_b, y_b, n_samples=n_samples)
                else:
                    y0_acc = torch.zeros(x_b.size(0), device=self.device)
                    y1_acc = torch.zeros(x_b.size(0), device=self.device)
                    for _ in range(n_samples):
                        z = torch.randn(x_b.size(0), self.model.z_dim, device=self.device)
                        _, _, _, y0_logit, y1_logit = self.model.decoder(z)
                        y0_acc = y0_acc + torch.sigmoid(y0_logit)
                        y1_acc = y1_acc + torch.sigmoid(y1_logit)
                    y0_mean = y0_acc / n_samples
                    y1_mean = y1_acc / n_samples
                uplift_preds.append((y1_mean - y0_mean).cpu().numpy())
        else:
            X_arr = _to_numpy(data)
            has_obs = t is not None and y is not None
            if has_obs:
                t_arr = _to_numpy(t)
                y_arr = _to_numpy(y)

            n_samples_total = len(X_arr)
            for offset in range(0, n_samples_total, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                if has_obs:
                    batch_t = torch.from_numpy(t_arr[offset : offset + batch_size]).to(self.device)
                    batch_y = torch.from_numpy(y_arr[offset : offset + batch_size]).to(self.device)
                    y0_mean, y1_mean = self.model.predict_uplift(batch_x, batch_t, batch_y, n_samples=n_samples)
                else:
                    y0_acc = torch.zeros(batch_x.size(0), device=self.device)
                    y1_acc = torch.zeros(batch_x.size(0), device=self.device)
                    for _ in range(n_samples):
                        z = torch.randn(batch_x.size(0), self.model.z_dim, device=self.device)
                        _, _, _, y0_logit, y1_logit = self.model.decoder(z)
                        y0_acc = y0_acc + torch.sigmoid(y0_logit)
                        y1_acc = y1_acc + torch.sigmoid(y1_logit)
                    y0_mean = y0_acc / n_samples
                    y1_mean = y1_acc / n_samples
                uplift_preds.append((y1_mean - y0_mean).cpu().numpy())

        return np.concatenate(uplift_preds, axis=0)

    @torch.no_grad()
    def predict_outcomes(
        self,
        data: Union[DataLoader, torch.Tensor, np.ndarray, pd.DataFrame],
        t: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        n_samples: int = 10,
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán potential outcomes (y0_prob, y1_prob)."""
        self.model.eval()
        y0_preds, y1_preds = [], []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0].to(self.device)
                if len(batch) >= 3:
                    t_b = batch[1].to(self.device)
                    y_b = batch[2].to(self.device)
                    y0_mean, y1_mean = self.model.predict_uplift(x_b, t_b, y_b, n_samples=n_samples)
                else:
                    y0_acc = torch.zeros(x_b.size(0), device=self.device)
                    y1_acc = torch.zeros(x_b.size(0), device=self.device)
                    for _ in range(n_samples):
                        z = torch.randn(x_b.size(0), self.model.z_dim, device=self.device)
                        _, _, _, y0_logit, y1_logit = self.model.decoder(z)
                        y0_acc = y0_acc + torch.sigmoid(y0_logit)
                        y1_acc = y1_acc + torch.sigmoid(y1_logit)
                    y0_mean = y0_acc / n_samples
                    y1_mean = y1_acc / n_samples
                y0_preds.append(y0_mean.cpu().numpy())
                y1_preds.append(y1_mean.cpu().numpy())
        else:
            X_arr = _to_numpy(data)
            has_obs = t is not None and y is not None
            if has_obs:
                t_arr = _to_numpy(t)
                y_arr = _to_numpy(y)

            n_samples_total = len(X_arr)
            for offset in range(0, n_samples_total, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                if has_obs:
                    batch_t = torch.from_numpy(t_arr[offset : offset + batch_size]).to(self.device)
                    batch_y = torch.from_numpy(y_arr[offset : offset + batch_size]).to(self.device)
                    y0_mean, y1_mean = self.model.predict_uplift(batch_x, batch_t, batch_y, n_samples=n_samples)
                else:
                    y0_acc = torch.zeros(batch_x.size(0), device=self.device)
                    y1_acc = torch.zeros(batch_x.size(0), device=self.device)
                    for _ in range(n_samples):
                        z = torch.randn(batch_x.size(0), self.model.z_dim, device=self.device)
                        _, _, _, y0_logit, y1_logit = self.model.decoder(z)
                        y0_acc = y0_acc + torch.sigmoid(y0_logit)
                        y1_acc = y1_acc + torch.sigmoid(y1_logit)
                    y0_mean = y0_acc / n_samples
                    y1_mean = y1_acc / n_samples
                y0_preds.append(y0_mean.cpu().numpy())
                y1_preds.append(y1_mean.cpu().numpy())

        return np.concatenate(y0_preds, axis=0), np.concatenate(y1_preds, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
