"""
trainer.py — Trainer / Wrapper cho DESCN (Deep Entire Space Cross Networks)
=============================================================================
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from .model import DESCNModel, gaussian_mmd
except ImportError:
    from baselines.descn.model import DESCNModel, gaussian_mmd

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


class DESCN:
    """
    Trainer / Wrapper cho mô hình DESCN.
    Nhận DataLoader trực tiếp từ bước preprocess.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        input_dim: int = 12,
        share_dim: int = 128,
        base_dim: int = 64,
        do_rate: float = 0.1,
        use_bn: bool = True,
        normalization: str = "divide",
        h1_w: float = 0.5,
        h0_w: float = 0.1,
        imb_dist_w: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = DESCNModel(
                input_dim=input_dim,
                share_dim=share_dim,
                base_dim=base_dim,
                do_rate=do_rate,
                use_bn=use_bn,
                normalization=normalization,
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.h1_w = h1_w
        self.h0_w = h0_w
        self.imb_dist_w = imb_dist_w

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Tính multi-task loss cho DESCN:
            L = h1_w * BCE(p_h1[T=1], y[T=1])
              + h0_w * BCE(p_h0[T=0], y[T=0])
              + imb_dist_w * MMD(shared_h[T=1], shared_h[T=0])
        """
        outputs = self.model(x)
        p_h1 = outputs[9]
        p_h0 = outputs[10]
        shared_h = outputs[11]

        t_mask = (t.view(-1, 1) == 1)
        c_mask = (t.view(-1, 1) == 0)
        y_col = y.view(-1, 1).float()

        total_loss = torch.tensor(0.0, device=self.device)

        if self.h1_w > 0 and t_mask.any():
            h1_loss = F.binary_cross_entropy(p_h1[t_mask], y_col[t_mask])
            total_loss = total_loss + self.h1_w * h1_loss

        if self.h0_w > 0 and c_mask.any():
            h0_loss = F.binary_cross_entropy(p_h0[c_mask], y_col[c_mask])
            total_loss = total_loss + self.h0_w * h0_loss

        if self.imb_dist_w > 0:
            mmd_loss = gaussian_mmd(shared_h, t)
            total_loss = total_loss + self.imb_dist_w * mmd_loss

        return total_loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Huấn luyện 1 epoch và trả về mean multi-task loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_b, t_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            t_b = t_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(x_b, t_b, y_b)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Validation: Tính nhanh loss trên val_loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_b, t_b, y_b in val_loader:
                x_b = x_b.to(self.device)
                t_b = t_b.to(self.device)
                y_b = y_b.to(self.device)

                loss = self.compute_loss(x_b, t_b, y_b)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        test_loader: DataLoader,
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

                loss = self.compute_loss(x_b_dev, t_b_dev, y_b_dev)
                total_loss += loss.item()
                n_batches += 1

                outputs = self.model(x_b_dev)
                p_mu1 = outputs[7]
                p_mu0 = outputs[8]
                uplift_b = (p_mu1 - p_mu0).squeeze(-1).cpu().numpy()

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
        model_name: str = "descn",
        writer: Optional[Any] = None,
        verbose: int = 1,
    ) -> Dict[str, list]:
        """Huấn luyện mô hình DESCN bằng DataLoaders truyền vào trực tiếp."""
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            logger.info(
                f"Starting DESCN training | Device: {self.device} | Epochs: {epochs}"
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
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Dự đoán uplift score τ(x) = p_mu1 - p_mu0."""
        self.model.eval()
        uplift_preds = []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_b = x_b.to(self.device)
                outputs = self.model(x_b)
                p_mu1 = outputs[7]
                p_mu0 = outputs[8]
                uplift_preds.append((p_mu1 - p_mu0).squeeze(-1).cpu().numpy())
        else:
            X_arr = _to_numpy(data)
            n_samples = len(X_arr)
            for offset in range(0, n_samples, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                outputs = self.model(batch_x)
                p_mu1 = outputs[7]
                p_mu0 = outputs[8]
                uplift_preds.append((p_mu1 - p_mu0).squeeze(-1).cpu().numpy())

        return np.concatenate(uplift_preds, axis=0)

    @torch.no_grad()
    def predict_outcomes(
        self,
        data: Union[DataLoader, torch.Tensor, np.ndarray, pd.DataFrame],
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán potential outcomes (p_mu0, p_mu1)."""
        self.model.eval()
        y0_preds, y1_preds = [], []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_b = x_b.to(self.device)
                outputs = self.model(x_b)
                p_mu1 = outputs[7].squeeze(-1).cpu().numpy()
                p_mu0 = outputs[8].squeeze(-1).cpu().numpy()
                y0_preds.append(p_mu0)
                y1_preds.append(p_mu1)
        else:
            X_arr = _to_numpy(data)
            n_samples = len(X_arr)
            for offset in range(0, n_samples, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                outputs = self.model(batch_x)
                p_mu1 = outputs[7].squeeze(-1).cpu().numpy()
                p_mu0 = outputs[8].squeeze(-1).cpu().numpy()
                y0_preds.append(p_mu0)
                y1_preds.append(p_mu1)

        return np.concatenate(y0_preds, axis=0), np.concatenate(y1_preds, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


# Aliases tương đương
Descn = DESCN
DESCNTrainer = DESCN
DESCNWrapper = DESCN
DescnWrapper = DESCN
