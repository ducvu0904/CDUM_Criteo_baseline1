"""
trainer.py — Trainer / Wrapper cho GANITE (Generative Adversarial Nets for ITE)
================================================================================
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
    from .model import GANITEGenerator, GANITEDiscriminator, GANITEInferenceNet
except ImportError:
    from baselines.ganite.model import GANITEGenerator, GANITEDiscriminator, GANITEInferenceNet

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


class GANITE:
    """
    Trainer / Wrapper cho mô hình GANITE 2-phase.
    Nhận DataLoader trực tiếp từ bước preprocess.
    """

    def __init__(
        self,
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        inference_net: Optional[nn.Module] = None,
        input_dim: int = 12,
        h_dim: int = 64,
        alpha: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.generator = (
            generator.to(self.device) if generator is not None
            else GANITEGenerator(input_dim=input_dim, h_dim=h_dim).to(self.device)
        )
        self.discriminator = (
            discriminator.to(self.device) if discriminator is not None
            else GANITEDiscriminator(input_dim=input_dim, h_dim=h_dim).to(self.device)
        )
        self.inference_net = (
            inference_net.to(self.device) if inference_net is not None
            else GANITEInferenceNet(input_dim=input_dim, h_dim=h_dim).to(self.device)
        )

        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.i_optimizer = torch.optim.Adam(
            self.inference_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.i_optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

    def train_gan_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Huấn luyện 1 epoch Phase 1 (G và D)."""
        self.generator.train()
        self.discriminator.train()

        total_d, total_g = 0.0, 0.0
        n_batches = 0

        for x_b, t_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            t_b = t_b.to(self.device).view(-1, 1).float()
            y_b = y_b.to(self.device).view(-1, 1).float()

            # 1. Update Discriminator
            with torch.no_grad():
                gen_logits = self.generator(x_b, t_b, y_b)
            d_logit = self.discriminator(x_b, t_b, y_b, gen_logits)
            d_loss = F.binary_cross_entropy_with_logits(d_logit, t_b)

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # 2. Update Generator
            gen_logits = self.generator(x_b, t_b, y_b)
            d_logit = self.discriminator(x_b, t_b, y_b, gen_logits)
            g_loss_gan = -F.binary_cross_entropy_with_logits(d_logit, t_b)

            factual_logit = t_b * gen_logits[:, 1:2] + (1.0 - t_b) * gen_logits[:, 0:1]
            g_loss_factual = F.binary_cross_entropy_with_logits(factual_logit, y_b)
            g_loss = g_loss_factual + self.alpha * g_loss_gan

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            total_d += d_loss.item()
            total_g += g_loss.item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return total_d / n_batches, total_g / n_batches

    def train_inference_epoch(self, dataloader: DataLoader) -> float:
        """Huấn luyện 1 epoch Phase 2 (InferenceNet)."""
        self.generator.eval()
        self.inference_net.train()

        total_loss = 0.0
        n_batches = 0

        for x_b, t_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            t_b = t_b.to(self.device).view(-1, 1).float()
            y_b = y_b.to(self.device).view(-1, 1).float()

            with torch.no_grad():
                g_logits = self.generator(x_b, t_b, y_b)
                g_prob = torch.sigmoid(g_logits)
                label_y1 = t_b * y_b + (1.0 - t_b) * g_prob[:, 1:2]
                label_y0 = (1.0 - t_b) * y_b + t_b * g_prob[:, 0:1]

            i_logits = self.inference_net(x_b)
            i_loss1 = F.binary_cross_entropy_with_logits(i_logits[:, 1:2], label_y1)
            i_loss0 = F.binary_cross_entropy_with_logits(i_logits[:, 0:1], label_y0)
            i_loss = i_loss1 + i_loss0

            self.i_optimizer.zero_grad()
            i_loss.backward()
            self.i_optimizer.step()

            total_loss += i_loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Validation: Tính factual BCE loss của InferenceNet trên val_loader."""
        self.inference_net.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_b, t_b, y_b in val_loader:
                x_b = x_b.to(self.device)
                t_b = t_b.to(self.device).view(-1, 1).float()
                y_b = y_b.to(self.device).view(-1, 1).float()

                i_logits = self.inference_net(x_b)
                factual_logit = t_b * i_logits[:, 1:2] + (1.0 - t_b) * i_logits[:, 0:1]
                loss = F.binary_cross_entropy_with_logits(factual_logit, y_b)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        test_loader: DataLoader,
        k: float = 0.3,
    ) -> Dict[str, float]:
        """Evaluation: Đánh giá toàn diện trên test_loader từ bước preprocess."""
        self.inference_net.eval()
        total_loss = 0.0
        n_batches = 0
        all_uplifts, all_t, all_y = [], [], []

        with torch.no_grad():
            for x_b, t_b, y_b in test_loader:
                x_b_dev = x_b.to(self.device)
                t_b_dev = t_b.to(self.device).view(-1, 1).float()
                y_b_dev = y_b.to(self.device).view(-1, 1).float()

                i_logits = self.inference_net(x_b_dev)
                factual_logit = t_b_dev * i_logits[:, 1:2] + (1.0 - t_b_dev) * i_logits[:, 0:1]
                loss = F.binary_cross_entropy_with_logits(factual_logit, y_b_dev)
                total_loss += loss.item()
                n_batches += 1

                i_prob = torch.sigmoid(i_logits)
                uplift_b = (i_prob[:, 1] - i_prob[:, 0]).cpu().numpy()

                all_uplifts.append(uplift_b)
                all_t.append(t_b.view(-1).cpu().numpy())
                all_y.append(y_b.view(-1).cpu().numpy())

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
        epochs_gan: int = 5,
        epochs_inf: int = 5,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "ganite",
        writer: Optional[Any] = None,
        verbose: int = 1,
    ) -> Dict[str, list]:
        """Huấn luyện GANITE cả 2 phase bằng DataLoaders truyền vào trực tiếp."""
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            logger.info(f"=== [Phase 1/2] GAN Training ({epochs_gan} epochs) ===")
        for epoch in range(1, epochs_gan + 1):
            d_loss, g_loss = self.train_gan_epoch(train_loader)
            if writer is not None:
                writer.add_scalar("GAN/d_loss", d_loss, epoch)
                writer.add_scalar("GAN/g_loss", g_loss, epoch)
            if verbose:
                logger.info(f"  [GAN] Epoch [{epoch:02d}/{epochs_gan:02d}] D_loss: {d_loss:.5f} | G_loss: {g_loss:.5f}")

        if verbose:
            logger.info(f"=== [Phase 2/2] InferenceNet Training ({epochs_inf} epochs) ===")
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs_inf + 1):
            inf_loss = self.train_inference_epoch(train_loader)
            history["train_loss"].append(inf_loss)

            if writer is not None:
                writer.add_scalar("Loss/train", inf_loss, epoch)

            log_msg = f"  [Inf] Epoch [{epoch:02d}/{epochs_inf:02d}] Train Loss: {inf_loss:.5f}"

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                log_msg += f" | Val Loss: {val_loss:.5f}"

                current_lr = self.i_optimizer.param_groups[0]["lr"]
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
                            logger.info(f"    -> Saved best checkpoint: {ckpt_best}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            logger.info(f"Early stopping triggered in Inference Phase at epoch {epoch}!")
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
        """Dự đoán uplift score τ(x) = y1 - y0 từ InferenceNet."""
        self.inference_net.eval()
        uplift_preds = []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_b = x_b.to(self.device)
                i_logits = self.inference_net(x_b)
                i_prob = torch.sigmoid(i_logits)
                uplift_preds.append((i_prob[:, 1] - i_prob[:, 0]).cpu().numpy())
        else:
            X_arr = _to_numpy(data)
            n_samples = len(X_arr)
            for offset in range(0, n_samples, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                i_logits = self.inference_net(batch_x)
                i_prob = torch.sigmoid(i_logits)
                uplift_preds.append((i_prob[:, 1] - i_prob[:, 0]).cpu().numpy())

        return np.concatenate(uplift_preds, axis=0)

    @torch.no_grad()
    def predict_outcomes(
        self,
        data: Union[DataLoader, torch.Tensor, np.ndarray, pd.DataFrame],
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán potential outcomes (y0_prob, y1_prob)."""
        self.inference_net.eval()
        y0_preds, y1_preds = [], []

        if isinstance(data, DataLoader):
            for batch in data:
                x_b = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_b = x_b.to(self.device)
                i_logits = self.inference_net(x_b)
                i_prob = torch.sigmoid(i_logits)
                y0_preds.append(i_prob[:, 0].cpu().numpy())
                y1_preds.append(i_prob[:, 1].cpu().numpy())
        else:
            X_arr = _to_numpy(data)
            n_samples = len(X_arr)
            for offset in range(0, n_samples, batch_size):
                batch_x = torch.from_numpy(X_arr[offset : offset + batch_size]).to(self.device)
                i_logits = self.inference_net(batch_x)
                i_prob = torch.sigmoid(i_logits)
                y0_preds.append(i_prob[:, 0].cpu().numpy())
                y1_preds.append(i_prob[:, 1].cpu().numpy())

        return np.concatenate(y0_preds, axis=0), np.concatenate(y1_preds, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "inference_net": self.inference_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.inference_net.load_state_dict(checkpoint["inference_net"])
