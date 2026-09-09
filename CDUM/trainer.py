"""Training wrapper for CPM.

The wrapper deliberately owns optimisation only.  Uplift metrics are computed
in :meth:`evaluate`, after training, from the two potential-outcome estimates.
This keeps an observed factual outcome from being confused with an uplift
supervision signal.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from metrics.uplift_metrics import (
        qini_auc_score1,
        uplift_at_k1,
        uplift_auc_score1,
    )

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


logger = logging.getLogger(__name__)


class CPMTrainer:
    """Train a CPM model using factual-outcome Huber loss.

    ``model`` must return a mapping containing ``"y_factual"`` when called
    as ``model(x_ids, treatment)``.  The trainer intentionally never derives
    its optimisation loss from an uplift output.

    The expected dataloader batch is ``(x_ids, treatment, outcome)``, which is
    the same shape convention used by the project's other trainers.  ``x_ids``
    is passed through without a dtype conversion because CPM inputs are already
    bucketed integer feature ids by the preprocessing step.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_factor: float = 0.5,
        lr_patience: int = 2,
        min_lr: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        self.criterion = nn.HuberLoss(delta=1.0)

    def _forward(self, x_ids: torch.Tensor, treatment: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(x_ids, treatment)
        if not isinstance(outputs, dict):
            raise TypeError("CPM model.forward must return a dictionary of outputs")
        if "y_factual" not in outputs:
            raise KeyError("CPM model output must contain 'y_factual' for factual-outcome training")
        return outputs

    def compute_loss(
        self, x_ids: torch.Tensor, treatment: torch.Tensor, outcome: torch.Tensor
    ) -> torch.Tensor:
        """Return Huber loss between observed outcomes and ``outputs['y_factual']``."""
        outputs = self._forward(x_ids, treatment)
        y_factual = outputs["y_factual"]
        target = outcome.to(dtype=y_factual.dtype).reshape_as(y_factual)
        return self.criterion(y_factual, target)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Optimise factual Huber loss for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_ids, treatment, outcome in dataloader:
            x_ids = x_ids.to(self.device)
            treatment = treatment.to(self.device)
            outcome = outcome.to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(x_ids, treatment, outcome)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Compute factual Huber loss without updating model parameters."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_ids, treatment, outcome in val_loader:
                x_ids = x_ids.to(self.device)
                treatment = treatment.to(self.device)
                outcome = outcome.to(self.device)

                loss = self.compute_loss(x_ids, treatment, outcome)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "cpm",
        writer: Optional[Any] = None,
        verbose: int = 1,
    ) -> Dict[str, list]:
        """Train CPM and save ``*_best.pth`` and ``*_final.pth`` checkpoints.

        Validation loss is used for scheduling, early stopping, and best-model
        selection.  If no validation loader is supplied, training loss is used
        solely as a fallback monitor; no uplift metric participates in either
        optimisation or model selection.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_monitor_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if verbose:
            logger.info("Starting CPM training | Device: %s | Epochs: %s", self.device, epochs)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                monitor_loss = val_loss
                history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)
            else:
                # Keep the history aligned by epoch while making the absence of
                # validation explicit.  This is not used as a validation score.
                val_loss = float("nan")
                monitor_loss = train_loss
                history["val_loss"].append(val_loss)
                self.scheduler.step(train_loss)

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loader is not None:
                    writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], epoch)

            if monitor_loss < best_monitor_loss:
                best_monitor_loss = monitor_loss
                patience_counter = 0
                if checkpoint_dir is not None:
                    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
                    self.save(best_path)
                    if verbose >= 2:
                        logger.info("  -> Saved best checkpoint: %s", best_path)
            else:
                patience_counter += 1

            if verbose:
                val_text = f" | Val Loss: {val_loss:.5f}" if val_loader is not None else ""
                logger.info(
                    "Epoch [%02d/%02d]  Train Loss: %.5f%s | LR: %.6f | EarlyStop: %d/%d",
                    epoch,
                    epochs,
                    train_loss,
                    val_text,
                    self.optimizer.param_groups[0]["lr"],
                    patience_counter,
                    early_stopping_patience,
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                break

        if checkpoint_dir is not None:
            final_path = os.path.join(checkpoint_dir, f"{model_name}_final.pth")
            self.save(final_path)
            if verbose >= 2:
                logger.info("  -> Saved final checkpoint: %s", final_path)

            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            if os.path.exists(best_path):
                self.load(best_path)

        return history

    def _potential_outcomes(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read potential outcomes for metric calculation, never for training."""
        if "y0_hat" in outputs and "y1_hat" in outputs:
            return outputs["y0_hat"], outputs["y1_hat"]
        if "y0" in outputs and "y1" in outputs:
            return outputs["y0"], outputs["y1"]
        raise KeyError("CPM model output must contain y0_hat/y1_hat (or y0/y1) for uplift metrics")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, k: float = 0.3) -> Dict[str, float]:
        """Evaluate factual loss and uplift metrics, separately from optimisation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        uplift_scores, treatments, outcomes = [], [], []

        for x_ids, treatment, outcome in test_loader:
            x_ids_dev = x_ids.to(self.device)
            treatment_dev = treatment.to(self.device)
            outcome_dev = outcome.to(self.device)

            outputs = self._forward(x_ids_dev, treatment_dev)
            y_factual = outputs["y_factual"]
            target = outcome_dev.to(dtype=y_factual.dtype).reshape_as(y_factual)
            total_loss += self.criterion(y_factual, target).item()
            n_batches += 1

            y0_hat, y1_hat = self._potential_outcomes(outputs)
            uplift_scores.append((y1_hat - y0_hat).reshape(-1).cpu().numpy())
            treatments.append(treatment.reshape(-1).cpu().numpy())
            outcomes.append(outcome.reshape(-1).cpu().numpy())

        results: Dict[str, float] = {"loss": total_loss / max(n_batches, 1)}
        if not HAS_METRICS or not uplift_scores:
            return results

        uplift_arr = np.concatenate(uplift_scores)
        treatment_arr = np.concatenate(treatments)
        outcome_arr = np.concatenate(outcomes)
        try:
            results["auuc"] = float(uplift_auc_score1(outcome_arr, uplift_arr, treatment_arr))
        except Exception as exc:
            logger.warning("Failed to compute AUUC: %s", exc)
        try:
            results["qini"] = float(qini_auc_score1(outcome_arr, uplift_arr, treatment_arr))
        except Exception as exc:
            logger.warning("Failed to compute Qini: %s", exc)
        try:
            results[f"lift@{int(k * 100)}%"] = float(
                uplift_at_k1(outcome_arr, uplift_arr, treatment_arr, strategy="overall", k=k)
            )
        except Exception as exc:
            logger.warning("Failed to compute Uplift@%s: %s", k, exc)
        return results

    def save(self, path: str) -> None:
        """Save model weights using the checkpoint convention of other trainers."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights saved by :meth:`save`."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


# Short alias follows the model-wrapper naming used by the baseline packages.
CPM = CPMTrainer
