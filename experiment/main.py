"""
main.py — Entry point để chạy thí nghiệm các baseline uplift models trên Criteo
=================================================================================
Sử dụng:
    python experiment/main.py --config experiment/config.yaml
    python experiment/main.py --model tarnet --data path/to/criteo.csv
    python experiment/main.py --config experiment/config.yaml --model cfrnet
    python experiment/main.py --model all --epochs 20 --batch_size 4096

Chạy song song (Parallel Runs để giảm thời gian huấn luyện):
    python experiment/main.py --model all --parallel 2
    python experiment/main.py --models tarnet cfrnet dragonnet --parallel 3
    python experiment/main.py --model tarnet,cfrnet --parallel 2 --gpus 0

Models có thể chọn (--model / --models):
    tarnet | cevae | descn | euen | ganite | dragonnet | cfrnet | efin | all

Ưu tiên: CLI args > config.yaml > default values

Chi tiết các tùy chọn:
    python experiment/main.py --help
"""

import os
import sys
import json
import yaml
import time
import shutil
import random
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

import numpy as np
import pandas as pd
import torch

# ── Đảm bảo project root nằm trong sys.path ──────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.data_loader import get_dataloaders
from baselines import TARNET, CEVAE, DESCN, EUEN, GANITE, DRAGONNET, CFRNET, EFIN, SLEARNER, TLEARNER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

ALL_MODELS = ["tarnet", "cevae", "descn", "euen", "ganite", "dragonnet", "cfrnet", "efin", "slearner", "tlearner"]



# ══════════════════════════════════════════════════════════════════════════════
# Config YAML helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml_config(config_path: str) -> dict:
    """Load file YAML và trả về dict phẳng với các key tương ứng args."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    section_map = {
        "data":      {"path": "data", "train_path": "train_path", "val_path": "val_path",
                      "test_path": "test_path", "label_col": "label_col", "num_workers": "num_workers",
                      "test_size": "test_size", "val_ratio": "val_ratio"},
        "model":     {"name": "model", "input_dim": "input_dim", "shared_dim": "shared_dim", "head_dim": "head_dim"},
        "training":  {"epochs": "epochs", "batch_size": "batch_size", "lr": "lr",
                      "lr_factor": "lr_factor", "lr_patience": "lr_patience", "min_lr": "min_lr",
                      "weight_decay": "weight_decay", "patience": "patience", "device": "device", "seeds": "seeds",
                      "seed": "seed", "parallel": "parallel", "gpus": "gpus"},
        "efin":      {"embed_dim": "efin_embed_dim", "lambda_c": "efin_lambda_c", "loss_type": "efin_loss_type"},
        "cfrnet":    {"ipm_mode": "ipm_mode", "lambda_ipm": "lambda_ipm"},
        "dragonnet": {"alpha": "alpha", "beta": "beta"},
        "ganite":    {"h_dim": "h_dim", "epochs_gan": "epochs_gan", "epochs_inf": "epochs_inf", "alpha": "gan_alpha"},
        "cevae":     {"z_dim": "z_dim", "n_hidden": "n_hidden"},
        "output":    {"checkpoint_dir": "checkpoint_dir", "results_dir": "results_dir",
                      "run_name": "run_name", "eval_k": "eval_k", "verbose": "verbose"},
    }
    for section, mapping in section_map.items():
        if section in cfg and cfg[section]:
            for yaml_key, arg_key in mapping.items():
                if yaml_key in cfg[section] and cfg[section][yaml_key] is not None:
                    flat[arg_key] = cfg[section][yaml_key]
    return flat


def merge_config_into_args(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    """
    Merge config.yaml vào args.
    CLI args (đã được set tường minh) luôn có ưu tiên cao hơn config.
    """
    yaml_cfg = load_yaml_config(config_path)

    # Lấy tập các arg mà CLI đã cung cấp tường minh (khác với default)
    parser = build_parser()
    defaults = vars(parser.parse_args([]))  # parse với empty args -> full defaults
    cli_provided = {k for k, v in vars(args).items() if v != defaults.get(k)}

    for key, value in yaml_cfg.items():
        if key not in cli_provided:          # chỉ ghi đè nếu CLI không cung cấp
            setattr(args, key, value)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Criteo Uplift Modeling — Baseline Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (e.g. experiment/config.yaml). "
                        "CLI args override config values.")

    # ── Data ──────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument("--data", type=str, default="/home/ducvu0904/Documents/dataset/Criteo",
                      help="Path to directory containing pre-split Criteo datasets (or explicit split file).")
    data.add_argument("--train_path", type=str, default=None,
                      help="Optional explicit path to train split (.pt or .csv).")
    data.add_argument("--val_path", type=str, default=None,
                      help="Optional explicit path to val split (.pt or .csv).")
    data.add_argument("--test_path", type=str, default=None,
                      help="Optional explicit path to test split (.pt or .csv).")
    data.add_argument("--label_col", type=str, default="visit", choices=["visit", "conversion"],
                      help="Target outcome column ('visit' or 'conversion').")
    data.add_argument("--seed", type=int, default=None,
                      help="Optional single seed. If set, overrides --seeds with [seed].")

    # ── Model ─────────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--model", type=str, default="tarnet",
                     help="Model to train. Options: tarnet | cevae | descn | euen | ganite | dragonnet | cfrnet | efin | slearner | tlearner | all. "
                          "Can also be comma-separated, e.g. 'tarnet,cfrnet'.")
    mdl.add_argument("--models", type=str, nargs="+", default=None,
                     help="Optional list of multiple models to run (e.g. --models tarnet cfrnet dragonnet). Overrides --model.")
    mdl.add_argument("--input_dim", type=int, default=12,
                     help="Number of input features (12 for Criteo f0..f11).")
    mdl.add_argument("--shared_dim", type=int, default=200,
                     help="Shared representation dimension.")
    mdl.add_argument("--head_dim", type=int, default=100,
                     help="Outcome head hidden dimension.")

    # ── EFIN-specific ─────────────────────────────────────────────────────────
    efn = p.add_argument_group("EFIN options")
    efn.add_argument("--efin_embed_dim", type=int, default=64,
                     help="Embedding dimension K_d for EFIN.")
    efn.add_argument("--efin_lambda_c", type=float, default=0.01,
                     help="Intervention constraint loss weight for EFIN.")
    efn.add_argument("--efin_loss_type", type=str, default="bce", choices=["bce", "mse"],
                     help="Loss type for EFIN ('bce' or 'mse').")

    # ── CFRNET-specific ───────────────────────────────────────────────────────
    cfr = p.add_argument_group("CFRNet options")
    cfr.add_argument("--ipm_mode", type=str, default="wass", choices=["mmd", "wass"],
                     help="IPM type for CFRNET: 'mmd' or 'wass'.")
    cfr.add_argument("--lambda_ipm", type=float, default=1.0,
                     help="IPM regularization weight for CFRNET.")

    # ── DRAGONNET-specific ────────────────────────────────────────────────────
    drg = p.add_argument_group("DragonNet options")
    drg.add_argument("--alpha", type=float, default=1.0,
                     help="Propensity loss weight for DRAGONNET.")
    drg.add_argument("--beta", type=float, default=1.0,
                     help="Targeted regularization weight for DRAGONNET.")

    # ── GANITE-specific ───────────────────────────────────────────────────────
    gan = p.add_argument_group("GANITE options")
    gan.add_argument("--h_dim", type=int, default=64,
                     help="Hidden dimension for GANITE generator/discriminator/inference.")
    gan.add_argument("--epochs_gan", type=int, default=5,
                     help="Number of GAN training epochs (Phase 1) for GANITE.")
    gan.add_argument("--epochs_inf", type=int, default=None,
                     help="Number of InferenceNet epochs (Phase 2) for GANITE. Defaults to --epochs.")
    gan.add_argument("--gan_alpha", type=float, default=1.0,
                     help="Adversarial loss weight for GANITE generator.")

    # ── CEVAE-specific ────────────────────────────────────────────────────────
    vae = p.add_argument_group("CEVAE options")
    vae.add_argument("--z_dim", type=int, default=20,
                     help="Latent dimension z for CEVAE.")
    vae.add_argument("--n_hidden", type=int, default=3,
                     help="Number of hidden layers for CEVAE sub-networks.")

    # ── Training ──────────────────────────────────────────────────────────────
    trn = p.add_argument_group("Training")
    trn.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    trn.add_argument("--batch_size", type=int, default=4096, help="Batch size.")
    trn.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    trn.add_argument("--lr_factor", type=float, default=0.5,
                     help="Factor by which the learning rate will be reduced (ReduceLROnPlateau).")
    trn.add_argument("--lr_patience", type=int, default=2,
                     help="Number of epochs with no improvement after which learning rate will be reduced.")
    trn.add_argument("--min_lr", type=float, default=1e-6,
                     help="Minimum learning rate for ReduceLROnPlateau.")
    trn.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay.")
    trn.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    trn.add_argument("--device", type=str, default=None,
                     help="Device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detect if not set.")
    trn.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                     help="List of random seeds to evaluate model stability (default: 1 2 3 4 5).")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                     help="Root directory to save best model checkpoints.")
    out.add_argument("--results_dir", type=str, default="results",
                     help="Root directory to save evaluation JSON results.")
    out.add_argument("--run_name", type=str, default=None,
                     help="Optional run name. Defaults to model name + timestamp.")
    out.add_argument("--eval_k", type=float, default=0.3,
                     help="Fraction k for Uplift@k evaluation (default 0.3 = 30%%).")
    out.add_argument("--verbose", type=int, default=1,
                     help="Verbosity level: 0=silent, 1=epoch log, 2=+checkpoint log.")

    # ── Parallel Execution ───────────────────────────────────────────────────
    par = p.add_argument_group("Parallel Execution")
    par.add_argument("--parallel", "--n_jobs", dest="parallel", type=int, default=1,
                     help="Number of models to run concurrently in parallel (default: 1 = sequential).")
    par.add_argument("--gpus", type=str, nargs="+", default=None,
                     help="List of GPU device IDs to distribute parallel runs across (e.g. --gpus 0 1 or --gpus 0). "
                          "Auto-detects available GPUs by default.")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, args: argparse.Namespace):
    """Khởi tạo trainer tương ứng với --model."""
    common = dict(
        input_dim=args.input_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        lr_factor=getattr(args, "lr_factor", 0.5),
        lr_patience=getattr(args, "lr_patience", 2),
        min_lr=getattr(args, "min_lr", 1e-6),
    )

    if model_name == "tarnet":
        return TARNET(shared_dim=args.shared_dim, head_dim=args.head_dim, **common)

    elif model_name == "cevae":
        return CEVAE(
            z_dim=args.z_dim,
            hidden_dim=args.shared_dim,
            n_hidden=args.n_hidden,
            **common
        )

    elif model_name == "descn":
        return DESCN(
            share_dim=args.shared_dim,
            base_dim=args.head_dim,
            **common
        )

    elif model_name == "euen":
        return EUEN(
            hc_dim=args.shared_dim,
            hu_dim=args.shared_dim,
            **common
        )

    elif model_name == "ganite":
        return GANITE(
            h_dim=args.h_dim,
            alpha=args.gan_alpha,
            **common
        )

    elif model_name == "dragonnet":
        return DRAGONNET(
            shared_dim=args.shared_dim,
            head_dim=args.head_dim,
            alpha=args.alpha,
            beta=args.beta,
            **common
        )

    elif model_name == "cfrnet":
        return CFRNET(
            shared_dim=args.shared_dim,
            head_dim=args.head_dim,
            mode=args.ipm_mode,
            lambda_ipm=args.lambda_ipm,
            **common
        )

    elif model_name == "efin":
        return EFIN(
            embed_dim=getattr(args, "efin_embed_dim", 64),
            shared_dim=args.shared_dim,
            head_dim=args.head_dim,
            lambda_c=getattr(args, "efin_lambda_c", 0.01),
            loss_type=getattr(args, "efin_loss_type", "bce"),
            **common
        )

    elif model_name == "slearner":
        return SLEARNER(
            shared_dim=args.shared_dim,
            head_dim=args.head_dim,
            **common
        )

    elif model_name == "tlearner":
        return TLEARNER(
            shared_dim=args.shared_dim,
            head_dim=args.head_dim,
            **common
        )


# ══════════════════════════════════════════════════════════════════════════════
# Seed & CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Thiết lập random seed toàn cục cho reproducibility tuyệt đối."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_summary_csv(summary_csv_path: str, model_name: str, summary_dict: dict):
    """
    Cập nhật hoặc thêm hàng kết quả summary (mean ± std) của mô hình vào results/summary.csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
    mean = summary_dict.get("mean", {})
    std = summary_dict.get("std", {})

    val_loss_m, val_loss_s = mean.get("val_loss", 0.0), std.get("val_loss", 0.0)
    auuc_m, auuc_s = mean.get("test_auuc", 0.0), std.get("test_auuc", 0.0)
    qini_m, qini_s = mean.get("test_qini", 0.0), std.get("test_qini", 0.0)
    lift_m, lift_s = mean.get("test_lift@30", 0.0), std.get("test_lift@30", 0.0)
    best_ep_m, best_ep_s = mean.get("best_epoch", 0.0), std.get("best_epoch", 0.0)

    row_data = {
        "model": model_name.lower(),
        "best_epoch": f"{best_ep_m:.1f} ± {best_ep_s:.2f}",
        "val_loss": f"{val_loss_m:.5f} ± {val_loss_s:.5f}",
        "test_auuc": f"{auuc_m:.5f} ± {auuc_s:.5f}",
        "test_qini": f"{qini_m:.5f} ± {qini_s:.5f}",
        "test_lift@30": f"{lift_m:.5f} ± {lift_s:.5f}",
        "val_loss_mean": val_loss_m,
        "val_loss_std": val_loss_s,
        "test_auuc_mean": auuc_m,
        "test_auuc_std": auuc_s,
        "test_qini_mean": qini_m,
        "test_qini_std": qini_s,
        "test_lift@30_mean": lift_m,
        "test_lift@30_std": lift_s,
    }

    lock_path = summary_csv_path + ".lock"
    lock_ctx = FileLock(lock_path, timeout=60) if FileLock is not None else nullcontext()

    with lock_ctx:
        if os.path.exists(summary_csv_path):
            try:
                df_summary = pd.read_csv(summary_csv_path)
                if "model" in df_summary.columns and model_name.lower() in df_summary["model"].values:
                    idx = df_summary[df_summary["model"] == model_name.lower()].index[0]
                    for k, v in row_data.items():
                        df_summary.loc[idx, k] = v
                else:
                    df_new = pd.DataFrame([row_data])
                    df_summary = pd.concat([df_summary, df_new], ignore_index=True)
            except Exception:
                df_summary = pd.DataFrame([row_data])
        else:
            df_summary = pd.DataFrame([row_data])

        df_summary.to_csv(summary_csv_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Run single seed
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
) -> dict:
    """Chạy huấn luyện và đánh giá mô hình cho 1 seed cụ thể."""
    set_seed(seed)
    model_lower = model_name.lower()

    # Thư mục checkpoint riêng cho seed: results/<model_name>/seed_<seed>
    seed_dir = os.path.join(args.results_dir, model_lower, f"seed_{seed}")
    tb_dir = os.path.join(args.results_dir, model_lower, f"seed_{seed}", "runs")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    logger.info(f"{'-'*60}")
    logger.info(f" Model: {model_name.upper()} | Seed: {seed} | Checkpoints: {seed_dir}")
    logger.info(f"{'-'*60}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_dir)
    except Exception as e:
        logger.warning(f"TensorBoard unavailable ({e}), continuing without TensorBoard logging.")

    trainer = build_model(model_name, args)
    t_start = time.time()

    # ── Fit ───────────────────────────────────────────────────────────────────
    fit_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping_patience=args.patience,
        checkpoint_dir=seed_dir,
        model_name=f"{model_lower}",
        writer=writer,
        verbose=args.verbose,
    )

    if model_lower == "ganite":
        fit_kwargs["epochs_gan"] = args.epochs_gan
        fit_kwargs["epochs_inf"] = args.epochs_inf or args.epochs
    else:
        fit_kwargs["epochs"] = args.epochs

    history = trainer.fit(**fit_kwargs)
    elapsed = time.time() - t_start

    # ── Lưu best_checkpoint.pth và last_checkpoint.pth ────────────────────────
    best_src = os.path.join(seed_dir, f"{model_lower}_best.pth")
    last_src = os.path.join(seed_dir, f"{model_lower}_final.pth")
    best_dst = os.path.join(seed_dir, "best_checkpoint.pth")
    last_dst = os.path.join(seed_dir, "last_checkpoint.pth")

    if os.path.exists(best_src):
        shutil.move(best_src, best_dst)
    if os.path.exists(last_src):
        shutil.move(last_src, last_dst)

    # ── Đánh giá trên test set ────────────────────────────────────────────────
    logger.info(f"Evaluating {model_name.upper()} (Seed {seed}) on test set...")
    metrics = trainer.evaluate(test_loader, k=args.eval_k)

    if writer is not None:
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)):
                writer.add_scalar(f"Test/{metric_name}", val, 0)
        writer.close()

    # ── Thu thập metrics cho seed này ─────────────────────────────────────────
    val_loss_hist = history.get("val_loss", [])
    if val_loss_hist and len(val_loss_hist) > 0:
        best_idx = int(np.argmin(val_loss_hist))
        best_epoch = best_idx + 1
        best_val_loss = float(val_loss_hist[best_idx])
    else:
        best_epoch = args.epochs
        best_val_loss = float("nan")

    auuc_val = float(metrics.get("auuc", 0.0))
    qini_val = float(metrics.get("qini", 0.0))
    lift_val = float(metrics.get("lift@30%", metrics.get(f"lift@{int(args.eval_k*100)}%", 0.0)))
    test_loss_val = float(metrics.get("loss", 0.0))

    seed_result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": round(best_val_loss, 6),
        "test_auuc": round(auuc_val, 6),
        "test_qini": round(qini_val, 6),
        "test_lift@30": round(lift_val, 6),
        "test_loss": round(test_loss_val, 6),
        "train_time_s": round(elapsed, 2),
    }

    logger.info(
        f"Seed {seed} Done | Best Epoch: {best_epoch} | Val Loss: {best_val_loss:.5f} "
        f"| Test AUUC: {auuc_val:.5f} | Test Qini: {qini_val:.5f} | Test Lift@30: {lift_val:.5f} "
        f"| Time: {elapsed:.1f}s"
    )
    return seed_result


# ══════════════════════════════════════════════════════════════════════════════
# Run multi-seed experiment for a model
# ══════════════════════════════════════════════════════════════════════════════

def run_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
) -> dict:
    """
    Chạy thí nghiệm cho một mô hình qua danh sách seeds (mặc định: 1-5).
    Lưu trữ cấu trúc chuẩn:
      - results/<model>/seed_<seed>/best_checkpoint.pth & last_checkpoint.pth
      - results/<model>/config.json
      - results/<model>/metrics.json (mỗi seed + mean/std summary)
      - results/summary.csv
    """
    model_lower = model_name.lower()
    model_dir = os.path.join(args.results_dir, model_lower)
    os.makedirs(model_dir, exist_ok=True)

    seeds = args.seeds
    if isinstance(seeds, int):
        seeds = [seeds]

    logger.info(f"\n{'='*70}")
    logger.info(f" MODEL: {model_name.upper()} | SEEDS: {seeds}")
    logger.info(f" Directory: {model_dir}")
    logger.info(f"{'='*70}")

    # 1. Lưu config.json riêng biệt (hyperparameters & options)
    config_dict = vars(args).copy()
    config_dict["model_name"] = model_lower
    config_dict["run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Hyperparameters saved to: {config_path}")

    # 2. Chạy từng seed
    raw_results = {}
    for i, s in enumerate(seeds, start=1):
        logger.info(f"\n>>> Running {model_name.upper()} — Seed {s} ({i}/{len(seeds)}) ...")
        res = run_single_seed(model_name, s, args, train_loader, val_loader, test_loader)
        raw_results[str(s)] = res

    # 3. Tính toán summary (mean và std)
    metric_keys = ["best_epoch", "val_loss", "test_auuc", "test_qini", "test_lift@30"]
    summary_mean = {}
    summary_std = {}

    for k in metric_keys:
        vals = [r[k] for r in raw_results.values() if k in r and not np.isnan(r[k])]
        if vals:
            summary_mean[k] = round(float(np.mean(vals)), 6)
            summary_std[k] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0
        else:
            summary_mean[k] = float("nan")
            summary_std[k] = float("nan")

    summary_section = {
        "mean": summary_mean,
        "std": summary_std,
    }

    # 4. Lưu metrics.json (chứa kết quả chi tiết từng seed và tổng hợp mean/std)
    metrics_record = {
        "model": model_lower,
        "seeds": raw_results,
        "summary": summary_section,
    }
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Metrics saved to: {metrics_path}")

    # 5. Cập nhật vào results/summary.csv
    summary_csv_path = os.path.join(args.results_dir, "summary.csv")
    update_summary_csv(summary_csv_path, model_lower, summary_section)
    logger.info(f"Summary CSV updated at: {summary_csv_path}")

    # Log summary cho model này
    logger.info("")
    logger.info(f"--- SUMMARY FOR {model_name.upper()} ({len(seeds)} Seeds) ---")
    for k in metric_keys:
        logger.info(f"  {k:15s}: {summary_mean[k]:.5f} ± {summary_std[k]:.5f}")

    return metrics_record


# ══════════════════════════════════════════════════════════════════════════════
# Parallel Experiments Runner
# ══════════════════════════════════════════════════════════════════════════════

def resolve_devices(gpus: list = None, device: str = None) -> list[str]:
    """Xác định danh sách các device (CUDA GPU IDs hoặc CPU) để phân bổ cho workers."""
    if device and str(device).lower() == "cpu":
        return ["cpu"]
    if gpus:
        devs = []
        for g in gpus:
            g_str = str(g).strip().lower()
            if not g_str.startswith("cuda") and g_str != "cpu":
                devs.append(f"cuda:{g_str}")
            else:
                devs.append(g_str)
        return devs
    if torch.cuda.is_available():
        cnt = torch.cuda.device_count()
        if cnt > 0:
            return [f"cuda:{i}" for i in range(cnt)]
    return ["cpu"]


def build_subprocess_cmd(
    model_name: str,
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    """Xây dựng câu lệnh CLI subprocess để chạy độc lập một model trên device cụ thể."""
    main_script = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        main_script,
        "--model", model_name,
        "--device", str(device),
        "--parallel", "1",  # worker chạy đơn lẻ
    ]

    skip_keys = {"model", "models", "device", "parallel", "gpus", "seed"}
    parser = build_parser()
    valid_actions = {act.dest: act for act in parser._actions}

    for k, v in vars(args).items():
        if k in skip_keys or v is None:
            continue
        if k not in valid_actions:
            continue
        if k == "seeds":
            if isinstance(v, (list, tuple)):
                cmd.extend(["--seeds"] + [str(s) for s in v])
            else:
                cmd.extend(["--seeds", str(v)])
        else:
            cmd.extend([f"--{k}", str(v)])

    return cmd


def run_parallel_experiments(
    models: list[str] | str,
    args: argparse.Namespace | dict | None = None,
    parallel: int = 2,
    devices: list[str | int] | None = None,
) -> dict:
    """
    Hàm thực thi song song các thí nghiệm của nhiều baseline models để giảm thời gian chạy.
    
    Tham số:
        models: Danh sách tên models (ví dụ: ['tarnet', 'cfrnet', 'dragonnet']) hoặc 'all'.
        args: Cấu hình arguments (argparse.Namespace hoặc dict hoặc None).
        parallel: Số lượng models chạy đồng thời tối đa.
        devices: Danh sách thiết bị (GPU IDs) phân bổ cho workers (mặc định auto-detect).
    
    Trả về:
        dict chứa kết quả chi tiết của tất cả các models đã chạy.
    """
    if args is None:
        args = build_parser().parse_args([])
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    if isinstance(models, str):
        if models.lower() == "all":
            models_to_run = list(ALL_MODELS)
        else:
            models_to_run = [m.strip().lower() for m in models.split(",") if m.strip()]
    else:
        models_to_run = [m.strip().lower() for m in models]

    # Validate models
    for m in models_to_run:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: '{m}'. Allowed: {ALL_MODELS}")

    target_devices = resolve_devices(devices or getattr(args, "gpus", None), getattr(args, "device", None))
    max_workers = max(1, min(parallel, len(models_to_run)))
    device_pool = [target_devices[i % len(target_devices)] for i in range(max_workers)]

    logger.info("=" * 80)
    logger.info(" PARALLEL EXPERIMENT RUNNER")
    logger.info(f" Total Models ({len(models_to_run)}): {[m.upper() for m in models_to_run]}")
    logger.info(f" Concurrency (Parallel Workers): {max_workers}")
    logger.info(f" Device Pool: {device_pool}")
    logger.info(f" Seeds ({len(args.seeds)}): {args.seeds}")
    logger.info(f" Results Dir: {args.results_dir}")
    logger.info("=" * 80)

    queue = list(models_to_run)
    active_jobs = {}
    all_results = {}
    completed_count = 0
    t0_all = time.time()
    last_heartbeat = time.time()

    try:
        while queue or active_jobs:
            # 1. Khởi động workers nếu còn slot và còn models trong hàng đợi
            while queue and len(active_jobs) < max_workers:
                model_name = queue.pop(0)
                assigned_dev = device_pool.pop(0)

                model_res_dir = os.path.join(args.results_dir, model_name.lower())
                os.makedirs(model_res_dir, exist_ok=True)
                log_path = os.path.join(model_res_dir, "run.log")
                log_file = open(log_path, "w", encoding="utf-8")

                cmd = build_subprocess_cmd(model_name, assigned_dev, args)
                proc_env = os.environ.copy()
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=proc_env,
                )

                active_jobs[model_name] = {
                    "model": model_name,
                    "proc": proc,
                    "device": assigned_dev,
                    "log_file": log_file,
                    "log_path": log_path,
                    "start_time": time.time(),
                    "read_pos": 0,
                }
                logger.info(
                    f"[Parallel Runner] [{completed_count + len(active_jobs)}/{len(models_to_run)}] "
                    f"Launched {model_name.upper()} on {assigned_dev} (PID: {proc.pid}) -> log: {log_path}"
                )

            time.sleep(0.5)

            # 2. Đọc logs mới từ các jobs đang chạy và in ra console
            for m_name, job in list(active_jobs.items()):
                lp = job["log_path"]
                if os.path.exists(lp):
                    try:
                        with open(lp, "r", encoding="utf-8", errors="replace") as rf:
                            rf.seek(job["read_pos"])
                            lines = rf.readlines()
                            job["read_pos"] = rf.tell()
                            for line in lines:
                                l_str = line.strip()
                                if not l_str:
                                    continue
                                # Lọc các mốc tiến độ quan trọng
                                if any(token in l_str for token in [
                                    "Epoch [", "Evaluating", "Seed ", "Done |",
                                    "ERROR", "Traceback", "Exception", "Summary CSV"
                                ]):
                                    logger.info(f"[{m_name.upper()}] {l_str}")
                    except Exception:
                        pass

            # 3. Kiểm tra các jobs đã hoàn thành
            finished_models = []
            for m_name, job in active_jobs.items():
                ret = job["proc"].poll()
                if ret is not None:
                    job["log_file"].close()
                    elapsed = time.time() - job["start_time"]
                    completed_count += 1

                    # Thu hồi device về pool
                    device_pool.append(job["device"])
                    finished_models.append(m_name)

                    if ret == 0:
                        logger.info(
                            f"[Parallel Runner] [{completed_count}/{len(models_to_run)}] "
                            f"SUCCESS: {m_name.upper()} finished in {elapsed:.1f}s (Exit code: 0)"
                        )
                        # Đọc metrics.json
                        metrics_path = os.path.join(args.results_dir, m_name.lower(), "metrics.json")
                        if os.path.exists(metrics_path):
                            try:
                                with open(metrics_path, "r", encoding="utf-8") as mf:
                                    res_data = json.load(mf)
                                    all_results[m_name] = res_data
                                    sm = res_data.get("summary", {}).get("mean", {})
                                    ss = res_data.get("summary", {}).get("std", {})
                                    logger.info(
                                        f"[{m_name.upper()} Summary] Val Loss: {sm.get('val_loss', 0):.5f} ± {ss.get('val_loss', 0):.5f} | "
                                        f"AUUC: {sm.get('test_auuc', 0):.5f} ± {ss.get('test_auuc', 0):.5f} | "
                                        f"Qini: {sm.get('test_qini', 0):.5f} ± {ss.get('test_qini', 0):.5f} | "
                                        f"Lift@30: {sm.get('test_lift@30', 0):.5f} ± {ss.get('test_lift@30', 0):.5f}"
                                    )
                            except Exception as e:
                                all_results[m_name] = {"model": m_name, "error": str(e)}
                        else:
                            all_results[m_name] = {"model": m_name}
                    else:
                        logger.error(
                            f"[Parallel Runner] [{completed_count}/{len(models_to_run)}] "
                            f"FAILED: {m_name.upper()} exited with error code {ret}! Check log: {job['log_path']}"
                        )
                        all_results[m_name] = {
                            "model": m_name,
                            "error": f"Process exited with code {ret}. See {job['log_path']}"
                        }

            for m_name in finished_models:
                del active_jobs[m_name]

            # 4. Heartbeat định kỳ mỗi 30s
            if active_jobs and (time.time() - last_heartbeat > 30):
                running_status = [
                    f"{m.upper()}({j['device']}, {time.time()-j['start_time']:.0f}s)"
                    for m, j in active_jobs.items()
                ]
                logger.info(
                    f"[Parallel Heartbeat] Running ({len(active_jobs)}): {', '.join(running_status)} | "
                    f"Queue remaining: {len(queue)} | Completed: {completed_count}/{len(models_to_run)}"
                )
                last_heartbeat = time.time()

    except KeyboardInterrupt:
        logger.warning("\n[Parallel Runner] KeyboardInterrupt received! Terminating active workers...")
        for m_name, job in active_jobs.items():
            try:
                job["proc"].terminate()
            except Exception:
                pass
        for m_name, job in active_jobs.items():
            try:
                job["proc"].wait(timeout=5)
            except Exception:
                job["proc"].kill()
            try:
                job["log_file"].close()
            except Exception:
                pass
        logger.warning("[Parallel Runner] All workers terminated.")
        sys.exit(130)

    total_time = time.time() - t0_all
    logger.info("")
    logger.info("=" * 88)
    logger.info(f" PARALLEL EXPERIMENT RUN COMPLETE (Total time: {total_time:.1f}s)")
    logger.info("=" * 88)
    header = f"{'Model':<12} {'Val Loss':<20} {'Test AUUC':<20} {'Test Qini':<20} {'Test Lift@30':<20}"
    logger.info(header)
    logger.info("-" * 88)
    for name in models_to_run:
        res = all_results.get(name, {})
        if "error" in res:
            logger.info(f"{name:<12}  ERROR: {res['error']}")
            continue
        sm = res.get("summary", {}).get("mean", {})
        ss = res.get("summary", {}).get("std", {})
        val_str = f"{sm.get('val_loss', 0.0):.5f} ± {ss.get('val_loss', 0.0):.5f}"
        auuc_str = f"{sm.get('test_auuc', 0.0):.5f} ± {ss.get('test_auuc', 0.0):.5f}"
        qini_str = f"{sm.get('test_qini', 0.0):.5f} ± {ss.get('test_qini', 0.0):.5f}"
        lift_str = f"{sm.get('test_lift@30', 0.0):.5f} ± {ss.get('test_lift@30', 0.0):.5f}"
        logger.info(f"{name:<12} {val_str:<20} {auuc_str:<20} {qini_str:<20} {lift_str:<20}")
    logger.info("=" * 88)
    logger.info(f"Summary CSV saved to: {os.path.join(args.results_dir, 'summary.csv')}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Merge config.yaml (nếu có) vào args ──────────────────────────────────
    if args.config is not None:
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        args = merge_config_into_args(args, args.config)
        logger.info(f"Loaded config from: {args.config}")

    # ── Xử lý danh sách seeds (mặc định 1..5) ─────────────────────────────────
    if args.seed is not None:
        args.seeds = [args.seed]
    elif isinstance(args.seeds, str):
        args.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    elif isinstance(args.seeds, int):
        args.seeds = [args.seeds]
    else:
        args.seeds = list(args.seeds)

    # ── Danh sách models cần chạy ─────────────────────────────────────────────
    if args.models:
        models_to_run = [m.lower().strip() for m in args.models if m.strip()]
    elif "," in args.model:
        models_to_run = [m.lower().strip() for m in args.model.split(",") if m.strip()]
    elif args.model.lower() == "all":
        models_to_run = list(ALL_MODELS)
    else:
        models_to_run = [args.model.lower().strip()]

    # Validate models
    invalid_models = [m for m in models_to_run if m not in ALL_MODELS]
    if invalid_models:
        parser.error(f"Invalid model(s): {invalid_models}. Allowed models: {ALL_MODELS}")

    # ── Kiểm tra chế độ chạy song song (Parallel execution) ───────────────────
    parallel = getattr(args, "parallel", 1)
    if parallel > 1 and len(models_to_run) > 1:
        return run_parallel_experiments(
            models=models_to_run,
            args=args,
            parallel=parallel,
            devices=getattr(args, "gpus", None),
        )

    # ── Nạp dữ liệu pre-split và tạo DataLoaders cho models (Tuần tự) ─────────
    data_dir = args.data if (os.path.exists(args.data) and os.path.isdir(args.data)) else None
    train_path = getattr(args, "train_path", None)
    if train_path is None and os.path.exists(args.data) and not os.path.isdir(args.data):
        train_path = args.data

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        train_path=train_path,
        val_path=getattr(args, "val_path", None),
        test_path=getattr(args, "test_path", None),
        batch_size=args.batch_size,
        label_col=getattr(args, "label_col", "visit"),
        num_workers=getattr(args, "num_workers", 2),
    )

    logger.info(
        f"DataLoaders ready | train batches: {len(train_loader):,} "
        f"| val batches: {len(val_loader) if val_loader else 0:,} "
        f"| test batches: {len(test_loader) if test_loader else 0:,}"
    )

    # ── Chạy thí nghiệm tuần tự qua các seeds ─────────────────────────────────
    all_results = {}
    for model_name in models_to_run:
        try:
            model_record = run_model(
                model_name, args,
                train_loader, val_loader, test_loader
            )
            all_results[model_name] = model_record
        except Exception as e:
            logger.error(f"Experiment [{model_name}] failed: {e}", exc_info=True)
            all_results[model_name] = {"error": str(e)}

    # ── Bảng tổng hợp các models ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 88)
    logger.info(f" MULTI-SEED EXPERIMENT SUMMARY ({len(args.seeds)} Seeds: {args.seeds})")
    logger.info("=" * 88)
    header = f"{'Model':<12} {'Val Loss':<20} {'Test AUUC':<20} {'Test Qini':<20} {'Test Lift@30':<20}"
    logger.info(header)
    logger.info("-" * 88)
    for name, res in all_results.items():
        if "error" in res:
            logger.info(f"{name:<12}  ERROR: {res['error']}")
            continue
        sm = res.get("summary", {}).get("mean", {})
        ss = res.get("summary", {}).get("std", {})
        val_str = f"{sm.get('val_loss', 0.0):.5f} ± {ss.get('val_loss', 0.0):.5f}"
        auuc_str = f"{sm.get('test_auuc', 0.0):.5f} ± {ss.get('test_auuc', 0.0):.5f}"
        qini_str = f"{sm.get('test_qini', 0.0):.5f} ± {ss.get('test_qini', 0.0):.5f}"
        lift_str = f"{sm.get('test_lift@30', 0.0):.5f} ± {ss.get('test_lift@30', 0.0):.5f}"
        logger.info(f"{name:<12} {val_str:<20} {auuc_str:<20} {qini_str:<20} {lift_str:<20}")
    logger.info("=" * 88)
    logger.info(f"Summary CSV saved to: {os.path.join(args.results_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
