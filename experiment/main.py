"""
main.py — Entry point để chạy thí nghiệm các baseline uplift models trên Criteo
=================================================================================
Sử dụng:
    python experiment/main.py --config experiment/config.yaml
    python experiment/main.py --model tarnet --data path/to/criteo.csv
    python experiment/main.py --config experiment/config.yaml --model cfrnet
    python experiment/main.py --model all --epochs 20 --batch_size 4096

Models có thể chọn (--model):
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
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Đảm bảo project root nằm trong sys.path ──────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.data_loader import load_dataset, split_dataset, get_dataloaders
from baselines import TARNET, CEVAE, DESCN, EUEN, GANITE, DRAGONNET, CFRNET, EFIN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

ALL_MODELS = ["tarnet", "cevae", "descn", "euen", "ganite", "dragonnet", "cfrnet", "efin"]


# ══════════════════════════════════════════════════════════════════════════════
# Config YAML helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml_config(config_path: str) -> dict:
    """Load file YAML và trả về dict phẳng với các key tương ứng args."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    section_map = {
        "data":      {"path": "data", "test_size": "test_size", "val_ratio": "val_ratio", "seed": "seed"},
        "model":     {"name": "model", "input_dim": "input_dim", "shared_dim": "shared_dim", "head_dim": "head_dim"},
        "training":  {"epochs": "epochs", "batch_size": "batch_size", "lr": "lr",
                      "weight_decay": "weight_decay", "patience": "patience", "device": "device"},
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
    data.add_argument("--data", type=str, default="data/criteo-uplift-v2.1.csv",
                      help="Path to Criteo CSV (or .csv.gz).")
    data.add_argument("--test_size", type=float, default=0.1,
                      help="Fraction of data reserved for test+val (default: 0.1 -> 8:1:1 split).")
    data.add_argument("--val_ratio", type=float, default=1/9,
                      help="Fraction within test_size for validation.")
    data.add_argument("--seed", type=int, default=42, help="Random seed.")

    # ── Model ─────────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--model", type=str, default="tarnet",
                     choices=ALL_MODELS + ["all"],
                     help="Model to train. Use 'all' to run every baseline sequentially.")
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
    trn.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay.")
    trn.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    trn.add_argument("--device", type=str, default=None,
                     help="Device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detect if not set.")

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

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Run single experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_name: str, args: argparse.Namespace,
                   train_loader, val_loader, test_loader) -> dict:
    """Chạy một thí nghiệm đơn và trả về dict kết quả."""
    from torch.utils.tensorboard import SummaryWriter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_upper = model_name.upper()

    # Cấu trúc lưu trữ chuẩn theo pattern partner: results/<MODEL>/checkpoints và results/<MODEL>/runs
    model_res_dir = os.path.join(args.results_dir, model_upper)
    ckpt_dir = os.path.join(model_res_dir, "checkpoints")
    tb_dir = os.path.join(model_res_dir, "runs", f"{model_name}_criteo")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f" Model       : {model_upper}")
    logger.info(f" Checkpoints : {ckpt_dir}")
    logger.info(f" TensorBoard : {tb_dir}")
    logger.info(f"{'='*60}")

    # Khởi tạo TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=tb_dir)

    trainer = build_model(model_name, args)

    t_start = time.time()

    # ── Fit ───────────────────────────────────────────────────────────────────
    fit_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping_patience=args.patience,
        checkpoint_dir=ckpt_dir,
        model_name=f"{model_name}_criteo",
        writer=writer,
        verbose=args.verbose,
    )

    if model_name == "ganite":
        fit_kwargs["epochs_gan"] = args.epochs_gan
        fit_kwargs["epochs_inf"] = args.epochs_inf or args.epochs
    else:
        fit_kwargs["epochs"] = args.epochs

    history = trainer.fit(**fit_kwargs)
    elapsed = time.time() - t_start

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info(f"Evaluating {model_upper} on test set...")
    metrics = trainer.evaluate(test_loader, k=args.eval_k)
    metrics["train_time_s"] = round(elapsed, 2)

    # Ghi test metrics vào TensorBoard
    for metric_name, val in metrics.items():
        if isinstance(val, (int, float)):
            writer.add_scalar(f"Test/{metric_name}", val, 0)
    writer.close()

    # ── Log results ───────────────────────────────────────────────────────────
    logger.info(f"Results [{model_upper}]:")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result_record = {
        "model": model_name,
        "timestamp": ts,
        "args": vars(args),
        "metrics": metrics,
        "history": history,
    }
    result_path = os.path.join(model_res_dir, f"{model_name}_criteo_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to: {result_path}")

    return metrics


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

    np.random.seed(args.seed)

    # ── Load và split dữ liệu ─────────────────────────────────────────────────
    feature_cols = [f"f{i}" for i in range(args.input_dim)]
    df = load_dataset(args.data, feature_cols=feature_cols)

    x_tr, y_tr, t_tr, x_val, y_val, t_val, x_te, y_te, t_te = split_dataset(
        df,
        feature_cols=feature_cols,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        x_train=x_tr, t_train=t_tr, y_train=y_tr,
        x_val=x_val,   t_val=t_val,   y_val=y_val,
        x_test=x_te,   t_test=t_te,   y_test=y_te,
        batch_size=args.batch_size,
    )

    logger.info(
        f"Data loaded | train: {len(x_tr):,} | val: {len(x_val):,} | test: {len(x_te):,}"
    )

    # ── Danh sách models cần chạy ─────────────────────────────────────────────
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    # ── Chạy từng thí nghiệm ──────────────────────────────────────────────────
    all_results = {}
    for model_name in models_to_run:
        try:
            metrics = run_experiment(
                model_name, args,
                train_loader, val_loader, test_loader
            )
            all_results[model_name] = metrics
        except Exception as e:
            logger.error(f"Experiment [{model_name}] failed: {e}", exc_info=True)
            all_results[model_name] = {"error": str(e)}

    # ── Summary bảng kết quả ──────────────────────────────────────────────────
    if len(models_to_run) > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info(" SUMMARY")
        logger.info("=" * 70)
        header = f"{'Model':<12} {'Loss':>10} {'AUUC':>10} {'Qini':>10} {'Lift@30%':>10} {'Time(s)':>8}"
        logger.info(header)
        logger.info("-" * 70)
        for name, m in all_results.items():
            if "error" in m:
                logger.info(f"{name:<12}  ERROR: {m['error']}")
                continue
            row = (
                f"{name:<12}"
                f" {m.get('loss', float('nan')):>10.5f}"
                f" {m.get('auuc', float('nan')):>10.5f}"
                f" {m.get('qini', float('nan')):>10.5f}"
                f" {m.get('lift@30%', float('nan')):>10.5f}"
                f" {m.get('train_time_s', 0):>8.1f}"
            )
            logger.info(row)
        logger.info("=" * 70)

        # Lưu tổng hợp
        summary_path = os.path.join(
            args.results_dir,
            f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
