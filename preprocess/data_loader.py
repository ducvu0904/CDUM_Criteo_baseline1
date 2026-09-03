"""
data_loader.py — Tiền xử lý và nạp dữ liệu Criteo Uplift v2.1 cho PyTorch
=========================================================================
Chức năng:
  - load_dataset()    : đọc file .csv.gz / .csv, in thống kê cơ bản.
  - split_dataset()   : chia train/val/test theo tỷ lệ (mặc định 8:1:1).
  - get_dataloader()  : tạo PyTorch DataLoader từ (X, t, y) dùng TensorDataset.
  - get_dataloaders() : tạo đồng thời train_loader, val_loader, test_loader.

Dataset : Criteo Uplift v2.1 (~14M rows, 12 features, binary treatment)
Columns : f0…f11 (numeric), treatment (0/1), visit (0/1)

Môi trường yêu cầu:
    pandas, numpy, scikit-learn, torch
"""

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Optional, Union, Tuple, List

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def _to_tensor(data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor], dtype=torch.float32) -> torch.Tensor:
    """Chuyển đổi DataFrame / Series / ndarray sang torch.Tensor float32."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.from_numpy(data.to_numpy(dtype=np.float32))
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32))
    else:
        return torch.tensor(data, dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Load dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    data_path: str,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "visit",
    treat_col: str = "treatment",
) -> pd.DataFrame:
    """
    Đọc file CSV (hoặc .csv.gz), in thống kê cơ bản và trả về DataFrame.

    Tham số
    -------
    data_path   : str  — đường dẫn file CSV / CSV.gz.
    feature_cols: list — danh sách tên cột feature (mặc định ['f0', ..., 'f11'] nếu None).
    label_col   : str  — tên cột nhãn kết quả (mặc định 'visit').
    treat_col   : str  — tên cột treatment (mặc định 'treatment').

    Trả về
    ------
    df_all : pd.DataFrame — toàn bộ dataset.
    """
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(12)]

    print(f"\n[load_dataset] Loading: {data_path}")
    df_all = pd.read_csv(data_path)
    print(f"               Total rows : {len(df_all):,}")
    print(f"               Treatment ratio (1/0): {df_all[treat_col].mean():.4f}")
    print(f"               Positive rate ({label_col}): {df_all[label_col].mean():.4f}")
    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# Train / Val / Test split
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(
    df_all: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "visit",
    treat_col: str = "treatment",
    test_size: float = 0.1,
    val_ratio: float = 1/9,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.Series, pd.Series]:
    """
    Chia dataset theo tỷ lệ (mặc định 8:1:1 train : val : test).

    Tham số
    -------
    df_all       : pd.DataFrame — toàn bộ dataset từ load_dataset().
    feature_cols : list  — danh sách tên cột feature (mặc định ['f0'...'f11']).
    label_col    : str   — tên cột nhãn (mặc định 'visit').
    treat_col    : str   — tên cột treatment (mặc định 'treatment').
    test_size    : float — tỷ lệ tách ra khỏi train (0.2 -> 20% cho val+test).
    val_ratio    : float — tỷ lệ chia val từ phần test_size (0.5 -> val chiếm 10%, test 10%).
    random_state : int   — seed để tái lập kết quả.

    Trả về
    ------
    (x_train, y_train, t_train,
     x_val,   y_val,   t_val,
     x_test,  y_test,  t_test)
    """
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(12)]

    print(f"\n[split_dataset] Splitting data ...")
    df_train, df_tmp = train_test_split(
        df_all, test_size=test_size, random_state=random_state
    )
    df_val, df_test = train_test_split(
        df_tmp, test_size=val_ratio, random_state=random_state
    )
    print(f"                train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    x_train = df_train[feature_cols]; y_train = df_train[label_col]; t_train = df_train[treat_col]
    x_val   = df_val[feature_cols];   y_val   = df_val[label_col];   t_val   = df_val[treat_col]
    x_test  = df_test[feature_cols];  y_test  = df_test[label_col];  t_test  = df_test[treat_col]

    return (
        x_train, y_train, t_train,
        x_val,   y_val,   t_val,
        x_test,  y_test,  t_test,
    )


def compute_denominators(df_train: pd.DataFrame, feature_cols: List[str]) -> List[float]:
    """Tính giá trị max từng feature trên train set để dùng cho bucket normalization nếu cần."""
    return [float(df_train[f].max()) for f in feature_cols]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch DataLoader Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloader(
    X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    t: Union[np.ndarray, pd.Series, torch.Tensor],
    y: Union[np.ndarray, pd.Series, torch.Tensor],
    batch_size: int = 2048,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Tạo PyTorch DataLoader trực tiếp từ (X, t, y) dùng TensorDataset.

    Tham số
    -------
    X           : features (DataFrame, ndarray, hoặc Tensor).
    t           : treatment indicator (Series, ndarray, hoặc Tensor).
    y           : outcome label (Series, ndarray, hoặc Tensor).
    batch_size  : kích thước batch.
    shuffle     : có xáo trộn mẫu hay không (mặc định True cho train, False cho val/test).
    num_workers : số tiến trình nạp dữ liệu song song (mặc định 0).
    pin_memory  : có pin memory sang GPU không (mặc định False).

    Trả về
    ------
    DataLoader trả về từng batch (x_batch, t_batch, y_batch) dạng float32.
    """
    dataset = TensorDataset(_to_tensor(X), _to_tensor(t), _to_tensor(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_dataloaders(
    x_train: Union[np.ndarray, pd.DataFrame],
    t_train: Union[np.ndarray, pd.Series],
    y_train: Union[np.ndarray, pd.Series],
    x_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    t_val: Optional[Union[np.ndarray, pd.Series]] = None,
    y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    x_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    t_test: Optional[Union[np.ndarray, pd.Series]] = None,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Tạo nhanh bộ DataLoaders (train, val, test) cho quá trình huấn luyện PyTorch.

    Trả về
    ------
    (train_loader, val_loader, test_loader)
    """
    train_loader = get_dataloader(
        x_train, t_train, y_train,
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = None
    if x_val is not None and t_val is not None and y_val is not None:
        val_loader = get_dataloader(
            x_val, t_val, y_val,
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    test_loader = None
    if x_test is not None and t_test is not None and y_test is not None:
        test_loader = get_dataloader(
            x_test, t_test, y_test,
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader
