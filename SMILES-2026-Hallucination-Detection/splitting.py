from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

_N_FOLDS = 5
_TEST_SIZE = 0.15


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = _TEST_SIZE,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices using stratified k-fold with a fixed held-out test set
    """
    idx = np.arange(len(y))

    # fixed held-out test set
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # stratified k-fold over  remaining samples
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for train_rel, val_rel in skf.split(idx_train_val, y[idx_train_val]):
        idx_tr = idx_train_val[train_rel]
        idx_va = idx_train_val[val_rel]
        splits.append((idx_tr, idx_va, idx_test))

    return splits
