from __future__ import annotations


from pathlib import Path

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.7
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    def __post_init__(self) -> None:
        total = self.train_size + self.valid_size + self.test_size
        if abs(total - 1.0) > 1e-6:  # relaxed tolerance (correct)
            raise ValueError(
                f"train_size + valid_size + test_size must equal 1.0 (got {total})"
            )


def group_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split: groups are disjoint across splits.

    Strategy:
    - Split train vs temp (valid+test)
    - Split temp into valid vs test
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col: {group_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target_col: {target_col}")

    # 1) Train vs Temp
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=cfg.train_size,
        random_state=cfg.random_state,
    )
    train_idx, temp_idx = next(
        gss1.split(df, y=df[target_col], groups=df[group_col])
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # 2) Valid vs Test within Temp
    temp_total = cfg.valid_size + cfg.test_size
    valid_frac_of_temp = cfg.valid_size / temp_total

    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=valid_frac_of_temp,
        random_state=cfg.random_state,
    )
    valid_idx, test_idx = next(
        gss2.split(
            temp_df,
            y=temp_df[target_col],
            groups=temp_df[group_col],
        )
    )

    valid_df = temp_df.iloc[valid_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, valid_df, test_df
