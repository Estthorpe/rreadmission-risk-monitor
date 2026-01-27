from __future__ import annotations

import pandas as pd

from readmission_risk_monitor.config import SETTINGS
from readmission_risk_monitor.features.leakage import assert_patient_disjoint
from readmission_risk_monitor.features.split import SplitConfig, group_split


def test_patient_disjoint_splits_on_fixture() -> None:
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    df = pd.read_parquet(fixture_path)

    cfg = SplitConfig(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42)
    train_df, valid_df, test_df = group_split(
        df,
        group_col=SETTINGS.patient_id_col,
        target_col=SETTINGS.target_col,
        cfg=cfg,
    )

    # This is the core healthcare leakage guard
    assert_patient_disjoint(train_df, valid_df, test_df, SETTINGS.patient_id_col)


def test_split_sizes_reasonable() -> None:
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    df = pd.read_parquet(fixture_path)

    cfg = SplitConfig(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42)
    train_df, valid_df, test_df = group_split(
        df,
        group_col=SETTINGS.patient_id_col,
        target_col=SETTINGS.target_col,
        cfg=cfg,
    )

    n = len(df)
    assert 0.60 * n <= len(train_df) <= 0.80 * n
    assert 0.05 * n <= len(valid_df) <= 0.25 * n
    assert 0.05 * n <= len(test_df) <= 0.25 * n


def test_split_reproducibility_same_cfg_same_patient_sets() -> None:
    """
    Docstring for test_split_reproducibility_same_cfg_same_patient_sets
    """
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    df = pd.read_parquet(fixture_path)

    cfg = SplitConfig(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42)


    t1, v1, s1 = group_split(
        df, group_col=SETTINGS.patient_id_col, target_col=SETTINGS.target_col, cfg=cfg,
    )
    t2, v2, s2 = group_split(
        df, group_col=SETTINGS.patient_id_col, target_col=SETTINGS.target_col, cfg=cfg,
    )

    def pset(x: pd.DataFrame) -> set:
        return set(x[SETTINGS.patient_id_col].unique())
    
    assert pset(t1) == pset(t2)
    assert pset(v1) == pset(v2) 
    assert pset(s1) == pset(s2)
