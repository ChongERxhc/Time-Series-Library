from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SET_NAME = "calendar_dow_month_v1"


def compute_features(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Exchange rate calendar features (daily); clean_df for API consistency."""
    _ = clean_df
    features = pd.DataFrame(index=raw_df.index)
    dt = pd.to_datetime(raw_df["date"])
    dow = dt.dt.dayofweek.astype(float)
    month = dt.dt.month.astype(float)
    features["llm_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["llm_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    features["llm_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["llm_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return features
