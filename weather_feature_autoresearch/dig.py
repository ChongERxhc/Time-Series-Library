from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SET_NAME = "selected_physics_wind_v1"


def compute_features(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic weather features.

    raw_df is kept unchanged in the final dataset. clean_df is only used to
    avoid sensor sentinels such as -9999 contaminating derived features.
    """
    _ = raw_df

    wind_dir_rad = np.deg2rad(clean_df["wd (deg)"])
    features = pd.DataFrame(index=clean_df.index)

    features["llm_temp_dew_gap"] = clean_df["T (degC)"] - clean_df["Tdew (degC)"]
    features["llm_vapor_pressure_ratio"] = clean_df["VPact (mbar)"] / (clean_df["VPmax (mbar)"] + 1e-6)
    features["llm_vapor_pressure_gap"] = clean_df["VPmax (mbar)"] - clean_df["VPact (mbar)"]
    features["llm_wind_u"] = clean_df["wv (m/s)"] * np.cos(wind_dir_rad)
    features["llm_wind_v"] = clean_df["wv (m/s)"] * np.sin(wind_dir_rad)
    features["llm_max_wind_u"] = clean_df["max. wv (m/s)"] * np.cos(wind_dir_rad)
    features["llm_max_wind_v"] = clean_df["max. wv (m/s)"] * np.sin(wind_dir_rad)
    features["llm_wind_gust_gap"] = clean_df["max. wv (m/s)"] - clean_df["wv (m/s)"]

    return features
