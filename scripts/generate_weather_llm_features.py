import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL_THRESHOLD = -1000
EPS = 1e-6
SELECTED_FEATURE_COLUMNS = [
    "llm_temp_dew_gap",
    "llm_vapor_pressure_ratio",
    "llm_vapor_pressure_gap",
    "llm_wind_u",
    "llm_wind_v",
    "llm_max_wind_u",
    "llm_max_wind_v",
    "llm_wind_gust_gap",
]


def clean_sensor_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known invalid sensor sentinels without changing the time index."""
    df = df.copy()
    numeric_cols = [col for col in df.columns if col != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= SENTINEL_THRESHOLD, col] = np.nan
        df[col] = df[col].ffill().bfill()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pressure = df["p (mbar)"]
    temp = df["T (degC)"]
    dew = df["Tdew (degC)"]
    humidity = df["rh (%)"]
    vpmax = df["VPmax (mbar)"]
    vpact = df["VPact (mbar)"]
    wind = df["wv (m/s)"]
    max_wind = df["max. wv (m/s)"]
    wind_dir_rad = np.deg2rad(df["wd (deg)"])
    swdr = df["SWDR (W/m�)"]
    par = df["PAR (�mol/m�/s)"]
    ot = df["OT"]

    # Physics-inspired cross-channel features proposed for the first small test.
    df["llm_temp_humidity"] = temp * humidity
    df["llm_temp_dew_gap"] = temp - dew
    df["llm_vapor_pressure_ratio"] = vpact / (vpmax + EPS)
    df["llm_vapor_pressure_gap"] = vpmax - vpact
    df["llm_wind_chill_proxy"] = temp - 0.7 * wind
    df["llm_radiation_efficiency"] = par / (swdr + EPS)

    df["llm_wind_u"] = wind * np.cos(wind_dir_rad)
    df["llm_wind_v"] = wind * np.sin(wind_dir_rad)
    df["llm_max_wind_u"] = max_wind * np.cos(wind_dir_rad)
    df["llm_max_wind_v"] = max_wind * np.sin(wind_dir_rad)
    df["llm_wind_gust_gap"] = max_wind - wind

    lag_1h = 6
    window_6h = 36
    df["llm_temp_diff_6"] = temp.diff(lag_1h).fillna(0.0)
    df["llm_pressure_diff_6"] = pressure.diff(lag_1h).fillna(0.0)
    df["llm_humidity_diff_6"] = humidity.diff(lag_1h).fillna(0.0)
    df["llm_ot_diff_6"] = ot.diff(lag_1h).fillna(0.0)

    df["llm_temp_roll_mean_36"] = temp.rolling(window_6h, min_periods=1).mean()
    df["llm_temp_roll_std_36"] = temp.rolling(window_6h, min_periods=2).std().fillna(0.0)
    df["llm_pressure_roll_mean_36"] = pressure.rolling(window_6h, min_periods=1).mean()
    df["llm_humidity_roll_mean_36"] = humidity.rolling(window_6h, min_periods=1).mean()
    df["llm_wind_roll_mean_36"] = wind.rolling(window_6h, min_periods=1).mean()
    df["llm_radiation_roll_mean_36"] = swdr.rolling(window_6h, min_periods=1).mean()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM-inspired weather features for TSLib.")
    parser.add_argument("--input", default="./dataset/weather/weather.csv")
    parser.add_argument("--clean-output", default="./dataset/weather/weather_clean.csv")
    parser.add_argument("--feature-output", default="./dataset/weather/weather_llm_features.csv")
    parser.add_argument("--selected-feature-output", default="./dataset/weather/weather_llm_selected_features.csv")
    parser.add_argument("--raw-base-feature-output", default="./dataset/weather/weather_raw_base_llm_features.csv")
    parser.add_argument("--raw-base-selected-feature-output",
                        default="./dataset/weather/weather_raw_base_llm_selected_features.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    clean_output_path = Path(args.clean_output)
    feature_output_path = Path(args.feature_output)
    selected_feature_output_path = Path(args.selected_feature_output)
    raw_base_feature_output_path = Path(args.raw_base_feature_output)
    raw_base_selected_feature_output_path = Path(args.raw_base_selected_feature_output)

    df = pd.read_csv(input_path)
    clean_df = clean_sensor_sentinels(df)
    feature_df = add_weather_features(clean_df)
    selected_feature_df = pd.concat(
        [clean_df, feature_df[SELECTED_FEATURE_COLUMNS]],
        axis=1,
    )
    feature_columns = [col for col in feature_df.columns if col.startswith("llm_")]
    raw_base_feature_df = pd.concat(
        [df, feature_df[feature_columns]],
        axis=1,
    )
    raw_base_selected_feature_df = pd.concat(
        [df, feature_df[SELECTED_FEATURE_COLUMNS]],
        axis=1,
    )

    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_base_feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_base_selected_feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_output_path, index=False)
    feature_df.to_csv(feature_output_path, index=False)
    selected_feature_df.to_csv(selected_feature_output_path, index=False)
    raw_base_feature_df.to_csv(raw_base_feature_output_path, index=False)
    raw_base_selected_feature_df.to_csv(raw_base_selected_feature_output_path, index=False)

    original_numeric_cols = len([col for col in df.columns if col != "date"])
    enhanced_numeric_cols = len([col for col in feature_df.columns if col != "date"])
    selected_numeric_cols = len([col for col in selected_feature_df.columns if col != "date"])
    raw_base_enhanced_numeric_cols = len([col for col in raw_base_feature_df.columns if col != "date"])
    raw_base_selected_numeric_cols = len([col for col in raw_base_selected_feature_df.columns if col != "date"])
    print(f"input rows: {len(df)}")
    print(f"clean output: {clean_output_path} ({original_numeric_cols} variables)")
    print(f"feature output: {feature_output_path} ({enhanced_numeric_cols} variables)")
    print(f"selected feature output: {selected_feature_output_path} ({selected_numeric_cols} variables)")
    print(f"raw-base feature output: {raw_base_feature_output_path} ({raw_base_enhanced_numeric_cols} variables)")
    print(f"raw-base selected feature output: {raw_base_selected_feature_output_path} "
          f"({raw_base_selected_numeric_cols} variables)")
    print(f"added features: {enhanced_numeric_cols - original_numeric_cols}")
    print(f"selected added features: {selected_numeric_cols - original_numeric_cols}")


if __name__ == "__main__":
    main()
