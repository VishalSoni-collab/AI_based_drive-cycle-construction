#!/usr/bin/env python3
"""
Build a representative driving cycle from real-world speed-time data.

This file is meant to be the main reproducible pipeline for the repo. It takes a
VED-style vehicle trip CSV and turns it into a shorter representative driving
cycle using the same basic logic used in microtrip-based drive-cycle papers:

    raw speed/GPS data
    -> 1 Hz resampling
    -> acceleration cleanup
    -> idle/moving run detection
    -> microtrip segmentation
    -> feature extraction
    -> PCA + K-means clustering
    -> optimized microtrip subset selection
    -> final stitched speed-time cycle

The code is written as one script on purpose. During development the workflow was
checked phase by phase in a notebook, but for the repo it is easier to maintain
and rerun a single pipeline. Intermediate CSV files are still saved so the result
can be inspected at each stage.

Expected input columns, matching the cleaned VED cycle files:

    VehId, Trip, Timestamp(s), Latitude[deg], Longitude[deg],
    Vehicle Speed[km/h], elevation[m]

Example:

    python drive_cycle_pipeline.py --input data/cycle_v1.csv --output outputs --target-duration 1200

A note on the optimizer:
This implementation uses exhaustive subset search when the number of candidate microtrips is small,
and falls back to random search for larger inputs. So the method is inspired by
the same microtrip/PCA/K-means workflow.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Column handling and utilities
# -----------------------------
# The VED files use descriptive column names. Internally I keep short names so
# the rest of the code stays readable. The original units are preserved.

COLUMN_MAP = {
    "VehId": "veh_id",
    "Trip": "trip_id",
    "Timestamp(s)": "time_s",
    "Latitude[deg]": "lat",
    "Longitude[deg]": "lon",
    "Vehicle Speed[km/h]": "speed_kmh",
    "elevation[m]": "elevation_m",
}

REQUIRED_COLUMNS = [
    "veh_id",
    "trip_id",
    "time_s",
    "lat",
    "lon",
    "speed_kmh",
    "elevation_m",
]

# Speed-bin percentages are part of the validation. They tell us whether the
# final cycle spends a similar amount of time in low, medium, and high speeds.
SPEED_BINS_KMH = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
SPEED_BIN_NAMES = [
    "speed_pct_0_10",
    "speed_pct_10_20",
    "speed_pct_20_30",
    "speed_pct_30_40",
    "speed_pct_40_50",
    "speed_pct_50_60",
    "speed_pct_60_70",
    "speed_pct_70_80",
    "speed_pct_80_plus",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up input columns and keep only the fields used downstream.

    The pipeline only needs speed, time, trip identity, GPS and elevation. Any
    row that cannot be converted to numeric values is dropped here instead of
    failing later inside interpolation or clustering.
    """
    df = df.rename(columns=COLUMN_MAP).copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing/non-numeric values.")

    df = df.sort_values(["veh_id", "trip_id", "time_s"]).reset_index(drop=True)
    return df


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Preprocessing
# -----------------------------
# The raw timestamps are often irregular. Acceleration is very sensitive to the
# time step, so the speed trace is resampled to 1 Hz before the final acceleration
# signal is calculated.


def resample_trip_to_1hz(trip_df: pd.DataFrame) -> pd.DataFrame:
    """Resample one vehicle-trip to a regular 1 Hz timeline.

    Linear interpolation is used for speed, GPS, and elevation. This keeps the
    trip shape close to the original recording while making the later acceleration
    and microtrip calculations consistent.
    """
    trip_df = trip_df.sort_values("time_s").drop_duplicates("time_s").copy()

    veh_id = trip_df["veh_id"].iloc[0]
    trip_id = trip_df["trip_id"].iloc[0]

    t_original = trip_df["time_s"].to_numpy(dtype=float)
    if len(t_original) < 2:
        return pd.DataFrame()

    t_start = np.ceil(t_original.min())
    t_end = np.floor(t_original.max())
    if t_end <= t_start:
        return pd.DataFrame()

    t_new = np.arange(t_start, t_end + 1, 1.0)

    return pd.DataFrame(
        {
            "veh_id": veh_id,
            "trip_id": trip_id,
            "time_s": t_new,
            "lat": np.interp(t_new, t_original, trip_df["lat"].to_numpy(dtype=float)),
            "lon": np.interp(t_new, t_original, trip_df["lon"].to_numpy(dtype=float)),
            "speed_kmh_raw_1hz": np.interp(
                t_new, t_original, trip_df["speed_kmh"].to_numpy(dtype=float)
            ),
            "elevation_m": np.interp(
                t_new, t_original, trip_df["elevation_m"].to_numpy(dtype=float)
            ),
        }
    )


def enforce_acceleration_limits(
    trip_df: pd.DataFrame,
    speed_col: str,
    max_accel: float,
    max_decel: float,
    max_speed_kmh: float,
) -> pd.DataFrame:
    """Apply simple physical limits to the speed trace.

    This does not smooth the whole signal. It only corrects points that would
    imply unrealistic acceleration or braking after 1 Hz resampling. The limits
    are intentionally exposed as CLI arguments because different datasets may
    need slightly different thresholds.
    """
    trip_df = trip_df.sort_values("time_s").copy()

    speed_mps = (trip_df[speed_col].to_numpy(dtype=float) / 3.6).copy()
    time_s = trip_df["time_s"].to_numpy(dtype=float)

    cleaned = speed_mps.copy()
    for i in range(1, len(cleaned)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            continue

        accel = (cleaned[i] - cleaned[i - 1]) / dt
        if accel > max_accel:
            cleaned[i] = cleaned[i - 1] + max_accel * dt
        elif accel < max_decel:
            cleaned[i] = cleaned[i - 1] + max_decel * dt

        cleaned[i] = max(cleaned[i], 0.0)

    trip_df["speed_mps_clean"] = cleaned
    trip_df["speed_kmh_clean"] = np.clip(cleaned * 3.6, 0, max_speed_kmh)
    trip_df["speed_mps_clean"] = trip_df["speed_kmh_clean"] / 3.6
    trip_df["accel_mps2_clean"] = trip_df["speed_mps_clean"].diff().fillna(0)
    return trip_df


def preprocess_to_1hz(
    raw_df: pd.DataFrame,
    max_accel: float,
    max_decel: float,
    max_speed_kmh: float,
) -> pd.DataFrame:
    """Sort, resample, and clean every vehicle-trip in the input file."""
    pieces = []
    for _, trip_df in raw_df.groupby(["veh_id", "trip_id"], sort=False):
        out = resample_trip_to_1hz(trip_df)
        if not out.empty:
            pieces.append(out)

    if not pieces:
        raise ValueError("No usable trips after 1 Hz resampling.")

    df_1hz = pd.concat(pieces, ignore_index=True)
    df_1hz = df_1hz.sort_values(["veh_id", "trip_id", "time_s"]).reset_index(drop=True)

    df_1hz["dt_s"] = df_1hz.groupby(["veh_id", "trip_id"])["time_s"].diff()
    df_1hz["speed_mps_raw_1hz"] = df_1hz["speed_kmh_raw_1hz"] / 3.6
    df_1hz["accel_mps2_raw_1hz"] = (
        df_1hz.groupby(["veh_id", "trip_id"])["speed_mps_raw_1hz"].diff()
        / df_1hz["dt_s"]
    )

    cleaned_parts = []
    for _, trip_df in df_1hz.groupby(["veh_id", "trip_id"], sort=False):
        cleaned_parts.append(
            enforce_acceleration_limits(
                trip_df,
                speed_col="speed_kmh_raw_1hz",
                max_accel=max_accel,
                max_decel=max_decel,
                max_speed_kmh=max_speed_kmh,
            )
        )

    return pd.concat(cleaned_parts, ignore_index=True)


# -----------------------------
# Microtrip segmentation
# -----------------------------
# A microtrip is the basic building block of the final cycle. Here it is defined
# as one moving run plus the immediately following idle run, which is a common
# convention in drive-cycle construction work.


def add_run_ids(trip_df: pd.DataFrame, idle_speed_kmh: float) -> pd.DataFrame:
    """Label continuous idle and moving runs within a trip.

    Exact zero speed is too strict for GPS/OBD data, so a small threshold such as
    1 km/h is used for stopped/idle points.
    """
    trip_df = trip_df.sort_values("time_s").copy()
    trip_df["is_idle"] = trip_df["speed_kmh_clean"] <= idle_speed_kmh
    trip_df["run_id"] = trip_df["is_idle"].ne(trip_df["is_idle"].shift()).cumsum()
    return trip_df


def make_run_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the continuous idle/moving runs created by add_run_ids."""
    run_table = (
        df.groupby(["veh_id", "trip_id", "run_id"])
        .agg(
            is_idle=("is_idle", "first"),
            start_time_s=("time_s", "min"),
            end_time_s=("time_s", "max"),
            n_points=("time_s", "count"),
            min_speed_kmh=("speed_kmh_clean", "min"),
            mean_speed_kmh=("speed_kmh_clean", "mean"),
            max_speed_kmh=("speed_kmh_clean", "max"),
        )
        .reset_index()
    )
    run_table["duration_s"] = run_table["end_time_s"] - run_table["start_time_s"] + 1
    return run_table


def build_microtrips(
    df: pd.DataFrame,
    min_duration_s: int,
    max_idle_duration_s: int,
    min_max_speed_kmh: float,
    min_distance_km: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create and filter microtrips.

    The first table returned is one row per microtrip. The second dataframe is
    the full 1 Hz time series with a microtrip_id attached to each row. Small
    creeping fragments and almost-stationary fragments are filtered out, because
    they tend to hurt clustering more than they help.
    """
    labeled_parts = []
    records = []
    global_microtrip_id = 0

    for (veh_id, trip_id), trip_df in df.groupby(["veh_id", "trip_id"], sort=False):
        trip_df = trip_df.copy()
        trip_df["microtrip_id"] = np.nan
        trip_runs = make_run_table(trip_df)
        trip_runs = trip_runs.sort_values("run_id").reset_index(drop=True)

        for i, row in trip_runs.iterrows():
            if bool(row["is_idle"]):
                continue

            start_run_id = row["run_id"]
            end_run_id = start_run_id

            if i + 1 < len(trip_runs):
                next_row = trip_runs.iloc[i + 1]
                if bool(next_row["is_idle"]):
                    end_run_id = next_row["run_id"]

            mask = (trip_df["run_id"] >= start_run_id) & (trip_df["run_id"] <= end_run_id)
            segment = trip_df.loc[mask].copy()
            if segment.empty:
                continue

            trip_df.loc[mask, "microtrip_id"] = global_microtrip_id

            duration_s = len(segment)
            idle_duration_s = int(segment["is_idle"].sum())
            moving_duration_s = duration_s - idle_duration_s
            distance_km = segment["speed_mps_clean"].sum() / 1000.0

            records.append(
                {
                    "veh_id": veh_id,
                    "trip_id": trip_id,
                    "microtrip_id": global_microtrip_id,
                    "start_time_s": segment["time_s"].min(),
                    "end_time_s": segment["time_s"].max(),
                    "duration_s": duration_s,
                    "moving_duration_s": moving_duration_s,
                    "idle_duration_s": idle_duration_s,
                    "idle_pct": 100 * idle_duration_s / duration_s,
                    "distance_km": distance_km,
                    "max_speed_kmh": segment["speed_kmh_clean"].max(),
                    "mean_speed_kmh": segment["speed_kmh_clean"].mean(),
                    "mean_moving_speed_kmh": segment.loc[
                        ~segment["is_idle"], "speed_kmh_clean"
                    ].mean(),
                }
            )

            global_microtrip_id += 1

        labeled_parts.append(trip_df)

    if not records:
        raise ValueError("No microtrips were created. Check the idle threshold or input data.")

    microtrip_table = pd.DataFrame(records)
    microtrip_table["valid_microtrip"] = (
        (microtrip_table["duration_s"] >= min_duration_s)
        & (microtrip_table["idle_duration_s"] <= max_idle_duration_s)
        & (microtrip_table["max_speed_kmh"] >= min_max_speed_kmh)
        & (microtrip_table["distance_km"] >= min_distance_km)
    )

    df_labeled = pd.concat(labeled_parts, ignore_index=True)
    valid_ids = set(microtrip_table.loc[microtrip_table["valid_microtrip"], "microtrip_id"])
    df_labeled["valid_microtrip"] = df_labeled["microtrip_id"].isin(valid_ids)

    return microtrip_table, df_labeled


# -----------------------------
# Feature extraction
# -----------------------------
# These features are the bridge between the raw speed trace and clustering. Most
# of them are standard drive-cycle descriptors: time in each driving mode, speed
# statistics, acceleration statistics, distance, and speed-bin percentages.


def extract_features_from_segment(
    segment: pd.DataFrame,
    idle_speed_kmh: float,
    accel_threshold: float,
) -> Dict[str, float]:
    """Extract drive-cycle features from one time-series segment.

    The driving modes are mutually exclusive: a row is idle, accelerating,
    decelerating, or cruising. This matters because the four percentages should
    add up to 100 instead of double-counting idle points.
    """
    segment = segment.sort_values("time_s").copy()

    speed_kmh = segment["speed_kmh_clean"].to_numpy(dtype=float)
    speed_mps = segment["speed_mps_clean"].to_numpy(dtype=float)
    accel = segment["accel_mps2_clean"].fillna(0).to_numpy(dtype=float)

    if "elevation_m" in segment.columns:
        elevation = segment["elevation_m"].to_numpy(dtype=float)
    else:
        elevation = np.zeros(len(segment), dtype=float)

    duration_s = len(segment)
    distance_km = speed_mps.sum() / 1000.0

    is_idle = speed_kmh <= idle_speed_kmh
    is_moving = ~is_idle
    is_accel = is_moving & (accel > accel_threshold)
    is_decel = is_moving & (accel < -accel_threshold)
    is_cruise = is_moving & (~is_accel) & (~is_decel)

    moving_speed = speed_kmh[is_moving]
    positive_accel = accel[is_accel]
    negative_accel = accel[is_decel]

    speed_bin_counts, _ = np.histogram(speed_kmh, bins=SPEED_BINS_KMH)
    speed_bin_pct = 100 * speed_bin_counts / duration_s

    features = {
        "duration_s": duration_s,
        "distance_km": distance_km,
        "max_speed_kmh": float(np.max(speed_kmh)),
        "mean_speed_kmh": float(np.mean(speed_kmh)),
        "mean_moving_speed_kmh": float(np.mean(moving_speed)) if len(moving_speed) else 0.0,
        "std_speed_kmh": float(np.std(speed_kmh, ddof=0)),
        "idle_pct": float(100 * np.mean(is_idle)),
        "accel_pct": float(100 * np.mean(is_accel)),
        "decel_pct": float(100 * np.mean(is_decel)),
        "cruise_pct": float(100 * np.mean(is_cruise)),
        "mean_positive_accel_mps2": float(np.mean(positive_accel)) if len(positive_accel) else 0.0,
        "mean_decel_abs_mps2": float(np.mean(np.abs(negative_accel))) if len(negative_accel) else 0.0,
        "std_accel_mps2": float(np.std(accel, ddof=0)),
        "min_elevation_m": float(np.min(elevation)),
        "max_elevation_m": float(np.max(elevation)),
        "elevation_change_m": float(elevation[-1] - elevation[0]),
        "elevation_range_m": float(np.max(elevation) - np.min(elevation)),
    }

    for name, value in zip(SPEED_BIN_NAMES, speed_bin_pct):
        features[name] = float(value)

    return features


def extract_microtrip_features(
    df_labeled: pd.DataFrame,
    valid_microtrips: pd.DataFrame,
    idle_speed_kmh: float,
    accel_threshold: float,
) -> pd.DataFrame:
    """Build the feature matrix used by PCA and K-means."""
    records = []

    for _, mt in valid_microtrips.iterrows():
        mt_id = int(mt["microtrip_id"])
        segment = df_labeled[
            (df_labeled["microtrip_id"] == mt_id) & (df_labeled["valid_microtrip"])
        ].copy()

        if segment.empty:
            continue

        features = extract_features_from_segment(segment, idle_speed_kmh, accel_threshold)
        features.update(
            {
                "veh_id": segment["veh_id"].iloc[0],
                "trip_id": segment["trip_id"].iloc[0],
                "microtrip_id": mt_id,
                "start_time_s": segment["time_s"].min(),
                "end_time_s": segment["time_s"].max(),
            }
        )
        records.append(features)

    features_df = pd.DataFrame(records)
    if features_df.empty:
        raise ValueError("No valid microtrip features were extracted.")

    meta_cols = ["veh_id", "trip_id", "microtrip_id", "start_time_s", "end_time_s"]
    other_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df = features_df[meta_cols + other_cols]

    features_df["mode_pct_sum"] = (
        features_df["idle_pct"]
        + features_df["accel_pct"]
        + features_df["decel_pct"]
        + features_df["cruise_pct"]
    )
    features_df["speed_bin_pct_sum"] = features_df[SPEED_BIN_NAMES].sum(axis=1)
    return features_df


def extract_features_from_speed(
    speed_kmh: Iterable[float],
    idle_speed_kmh: float,
    accel_threshold: float,
) -> Dict[str, float]:
    """Convenience wrapper used for validating the final stitched cycle."""
    speed_kmh = np.asarray(list(speed_kmh), dtype=float)
    speed_mps = speed_kmh / 3.6
    accel = np.diff(speed_mps, prepend=speed_mps[0])

    temp = pd.DataFrame(
        {
            "time_s": np.arange(len(speed_kmh)),
            "speed_kmh_clean": speed_kmh,
            "speed_mps_clean": speed_mps,
            "accel_mps2_clean": accel,
            "elevation_m": np.zeros(len(speed_kmh)),
        }
    )
    return extract_features_from_segment(temp, idle_speed_kmh, accel_threshold)


# -----------------------------
# PCA and K-means
# -----------------------------
# PCA keeps the clustering stable by reducing correlated features into a smaller
# set of components. K-means then groups microtrips into broad driving regimes.


def get_ml_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """Pick the columns that should be used for PCA/K-means.

    Elevation is saved for analysis but excluded from clustering here. Otherwise
    the model can start grouping route geography instead of driving behavior.
    """
    exclude = {
        "veh_id",
        "trip_id",
        "microtrip_id",
        "start_time_s",
        "end_time_s",
        "mode_pct_sum",
        "speed_bin_pct_sum",
        # Keep clustering focused on driving behavior, not the route elevation.
        "min_elevation_m",
        "max_elevation_m",
        "elevation_change_m",
        "elevation_range_m",
    }

    cols = [c for c in features_df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(features_df[c])]

    # PCA does not need columns that never change.
    return [c for c in cols if features_df[c].std(ddof=0) > 0]


def cluster_microtrips(
    features_df: pd.DataFrame,
    n_clusters: int,
    n_pcs: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Run standardization, PCA, and K-means on valid microtrips."""
    feature_cols = get_ml_feature_columns(features_df)
    if len(feature_cols) == 0:
        raise ValueError("No usable ML features after dropping zero-variance columns.")

    if len(features_df) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} valid microtrips for K-means; found {len(features_df)}."
        )

    X = features_df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pcs = min(n_pcs, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    pc_df = pd.DataFrame(X_pca, columns=pc_cols)
    features_pca = pd.concat([features_df.reset_index(drop=True), pc_df], axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    features_pca["cluster"] = kmeans.fit_predict(X_pca)

    cluster_speed_order = (
        features_pca.groupby("cluster")["mean_moving_speed_kmh"].mean().sort_values()
    )
    speed_names = ["low_speed", "medium_speed", "high_speed"]
    if n_clusters != 3:
        speed_names = [f"cluster_{i}" for i in range(n_clusters)]

    cluster_name_map = {
        int(cluster_id): speed_names[i] for i, cluster_id in enumerate(cluster_speed_order.index)
    }
    features_pca["cluster_name"] = features_pca["cluster"].map(cluster_name_map)

    cluster_summary = (
        features_pca.groupby(["cluster", "cluster_name"])
        .agg(
            n_microtrips=("microtrip_id", "count"),
            total_duration_s=("duration_s", "sum"),
            total_distance_km=("distance_km", "sum"),
            mean_duration_s=("duration_s", "mean"),
            mean_speed_kmh=("mean_speed_kmh", "mean"),
            mean_moving_speed_kmh=("mean_moving_speed_kmh", "mean"),
            mean_max_speed_kmh=("max_speed_kmh", "mean"),
            mean_idle_pct=("idle_pct", "mean"),
            mean_accel_pct=("accel_pct", "mean"),
            mean_decel_pct=("decel_pct", "mean"),
            mean_cruise_pct=("cruise_pct", "mean"),
        )
        .reset_index()
    )

    total_duration = features_pca["duration_s"].sum()
    cluster_proportions = (
        features_pca.groupby(["cluster", "cluster_name"])
        .agg(
            n_microtrips=("microtrip_id", "count"),
            total_duration_s=("duration_s", "sum"),
            total_distance_km=("distance_km", "sum"),
        )
        .reset_index()
    )
    cluster_proportions["duration_pct"] = (
        100 * cluster_proportions["total_duration_s"] / total_duration
    )
    cluster_proportions["microtrip_count_pct"] = (
        100 * cluster_proportions["n_microtrips"] / len(features_pca)
    )

    pca_report = pd.DataFrame(
        {
            "PC": pc_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    return features_pca, cluster_summary, cluster_proportions, pca_report, feature_cols


# -----------------------------
# Optimized subset selection
# -----------------------------
# The final cycle is not made by taking one representative fragment per cluster.
# Instead, I select a subset of real microtrips whose combined statistics best
# match the target trip and the target cluster proportions.


def add_start_end_speed(features: pd.DataFrame, df_labeled: pd.DataFrame) -> pd.DataFrame:
    """Attach start/end speed to each microtrip for sanity checks."""
    records = []
    for mt_id in features["microtrip_id"]:
        seg = df_labeled[(df_labeled["microtrip_id"] == mt_id) & (df_labeled["valid_microtrip"])]
        seg = seg.sort_values("time_s")
        records.append(
            {
                "microtrip_id": int(mt_id),
                "start_speed_kmh": float(seg["speed_kmh_clean"].iloc[0]),
                "end_speed_kmh": float(seg["speed_kmh_clean"].iloc[-1]),
            }
        )
    return features.merge(pd.DataFrame(records), on="microtrip_id", how="left")


def make_candidate_df(
    features_clusters: pd.DataFrame,
    exclude_truncated_start: bool,
    idle_speed_kmh: float,
) -> pd.DataFrame:
    """Choose which microtrips are allowed in the final constructed cycle."""
    candidates = features_clusters.copy()

    if exclude_truncated_start:
        # A first fragment can begin while the vehicle is already moving because the
        # original recording starts mid-trip. It is safer not to use that in a constructed cycle.
        first_start = candidates["start_time_s"].min()
        mask_truncated = (candidates["start_time_s"] == first_start) & (
            candidates["start_speed_kmh"] > idle_speed_kmh
        )
        candidates = candidates.loc[~mask_truncated].copy()

    return candidates.reset_index(drop=True)


def prepare_optimizer_data(candidate_df: pd.DataFrame, cluster_targets: Dict[str, float]) -> Dict:
    """Pack candidate values into arrays so subset search stays quick."""
    return {
        "dur": candidate_df["duration_s"].to_numpy(dtype=float),
        "dist": candidate_df["distance_km"].to_numpy(dtype=float),
        "mean_speed": candidate_df["mean_speed_kmh"].to_numpy(dtype=float),
        "mean_moving_speed": candidate_df["mean_moving_speed_kmh"].to_numpy(dtype=float),
        "max_speed": candidate_df["max_speed_kmh"].to_numpy(dtype=float),
        "std_speed": candidate_df["std_speed_kmh"].to_numpy(dtype=float),
        "idle_pct": candidate_df["idle_pct"].to_numpy(dtype=float),
        "accel_pct": candidate_df["accel_pct"].to_numpy(dtype=float),
        "decel_pct": candidate_df["decel_pct"].to_numpy(dtype=float),
        "cruise_pct": candidate_df["cruise_pct"].to_numpy(dtype=float),
        "mean_pos_acc": candidate_df["mean_positive_accel_mps2"].to_numpy(dtype=float),
        "mean_dec_abs": candidate_df["mean_decel_abs_mps2"].to_numpy(dtype=float),
        "std_accel": candidate_df["std_accel_mps2"].to_numpy(dtype=float),
        "speed_bins": candidate_df[SPEED_BIN_NAMES].to_numpy(dtype=float),
        "cluster_names": candidate_df["cluster_name"].to_numpy(),
        "microtrip_ids": candidate_df["microtrip_id"].to_numpy(dtype=int),
        "cluster_target_names": list(cluster_targets.keys()),
    }


def aggregate_subset_fast(idx: np.ndarray, opt: Dict, initial_idle_s: int) -> Dict:
    """Fast approximate aggregate features for a selected subset."""
    idx = np.asarray(idx, dtype=int)
    dur = opt["dur"][idx]

    total_microtrip_duration = float(dur.sum())
    total_duration = total_microtrip_duration + initial_idle_s
    total_distance = float(opt["dist"][idx].sum())

    mean_speed_values = opt["mean_speed"][idx]
    mean_speed = float((dur * mean_speed_values).sum() / total_duration)

    idle_pct = opt["idle_pct"][idx]
    moving_duration = dur * (100 - idle_pct) / 100
    total_moving_duration = moving_duration.sum()
    if total_moving_duration > 0:
        mean_moving_speed = float((moving_duration * opt["mean_moving_speed"][idx]).sum() / total_moving_duration)
    else:
        mean_moving_speed = 0.0

    idle_count = float((dur * idle_pct / 100).sum() + initial_idle_s)
    accel_count = float((dur * opt["accel_pct"][idx] / 100).sum())
    decel_count = float((dur * opt["decel_pct"][idx] / 100).sum())
    cruise_count = float((dur * opt["cruise_pct"][idx] / 100).sum())

    std_speed_values = opt["std_speed"][idx]
    ev2 = (dur * (std_speed_values**2 + mean_speed_values**2)).sum() / total_duration
    std_speed = float(np.sqrt(max(0.0, ev2 - mean_speed**2)))

    std_accel = float(np.sqrt((dur * opt["std_accel"][idx] ** 2).sum() / total_duration))

    if accel_count > 0:
        mean_pos_acc = float(((dur * opt["accel_pct"][idx] / 100) * opt["mean_pos_acc"][idx]).sum() / accel_count)
    else:
        mean_pos_acc = 0.0

    if decel_count > 0:
        mean_dec_abs = float(((dur * opt["decel_pct"][idx] / 100) * opt["mean_dec_abs"][idx]).sum() / decel_count)
    else:
        mean_dec_abs = 0.0

    speed_bin_counts = (dur[:, None] * opt["speed_bins"][idx] / 100).sum(axis=0)
    speed_bin_counts[0] += initial_idle_s
    speed_bin_pct = 100 * speed_bin_counts / total_duration

    cluster_pct = {}
    selected_cluster_names = opt["cluster_names"][idx]
    for cname in opt["cluster_target_names"]:
        c_dur = dur[selected_cluster_names == cname].sum()
        cluster_pct[cname] = float(100 * c_dur / total_microtrip_duration) if total_microtrip_duration else 0.0

    out = {
        "duration_s": total_duration,
        "selected_microtrip_duration_s": total_microtrip_duration,
        "distance_km": total_distance,
        "mean_speed_kmh": mean_speed,
        "mean_moving_speed_kmh": mean_moving_speed,
        "max_speed_kmh": float(opt["max_speed"][idx].max()),
        "std_speed_kmh": std_speed,
        "idle_pct": 100 * idle_count / total_duration,
        "accel_pct": 100 * accel_count / total_duration,
        "decel_pct": 100 * decel_count / total_duration,
        "cruise_pct": 100 * cruise_count / total_duration,
        "mean_positive_accel_mps2": mean_pos_acc,
        "mean_decel_abs_mps2": mean_dec_abs,
        "std_accel_mps2": std_accel,
        "cluster_pct": cluster_pct,
    }
    for name, value in zip(SPEED_BIN_NAMES, speed_bin_pct):
        out[name] = float(value)
    return out


def relative_error(value: float, target_value: float) -> float:
    if abs(target_value) < 1e-9:
        return abs(value - target_value)
    return abs(value - target_value) / abs(target_value)


def score_candidate(agg: Dict, target: Dict, target_duration_s: float, cluster_targets: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Score one candidate final cycle. Lower is better.

    The weights are deliberately simple: duration, mean speed, moving speed,
    driving-mode percentages, speed-bin percentages, cluster proportions, and
    acceleration profile all contribute to the final score.
    """
    score = 0.0

    score += 3.0 * relative_error(agg["duration_s"], target_duration_s)
    score += 2.0 * relative_error(agg["mean_speed_kmh"], target["mean_speed_kmh"])
    score += 1.5 * relative_error(agg["mean_moving_speed_kmh"], target["mean_moving_speed_kmh"])
    score += 0.7 * relative_error(agg["std_speed_kmh"], target["std_speed_kmh"])

    mode_error_pp = float(
        np.mean(
            [
                abs(agg["idle_pct"] - target["idle_pct"]),
                abs(agg["accel_pct"] - target["accel_pct"]),
                abs(agg["decel_pct"] - target["decel_pct"]),
                abs(agg["cruise_pct"] - target["cruise_pct"]),
            ]
        )
    )
    score += 2.0 * mode_error_pp / 100

    speed_bin_error_pp = float(np.mean([abs(agg[name] - target[name]) for name in SPEED_BIN_NAMES]))
    score += 2.0 * speed_bin_error_pp / 100

    cluster_error_pp = float(
        np.mean([abs(agg["cluster_pct"].get(cname, 0.0) - pct) for cname, pct in cluster_targets.items()])
    )
    score += 1.0 * cluster_error_pp / 100

    score += 0.5 * relative_error(agg["mean_positive_accel_mps2"], target["mean_positive_accel_mps2"])
    score += 0.5 * relative_error(agg["mean_decel_abs_mps2"], target["mean_decel_abs_mps2"])

    return float(score), mode_error_pp, speed_bin_error_pp, cluster_error_pp


def evaluate_subset_fast(
    idx: np.ndarray,
    opt: Dict,
    target: Dict,
    target_duration_s: float,
    cluster_targets: Dict[str, float],
    initial_idle_s: int,
) -> Dict:
    agg = aggregate_subset_fast(idx, opt, initial_idle_s)
    score, mode_error_pp, speed_bin_error_pp, cluster_error_pp = score_candidate(
        agg, target, target_duration_s, cluster_targets
    )
    return {
        "score": score,
        "duration_s": agg["duration_s"],
        "distance_km": agg["distance_km"],
        "mean_speed_kmh": agg["mean_speed_kmh"],
        "mean_moving_speed_kmh": agg["mean_moving_speed_kmh"],
        "max_speed_kmh": agg["max_speed_kmh"],
        "idle_pct": agg["idle_pct"],
        "accel_pct": agg["accel_pct"],
        "decel_pct": agg["decel_pct"],
        "cruise_pct": agg["cruise_pct"],
        "std_speed_kmh": agg["std_speed_kmh"],
        "mode_error_pp": mode_error_pp,
        "speed_bin_error_pp": speed_bin_error_pp,
        "cluster_error_pp": cluster_error_pp,
        "n_microtrips": len(idx),
        "indices": idx.tolist(),
        "selected_microtrip_ids": opt["microtrip_ids"][idx].astype(int).tolist(),
    }


def optimize_subset_exhaustive(
    candidate_df: pd.DataFrame,
    target: Dict,
    target_duration_s: float,
    cluster_targets: Dict[str, float],
    initial_idle_s: int,
    duration_window_s: int,
) -> pd.DataFrame:
    """Try every valid subset of candidate microtrips.

    For a small demo file this is better than using a heuristic optimizer because
    it actually checks all valid combinations inside the duration window.
    """
    n = len(candidate_df)
    opt = prepare_optimizer_data(candidate_df, cluster_targets)
    durations = opt["dur"]
    required_clusters = set(cluster_targets.keys())
    cluster_arr = opt["cluster_names"]

    min_duration = target_duration_s - duration_window_s
    max_duration = target_duration_s + duration_window_s

    records = []
    bit_values = 1 << np.arange(n)

    for mask in range(1, 1 << n):
        idx = np.flatnonzero((mask & bit_values) > 0)
        final_duration = durations[idx].sum() + initial_idle_s
        if final_duration < min_duration or final_duration > max_duration:
            continue
        if set(cluster_arr[idx]) != required_clusters:
            continue

        records.append(
            evaluate_subset_fast(
                idx,
                opt,
                target,
                target_duration_s,
                cluster_targets,
                initial_idle_s,
            )
        )

    if not records:
        raise ValueError("No valid subset found. Try a wider duration window or more input data.")

    return pd.DataFrame(records).sort_values("score").reset_index(drop=True)


def optimize_subset_random(
    candidate_df: pd.DataFrame,
    target: Dict,
    target_duration_s: float,
    cluster_targets: Dict[str, float],
    initial_idle_s: int,
    duration_window_s: int,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    """Fallback search for larger datasets where exhaustive search is too slow."""
    rng = np.random.default_rng(random_state)
    n = len(candidate_df)
    opt = prepare_optimizer_data(candidate_df, cluster_targets)
    durations = opt["dur"]
    cluster_arr = opt["cluster_names"]
    required_clusters = set(cluster_targets.keys())

    min_duration = target_duration_s - duration_window_s
    max_duration = target_duration_s + duration_window_s
    p = min(0.95, max(0.05, (target_duration_s - initial_idle_s) / durations.sum()))

    records = []
    seen = set()

    for _ in range(iterations):
        selected = rng.random(n) < p
        if not selected.any():
            selected[rng.integers(0, n)] = True

        idx = np.flatnonzero(selected)
        key = tuple(idx.tolist())
        if key in seen:
            continue
        seen.add(key)

        final_duration = durations[idx].sum() + initial_idle_s
        if final_duration < min_duration or final_duration > max_duration:
            continue
        if set(cluster_arr[idx]) != required_clusters:
            continue

        records.append(
            evaluate_subset_fast(
                idx,
                opt,
                target,
                target_duration_s,
                cluster_targets,
                initial_idle_s,
            )
        )

    if not records:
        raise ValueError("Random search found no valid subset. Increase iterations or duration window.")

    return pd.DataFrame(records).sort_values("score").reset_index(drop=True)


# -----------------------------
# Cycle stitching and validation
# -----------------------------
# Once the subset is selected, the final cycle is simply those real microtrips
# stitched together. The last transition cleanup pass only protects against speed
# jumps at the boundaries between fragments.


def stitch_microtrips(
    df_labeled: pd.DataFrame,
    selected_ids: List[int],
    features_clusters: pd.DataFrame,
    initial_idle_s: int,
) -> pd.DataFrame:
    """Create the final speed-time cycle from selected microtrips."""
    parts = []
    current_time = 0
    segment_order = 0
    cluster_lookup = features_clusters.set_index("microtrip_id")["cluster_name"].to_dict()

    if initial_idle_s > 0:
        parts.append(
            pd.DataFrame(
                {
                    "cycle_time_s": np.arange(current_time, current_time + initial_idle_s),
                    "source_microtrip_id": -1,
                    "source_cluster_name": "initial_idle",
                    "speed_kmh": 0.0,
                    "speed_mps": 0.0,
                    "original_time_s": np.nan,
                    "lat": np.nan,
                    "lon": np.nan,
                    "elevation_m": np.nan,
                    "segment_order": segment_order,
                }
            )
        )
        current_time += initial_idle_s
        segment_order += 1

    for mt_id in selected_ids:
        seg = df_labeled[
            (df_labeled["microtrip_id"] == mt_id) & (df_labeled["valid_microtrip"])
        ].sort_values("time_s")

        if seg.empty:
            continue

        out = pd.DataFrame(
            {
                "cycle_time_s": np.arange(current_time, current_time + len(seg)),
                "source_microtrip_id": int(mt_id),
                "source_cluster_name": cluster_lookup.get(mt_id, "unknown"),
                "speed_kmh": seg["speed_kmh_clean"].to_numpy(dtype=float),
                "speed_mps": seg["speed_mps_clean"].to_numpy(dtype=float),
                "original_time_s": seg["time_s"].to_numpy(dtype=float),
                "lat": seg["lat"].to_numpy(dtype=float),
                "lon": seg["lon"].to_numpy(dtype=float),
                "elevation_m": seg["elevation_m"].to_numpy(dtype=float),
                "segment_order": segment_order,
            }
        )
        parts.append(out)
        current_time += len(out)
        segment_order += 1

    cycle = pd.concat(parts, ignore_index=True)
    cycle["accel_mps2"] = cycle["speed_mps"].diff().fillna(0)
    return cycle


def clean_final_cycle_transitions(
    cycle: pd.DataFrame,
    max_accel: float,
    max_decel: float,
) -> pd.DataFrame:
    """Limit acceleration at fragment boundaries after stitching."""
    cycle = cycle.copy()
    speed = cycle["speed_mps"].to_numpy(dtype=float)
    cleaned = speed.copy()

    for i in range(1, len(cleaned)):
        accel = cleaned[i] - cleaned[i - 1]  # dt = 1 s
        if accel > max_accel:
            cleaned[i] = cleaned[i - 1] + max_accel
        elif accel < max_decel:
            cleaned[i] = cleaned[i - 1] + max_decel
        cleaned[i] = max(cleaned[i], 0.0)

    cycle["speed_mps_final"] = cleaned
    cycle["speed_kmh_final"] = cleaned * 3.6
    cycle["accel_mps2_final"] = cycle["speed_mps_final"].diff().fillna(0)
    return cycle


def compare_features(
    final_features: Dict,
    original_features: Dict,
    target_duration_s: float,
) -> pd.DataFrame:
    """Compare the final cycle to a fair target.

    The original trip may be longer than the target cycle, so duration and
    distance are compared against the requested target duration. The other
    statistics use the original trip as the reference.
    """
    fair_target = dict(original_features)
    fair_target["duration_s"] = target_duration_s
    fair_target["distance_km"] = original_features["mean_speed_kmh"] * target_duration_s / 3600.0

    rows = []
    for key, value in final_features.items():
        if key not in fair_target:
            continue

        target_value = fair_target[key]
        abs_error = value - target_value
        pct_error = np.nan if abs(target_value) < 1e-9 else 100 * abs_error / target_value

        rows.append(
            {
                "feature": key,
                "target": target_value,
                "optimized_constructed": value,
                "abs_error": abs_error,
                "pct_error": pct_error,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Pipeline runner
# -----------------------------
# This is the full repo workflow. Each major intermediate table is written out so
# a reviewer can check the data cleaning, microtrip filtering, clustering, and
# final selection without editing the script.


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete drive-cycle construction workflow."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    print(f"Reading: {input_path}")
    raw_df = pd.read_csv(input_path)
    raw_df = standardize_columns(raw_df)

    print("Preprocessing and resampling to 1 Hz...")
    df_1hz = preprocess_to_1hz(raw_df, args.max_accel, args.max_decel, args.max_speed_kmh)
    df_1hz.to_csv(output_dir / "cleaned_1hz_timeseries.csv", index=False)

    print("Detecting idle periods and building microtrips...")
    labeled_parts = []
    for _, trip_df in df_1hz.groupby(["veh_id", "trip_id"], sort=False):
        labeled_parts.append(add_run_ids(trip_df, args.idle_speed_kmh))
    df_runs = pd.concat(labeled_parts, ignore_index=True)

    run_table = make_run_table(df_runs)
    run_table.to_csv(output_dir / "run_table.csv", index=False)

    microtrip_table, df_labeled = build_microtrips(
        df_runs,
        min_duration_s=args.min_microtrip_duration,
        max_idle_duration_s=args.max_idle_duration,
        min_max_speed_kmh=args.min_max_speed,
        min_distance_km=args.min_distance,
    )
    microtrip_table.to_csv(output_dir / "microtrip_table.csv", index=False)
    df_labeled.to_csv(output_dir / "labeled_timeseries.csv", index=False)

    valid_microtrips = microtrip_table[microtrip_table["valid_microtrip"]].copy()
    valid_microtrips.to_csv(output_dir / "valid_microtrips.csv", index=False)

    print(f"Valid microtrips: {len(valid_microtrips)} / {len(microtrip_table)}")

    print("Extracting microtrip features...")
    features_df = extract_microtrip_features(
        df_labeled,
        valid_microtrips,
        idle_speed_kmh=args.idle_speed_kmh,
        accel_threshold=args.accel_threshold,
    )
    features_df.to_csv(output_dir / "microtrip_features.csv", index=False)

    original_features = extract_features_from_segment(
        df_1hz.copy(),
        idle_speed_kmh=args.idle_speed_kmh,
        accel_threshold=args.accel_threshold,
    )
    pd.DataFrame([original_features]).to_csv(output_dir / "original_target_features.csv", index=False)

    print("Running PCA and K-means...")
    features_clusters, cluster_summary, cluster_proportions, pca_report, feature_cols = cluster_microtrips(
        features_df,
        n_clusters=args.n_clusters,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
    )
    features_clusters = add_start_end_speed(features_clusters, df_labeled)

    features_clusters.to_csv(output_dir / "microtrip_clusters.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    cluster_proportions.to_csv(output_dir / "cluster_proportions.csv", index=False)
    pca_report.to_csv(output_dir / "pca_report.csv", index=False)
    (output_dir / "ml_feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    candidate_df = make_candidate_df(
        features_clusters,
        exclude_truncated_start=not args.keep_truncated_start,
        idle_speed_kmh=args.idle_speed_kmh,
    )

    if len(candidate_df) < args.n_clusters:
        raise ValueError("Not enough candidate microtrips after filtering/exclusions.")

    cluster_targets = cluster_proportions.set_index("cluster_name")["duration_pct"].to_dict()

    print(f"Optimizing selected microtrips from {len(candidate_df)} candidates...")
    if len(candidate_df) <= args.max_exhaustive_candidates:
        search_results = optimize_subset_exhaustive(
            candidate_df,
            target=original_features,
            target_duration_s=args.target_duration,
            cluster_targets=cluster_targets,
            initial_idle_s=args.initial_idle,
            duration_window_s=args.duration_window,
        )
        search_method = "exhaustive"
    else:
        search_results = optimize_subset_random(
            candidate_df,
            target=original_features,
            target_duration_s=args.target_duration,
            cluster_targets=cluster_targets,
            initial_idle_s=args.initial_idle,
            duration_window_s=args.duration_window,
            iterations=args.random_search_iterations,
            random_state=args.random_state,
        )
        search_method = "random"

    search_results.head(args.save_top_n).to_csv(output_dir / "top_search_results.csv", index=False)

    best = search_results.iloc[0]
    best_ids = list(best["selected_microtrip_ids"])
    if isinstance(best_ids, str):
        # Needed only if the dataframe was read back from disk; kept here for safety.
        best_ids = json.loads(best_ids)

    selected_microtrips = candidate_df[candidate_df["microtrip_id"].isin(best_ids)].copy()
    selected_microtrips = selected_microtrips.sort_values("start_time_s").reset_index(drop=True)
    selected_microtrips.to_csv(output_dir / "optimized_selected_microtrips.csv", index=False)

    print("Stitching final cycle...")
    final_cycle = stitch_microtrips(
        df_labeled,
        selected_microtrips["microtrip_id"].astype(int).tolist(),
        features_clusters,
        initial_idle_s=args.initial_idle,
    )
    final_cycle = clean_final_cycle_transitions(final_cycle, args.max_accel, args.max_decel)
    final_cycle.to_csv(output_dir / "optimized_drive_cycle_full.csv", index=False)

    final_simple = final_cycle[
        [
            "cycle_time_s",
            "speed_kmh_final",
            "speed_mps_final",
            "accel_mps2_final",
            "source_microtrip_id",
            "source_cluster_name",
        ]
    ].rename(
        columns={
            "cycle_time_s": "time_s",
            "speed_kmh_final": "speed_kmh",
            "speed_mps_final": "speed_mps",
            "accel_mps2_final": "accel_mps2",
        }
    )
    final_simple.to_csv(output_dir / "FINAL_optimized_representative_drive_cycle.csv", index=False)

    final_features = extract_features_from_speed(
        final_cycle["speed_kmh_final"].to_numpy(dtype=float),
        idle_speed_kmh=args.idle_speed_kmh,
        accel_threshold=args.accel_threshold,
    )
    pd.DataFrame([final_features]).to_csv(output_dir / "optimized_cycle_features.csv", index=False)

    comparison = compare_features(final_features, original_features, args.target_duration)
    comparison.to_csv(output_dir / "fair_target_vs_optimized_comparison.csv", index=False)

    key_features = [
        "duration_s",
        "distance_km",
        "mean_speed_kmh",
        "mean_moving_speed_kmh",
        "max_speed_kmh",
        "idle_pct",
        "accel_pct",
        "decel_pct",
        "cruise_pct",
        "std_speed_kmh",
        "std_accel_mps2",
    ]
    key_comparison = comparison[comparison["feature"].isin(key_features)].copy()
    key_comparison.to_csv(output_dir / "key_feature_comparison.csv", index=False)

    summary = {
        "input_file": str(input_path),
        "search_method": search_method,
        "rows_raw": int(len(raw_df)),
        "rows_1hz": int(len(df_1hz)),
        "total_microtrips": int(len(microtrip_table)),
        "valid_microtrips": int(len(valid_microtrips)),
        "candidate_microtrips": int(len(candidate_df)),
        "selected_microtrips": [int(x) for x in selected_microtrips["microtrip_id"].tolist()],
        "target_duration_s": float(args.target_duration),
        "final_duration_s": int(len(final_cycle)),
        "final_distance_km": float(final_cycle["speed_mps_final"].sum() / 1000),
        "final_mean_speed_kmh": float(final_cycle["speed_kmh_final"].mean()),
        "final_moving_mean_speed_kmh": float(
            final_cycle.loc[final_cycle["speed_kmh_final"] > args.idle_speed_kmh, "speed_kmh_final"].mean()
        ),
        "final_max_speed_kmh": float(final_cycle["speed_kmh_final"].max()),
        "final_idle_pct": float(100 * (final_cycle["speed_kmh_final"] <= args.idle_speed_kmh).mean()),
        "best_score": float(best["score"]),
    }
    save_json(output_dir / "run_summary.json", summary)

    print("\nDone. Main outputs:")
    print(f"  {output_dir / 'FINAL_optimized_representative_drive_cycle.csv'}")
    print(f"  {output_dir / 'key_feature_comparison.csv'}")
    print(f"  {output_dir / 'run_summary.json'}")
    print("\nFinal cycle summary:")
    print(json.dumps(summary, indent=2))


# -----------------------------
# CLI
# -----------------------------
# Defaults are chosen for the VED demo file used in this repo, but most values
# are exposed so the same script can be reused on other speed-time datasets.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a representative driving cycle from VED-style speed data.")

    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", default="outputs", help="Output directory.")

    parser.add_argument("--target-duration", type=float, default=1200, help="Target final cycle duration in seconds.")
    parser.add_argument("--duration-window", type=int, default=100, help="Allowed duration window around the target.")
    parser.add_argument("--initial-idle", type=int, default=5, help="Seconds of idle added at the start.")

    parser.add_argument("--idle-speed-kmh", type=float, default=1.0, help="Speed threshold used for idle/stopped points.")
    parser.add_argument("--accel-threshold", type=float, default=0.10, help="Small acceleration threshold for accel/decel/cruise classification.")
    parser.add_argument("--max-accel", type=float, default=4.0, help="Maximum allowed acceleration in m/s^2.")
    parser.add_argument("--max-decel", type=float, default=-8.0, help="Minimum allowed acceleration in m/s^2.")
    parser.add_argument("--max-speed-kmh", type=float, default=160.0, help="Safety cap for vehicle speed.")

    parser.add_argument("--min-microtrip-duration", type=int, default=10, help="Minimum valid microtrip duration in seconds.")
    parser.add_argument("--max-idle-duration", type=int, default=180, help="Maximum valid idle duration inside a microtrip.")
    parser.add_argument("--min-max-speed", type=float, default=10.0, help="Minimum required max speed for a valid microtrip.")
    parser.add_argument("--min-distance", type=float, default=0.01, help="Minimum valid microtrip distance in km.")

    parser.add_argument("--n-clusters", type=int, default=3, help="Number of K-means clusters.")
    parser.add_argument("--n-pcs", type=int, default=4, help="Number of PCA components.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    parser.add_argument("--max-exhaustive-candidates", type=int, default=22, help="Use exhaustive search up to this many candidates.")
    parser.add_argument("--random-search-iterations", type=int, default=100000, help="Iterations if random search is needed.")
    parser.add_argument("--save-top-n", type=int, default=50, help="How many search results to save.")

    parser.add_argument(
        "--keep-truncated-start",
        action="store_true",
        help="Keep the first fragment even if it starts while already moving.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
