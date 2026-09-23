"""Derive canonical mobility and reference-zone features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ..config import LocationSpec
from ..time import floor_15min

VALID_ZONES = ("ROOM", "HALLWAY", "OTHER")
REFERENCE_POINT_FIELDS = frozenset(("latitude", "longitude", "zone"))


def haversine_np(lat1, lon1, lat2, lon2):
    radius = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_cluster_entropy(labels):
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return 0.0, 0.0
    counts = np.unique(valid, return_counts=True)[1]
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    normalized = entropy / np.log(len(counts)) if len(counts) > 1 else 0.0
    return entropy, normalized


def validate_reference_points(
    reference_points: Sequence[Mapping[str, object]],
) -> tuple[tuple[float, float, str], ...]:
    """Validate and normalize an ordered nearest-reference point list."""
    if isinstance(reference_points, (str, bytes)) or not isinstance(reference_points, Sequence):
        raise TypeError("reference_points must be an ordered sequence")
    validated = []
    for index, point in enumerate(reference_points):
        if not isinstance(point, Mapping) or set(point) != REFERENCE_POINT_FIELDS:
            raise ValueError(
                f"reference_points[{index}] must contain exactly latitude, longitude, and zone"
            )
        latitude, longitude, zone = (
            point["latitude"],
            point["longitude"],
            point["zone"],
        )
        if isinstance(latitude, bool) or not isinstance(latitude, Real):
            raise ValueError(f"reference_points[{index}].latitude must be numeric")
        if isinstance(longitude, bool) or not isinstance(longitude, Real):
            raise ValueError(f"reference_points[{index}].longitude must be numeric")
        latitude, longitude = float(latitude), float(longitude)
        if not np.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError(f"reference_points[{index}].latitude is out of range")
        if not np.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError(f"reference_points[{index}].longitude is out of range")
        if zone not in VALID_ZONES:
            raise ValueError(
                f"reference_points[{index}].zone must be one of {', '.join(VALID_ZONES)}"
            )
        validated.append((latitude, longitude, zone))
    return tuple(validated)


def process_patient_location(
    patient_frame: pd.DataFrame,
    spec: LocationSpec,
    *,
    reference_points: Sequence[Mapping[str, object]],
) -> pd.DataFrame | None:
    validated_points = validate_reference_points(reference_points)
    patient_frame = patient_frame.sort_values("time").copy()
    patient_frame["time"] = floor_15min(patient_frame["time"])
    result = patient_frame.groupby("time")[["Latitude", "Longitude"]].median().reset_index()
    if result.empty:
        return None

    epsilon = (spec.dbscan_eps_meters / 1000) / 6371.0088
    coordinates = np.radians(result[["Latitude", "Longitude"]].values)
    if len(coordinates) < spec.dbscan_min_samples:
        result["cluster_label"] = -1
    else:
        model = DBSCAN(
            eps=epsilon,
            min_samples=spec.dbscan_min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        result["cluster_label"] = model.fit_predict(coordinates)

    times = result["time"].values
    labels = result["cluster_label"].values
    entropy, normalized = [], []
    for current in times:
        mask = (times >= current - np.timedelta64(24, "h")) & (times <= current)
        raw, norm = get_cluster_entropy(labels[mask])
        entropy.append(raw)
        normalized.append(norm)
    result["Entropy_Day"] = entropy
    result["Entropy_Day_Norm"] = normalized

    distances = haversine_np(
        result["Latitude"].shift(1).values,
        result["Longitude"].shift(1).values,
        result["Latitude"].values,
        result["Longitude"].values,
    )
    result["Distance_Traveled"] = np.nan_to_num(distances, nan=0.0)

    variability = []
    for current in times:
        mask = (times >= current - np.timedelta64(2, "h")) & (times <= current)
        latitudes = result.loc[mask, "Latitude"].values
        longitudes = result.loc[mask, "Longitude"].values
        if len(latitudes) < 2:
            variability.append(0.0)
            continue
        distances = haversine_np(
            np.full_like(latitudes, np.mean(latitudes)),
            np.full_like(longitudes, np.mean(longitudes)),
            latitudes,
            longitudes,
        )
        variability.append(np.mean(distances ** 2))
    result["Location_Variability"] = variability

    if not validated_points:
        result["place_room"] = 0
        result["Place_hallway"] = 0
        result["Place_other"] = 1
        return result

    reference_coordinates = np.array(
        [(latitude, longitude) for latitude, longitude, _ in validated_points]
    )
    reference_labels = np.array([zone for _, _, zone in validated_points])
    current = result[["Latitude", "Longitude"]].values
    distance_matrix = haversine_np(
        current[:, 0, np.newaxis],
        current[:, 1, np.newaxis],
        reference_coordinates[np.newaxis, :, 0],
        reference_coordinates[np.newaxis, :, 1],
    )
    mapped = reference_labels[np.argmin(distance_matrix, axis=1)]
    result["place_room"] = (mapped == "ROOM").astype(int)
    result["Place_hallway"] = (mapped == "HALLWAY").astype(int)
    result["Place_other"] = (mapped == "OTHER").astype(int)
    return result


def calculate_location_features(
    frame: pd.DataFrame,
    institution: str,
    spec: LocationSpec,
    *,
    reference_points: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Calculate all participant mobility frames without repository I/O."""
    results = []
    for key in frame["key"].unique():
        result = process_patient_location(
            frame.loc[frame["key"] == key].copy(),
            spec,
            reference_points=reference_points,
        )
        if result is not None:
            result["key"] = key
            result["institution"] = institution
            results.append(result)
    if not results:
        raise ValueError(f"{institution}: no location data could be processed")
    return pd.concat(results, ignore_index=True)
