from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CosinorSpec:
    period_hours: float = 24
    trailing_window_days: int = 7
    min_window_fraction: float = 0.5
    max_missing_ratio: float = 0.5
    nonwear_hr_threshold: int = 30
    nonwear_ratio_drop: float = 0.995
    phase_unwrap_max_jump_hours: float = 6
    median_filter_kernel: int = 5


@dataclass(frozen=True)
class LocationSpec:
    dbscan_eps_meters: float = 1.5
    dbscan_min_samples: int = 3


@dataclass(frozen=True)
class WindowSpec:
    observation_hours: int = 48
    zero_padding_max_hours: int = 18
    min_case_timesteps: int = 52
    control_timesteps: int = 192
    control_samples_per_patient: int = 4
    control_max_attempts: int = 300
    control_max_nonwear_ratio: float = 0.5


@dataclass(frozen=True)
class ImputeSpec:
    invalid_control_sensor_nan_ratio: float = 0.9


@dataclass(frozen=True)
class PreprocessingSpec:
    """Immutable scientific settings for the public calculation functions."""

    cosinor: CosinorSpec = field(default_factory=CosinorSpec)
    location: LocationSpec = field(default_factory=LocationSpec)
    window: WindowSpec = field(default_factory=WindowSpec)
    impute: ImputeSpec = field(default_factory=ImputeSpec)


DEFAULT_SPEC = PreprocessingSpec()
