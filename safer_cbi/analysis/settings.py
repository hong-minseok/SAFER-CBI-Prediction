"""Result-free numeric-analysis constants."""

from ..contracts import MODEL_HORIZONS, MODEL_ORDER, PRIMARY_HORIZON


# Analysis constants
N_BOOT = 5000
N_PERM = 5000
CI_ALPHA = 0.95
GRID_POINTS = 201
RANDOM_STATE = 42      # bootstrap resampling seed (model results are fixed CSVs)

LEADTIME_CONFIGS = [("Original", 0), ("15 min", 1), ("1 hr", 4)]

# Primary-outcome reporting contract (shared across analysis stages)
CALIBRATION_METHOD = "beta"
DCA_X_MAX_FACTOR = 5.1
SPLITS = ("internal", "external")

# Manuscript discrimination domain: every model-horizon cell the input contract
# defines -- nineteen per split. The grid is deliberately saturated: a later
# reporting change can then be served from the emit layer or a checkpoint
# re-emit, never by drawing a new bootstrap.
MANUSCRIPT_DISCRIMINATION_SCHEDULE = tuple(
    (model, horizon)
    for model in MODEL_ORDER
    for horizon in MODEL_HORIZONS[model]
)


def model_order():
    """Return the stable machine-model order."""
    return list(MODEL_ORDER)
