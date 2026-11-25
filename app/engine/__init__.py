"""Statistical post-processing engine for precipitation forecast bias correction.

The engine deliberately separates the two statistical problems that a zero-inflated
precipitation variable presents:

1. **Occurrence** -- whether a day is wet (the point mass at zero).  See
   :mod:`app.engine.occurrence`.
2. **Amount** -- the magnitude *given* that the day is wet (a continuous, skewed,
   heavy-tailed distribution).  See :mod:`app.engine.amount`.

Everything is deterministic by default.  The stochastic occurrence option persists
the random-generator seed, the full generator state and every uniform draw so a
calibration can be reproduced exactly; frequency changes are never achieved by
silently deleting points.
"""

from .distributions import EmpiricalDist, fit_tail, TailModel
from .occurrence import OccurrenceCorrector, OCC_DETERMINISTIC, OCC_STOCHASTIC
from .amount import AmountCorrector, METHOD_QM, METHOD_QDM
from .seasons import season_key, SEASON_MODES
from .splits import contiguous_year_splits, SplitPlan
from .pipeline import (Calibration, CorrectedProduct, CorrectionPipeline,
                       CorrectionConfig, CANDIDATES)
from .synthetic import generate_synthetic_dataset, SyntheticConfig
from . import metrics

__all__ = [
    "EmpiricalDist",
    "fit_tail",
    "TailModel",
    "OccurrenceCorrector",
    "OCC_DETERMINISTIC",
    "OCC_STOCHASTIC",
    "AmountCorrector",
    "METHOD_QM",
    "METHOD_QDM",
    "season_key",
    "SEASON_MODES",
    "contiguous_year_splits",
    "SplitPlan",
    "Calibration",
    "CorrectedProduct",
    "CorrectionPipeline",
    "CorrectionConfig",
    "CANDIDATES",
    "generate_synthetic_dataset",
    "SyntheticConfig",
    "metrics",
]
