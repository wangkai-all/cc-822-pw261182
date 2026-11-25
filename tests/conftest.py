"""Shared pytest fixtures."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from app.engine.synthetic import SyntheticConfig, generate_synthetic_dataset
from app.engine.pipeline import CorrectionConfig, CorrectionPipeline


@pytest.fixture(scope="session")
def dataset():
    return generate_synthetic_dataset(SyntheticConfig(seed=20261125))


@pytest.fixture(scope="session")
def history(dataset):
    return dataset["history"]


@pytest.fixture(scope="session")
def future(dataset):
    return dataset["future"]


@pytest.fixture
def pipe_default():
    return CorrectionPipeline(CorrectionConfig(
        train_years=6, val_years=3, test_years=3,
        occ_method="trend", amount_method="qdm"))
