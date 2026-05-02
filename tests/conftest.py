"""Shared pytest fixtures for all test modules."""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(scope="session")
def sample_voters():
    """Small voter DataFrame for unit tests (200 rows, fixed seed)."""
    from src.data.generate_voters import generate_voters
    return generate_voters(n_voters=200, random_seed=0)


@pytest.fixture(scope="session")
def config():
    """Load project config."""
    import yaml
    cfg_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)
