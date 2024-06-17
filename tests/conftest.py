from pathlib import Path

import hydra
import numpy as np
import pytest

@pytest.fixture
def sample_gage_cfg():
    with hydra.initialize(
        version_base="1.3",
        config_path="../marquette/conf/",
    ):
        cfg = hydra.compose(config_name="config")
    cfg.zone = "73"
    return cfg

@pytest.fixture
def q_prime_data() -> np.ndarray:
    q_prime_path = Path("/projects/mhpi/tbindas/marquette/tests/validated_data/q_prime_89905.npy")
    if not q_prime_path.exists():
        raise FileNotFoundError(f"File not found: {q_prime_path}")
    return np.load(q_prime_path)
