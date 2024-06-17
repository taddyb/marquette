import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def test_graph(sample_gage_cfg: DictConfig):
    print(sample_gage_cfg)
    
    
def test_q_prime(sample_gage_cfg: DictConfig):
    root = zarr.open(sample_gage_cfg.merit.zarr_path, mode="r")
