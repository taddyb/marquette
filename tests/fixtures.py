import hydra
import geopandas as gpd
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import pytest

@pytest.fixture()
def soils_data():
    gdf = gpd.GeoDataFrame({
        'COMID': [1, 2, 3, 4, 5],
        'up1': [2, 3, 0.0, 0.0, 2],
        'NextDownID': [0.0, 1, 0.0, 0.0, 2],
        'attribute': [np.nan, 20, 30, np.nan, np.nan]
    })
    return gdf

def sample_gage_cfg():
    with hydra.initialize(
        version_base="1.3",
        config_path="../marquette/conf/",
    ):
        cfg = hydra.compose(config_name="config")
    cfg.zone = "73"
    return cfg
