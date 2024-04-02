import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import zarr
from tqdm import tqdm

from marquette.merit.extensions import Direction, traverse
from tests.fixtures import sample_gage_cfg, soils_data

log = logging.getLogger(__name__)


# def test_traverse_up(soils_data):
#     # Initialize test data
#     comid = 1
#     attribute = "attribute"
#     direction = Direction.up

#     # Invoke the function under test
#     result = traverse(soils_data, comid, attribute, direction)

#     # Check the result
#     assert result == 20, "Upwards traversing is broken"


# def test_traverse_down(soils_data):
#     # Initialize test data
#     comid = 5
#     attribute = "attribute"
#     direction = Direction.down

#     # Invoke the function under test
#     result = traverse(soils_data, comid, attribute, direction)

#     # Check the result
#     assert result == 20, "Downward traversing is broken"


# def test_invalid_direction(soils_data):
#     # Initialize test data
#     comid = 1
#     attribute = "attribute"
#     direction = "invalid"

#     # Invoke the function under test and check for ValueError
#     with pytest.raises(ValueError):
#         traverse(soils_data, comid, attribute, direction)


# def test_soils_extension() -> None:
#     cfg = sample_gage_cfg()
