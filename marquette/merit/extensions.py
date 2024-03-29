import logging
from enum import Enum
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.dataframe.io.io import from_pandas
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

class Direction(Enum):
    up = "up"
    down = "down"

def traverse(gdf: gpd.GeoDataFrame, comid: float, attribute: str, direction: Direction) :
    if direction == Direction.up:
        # Using only the up1 dir as we know it always exists
        _traverse = "up1"
    elif direction == Direction.down:
        _traverse = "NextDownID"
    else:
        raise ValueError("Invalid direction. Look at the Direction Enum and ")
    if comid == 0:
        return np.nan
    else:
        fill_value = gdf[attribute][gdf.COMID == comid].values
        if np.isnan(fill_value)[0]:
            next_id = gdf[_traverse][gdf.COMID == comid].values[0]
            return traverse(gdf, next_id, attribute, direction)
        else:
            return fill_value
            

def spatial_nan_filter(attribute: str, attr_values: np.ndarray, mapping: np.ndarray, polyline_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Filling NaN values based on the predictions up or downstream
    Using the min value of the attribute is no Up or Downstream values are available
    """
    _attr_mapped = attr_values[mapping]
    mask = np.isnan(_attr_mapped)
    if mask.sum() == 0:
        return _attr_mapped
    else:
        nan_idx = mapping[mask]
        nan_fill = np.empty_like(nan_idx, dtype=float)
        for i, idx in enumerate(tqdm(nan_idx, desc=f"\rFilling NaN values for {attribute}")):
            downstream_comid = polyline_gdf.NextDownID[idx]
            fill_value = traverse(polyline_gdf, downstream_comid, attribute, Direction.down)
            if np.isnan(fill_value):
                upstream_comid = polyline_gdf.up1[idx]
                fill_value = traverse(polyline_gdf, upstream_comid, attribute, Direction.up)
                if np.isnan(fill_value):
                     fill_value = np.nanmin(polyline_gdf[attribute])
            nan_fill[i] = fill_value
        _attr_mapped[mask] = nan_fill
        return _attr_mapped
        
        


def soils_data(cfg: DictConfig, edges: zarr.Group) -> None:
    soils_data_path = Path(cfg.data_path) / f"extensions/soils_data/{cfg.zone}"
    if soils_data_path.exists():
        log.info("Soils data already exists in zarr format")
    else:
        root = zarr.group(soils_data_path)
        flowline_file = Path(cfg.data_path) / f"raw/routing_soil_properties/riv_pfaf_{cfg.zone}_buff_split_soil_properties.shp"
        polyline_gdf = gpd.read_file(flowline_file)
        edge_merit_basins: np.ndarray = edges.merit_basin[:]
        gdf_ids = np.array(polyline_gdf.COMID, dtype=int)
        mapping = np.empty_like(edge_merit_basins, dtype=int)
        for i, id in enumerate(tqdm(gdf_ids, desc="\rProcessing soil data")):
            idx = np.where(edge_merit_basins == id)[0]
            mapping[idx] = i
        attributes = [
            'Ks_05_M_25', 
            'N_05_M_250', 
            'SF_05_M_25', 
            'AF_05_M_25',
            'OM_05_M_25', 
            'Cl_05_Mn',
            'Sd_05_Mn', 
            'St_05_Mn', 
        ]
        names = [
            'ksat', 
            'N', 
            'sat-field', 
            'alpha',
            'ormc', 
            'clay_mean_05',
            'sand_mean_05', 
            'silt_mean_05', 
        ]
        for i, attr in enumerate(attributes):
            attr_values = polyline_gdf[attr].values
            _attr_nan_filter = spatial_nan_filter(attr, attr_values, mapping, polyline_gdf)
            root.array(name=names[i], data=_attr_nan_filter[mapping])

            
