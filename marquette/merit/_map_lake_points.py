import logging
from pathlib import Path

import cuspatial
import geopandas as gpd
import pandas as pd
import zarr
from omegaconf import DictConfig
from shapely.geometry import LineString
from shapely.wkt import loads

log = logging.getLogger(name=__name__)


def _map_lake_points(cfg: DictConfig, edges: zarr.Group) -> None:
    """A function that reads in a gdf of hydrolakes information, then intersects that point with edge data
    
    Parameters
    ----------
    cfg: DictConfig
        The configuration object
    edges: zarr.Group
        The zarr group containing the edges
    """
    log.info("Reading in HydroLAKES data")
    data_path = Path(cfg.map_lake_points.hydrolakes)
    if not data_path.exists():
        msg = f"{data_path} does not exist"
        log.exception(msg)
        raise FileNotFoundError(msg)
    gdf = gpd.read_file(data_path)
    reserviors_geoseries = cuspatial.GeoSeries(gdf["geometry"])
    edge_coords = [loads(coords) for coords in edges.coords]
    edge_geoseries = cuspatial.GeoSeries(gpd.GeoSeries(edge_coords))
    log.info("Intersecting HydroLAKES data with edge data")
    for reservoir in reserviors_geoseries
