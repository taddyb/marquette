import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(name=__name__)

def _map_lake_points(cfg: DictConfig, edges: zarr.Group) -> None:
    """A function that reads in a gdf of hydrolakes information, finds its corresponding edge, then saves the data
    
    Parameters
    ----------
    cfg: DictConfig
        The configuration object
    edges: zarr.Group
        The zarr group containing the edges
    """
    data_path = Path(cfg.map_lake_points.lake_points)
    if not data_path.exists():
        msg = "Cannot find the lake points file"
        log.exception(msg)
        raise FileNotFoundError(msg)
    gdf = gpd.read_file(data_path)
    lake_comids = gdf["COMID"].astype(int).values
    edges_comids : np.ndarray = edges["merit_basin"][:].astype(np.int32) # type: ignore
    
    hylak_id = np.full(len(edges_comids), -1, dtype=np.int32)
    grand_id = np.full_like(edges_comids, -1, dtype=np.int32)
    lake_area = np.full_like(edges_comids, -1, dtype=np.float32)
    vol_total = np.full_like(edges_comids, -1, dtype=np.float32)
    depth_avg = np.full_like(edges_comids, -1, dtype=np.float32)
    vol_res = np.full(len(edges_comids), -1, dtype=np.int32)
    elevation = np.full_like(edges_comids, -1, dtype=np.int32)
    shore_dev = np.full_like(edges_comids, -1, dtype=np.float32)
    dis_avg = np.full_like(edges_comids, -1, dtype=np.float32)
    
    for idx, lake_id in enumerate(tqdm(
        lake_comids,
        desc="Mapping Lake COMIDS to edges",
        ncols=140,
        ascii=True,
    )) :
        jdx = np.where(edges_comids == lake_id)[0]
        if not jdx.size:
            log.info(f"No lake found for COMID {lake_id}")
        else:
            # Assumung the pour point is at the end of the COMID
            edge_idx = jdx[-1]
            lake_row = gdf.iloc[idx]
            hylak_id[edge_idx] = lake_row["Hylak_id"]
            grand_id[edge_idx] = lake_row["Grand_id"]
            lake_area[edge_idx] = lake_row["Lake_area"]
            vol_total[edge_idx] = lake_row["Vol_total"]
            depth_avg[edge_idx] = lake_row["Depth_avg"]
            
            vol_res[edge_idx] = lake_row["Vol_res"]
            elevation[edge_idx] = lake_row["Elevation"]
            shore_dev[edge_idx] = lake_row["Shore_dev"]
            dis_avg[edge_idx] = lake_row["Dis_avg"]
    
    edges.array("hylak_id", data=hylak_id)
    edges.array("grand_id", data=grand_id)
    edges.array("lake_area", data=lake_area)
    edges.array("vol_total", data=vol_total)
    edges.array("depth_avg", data=depth_avg)
    edges.array("vol_res", data=vol_res)
    edges.array("elevation", data=elevation)
    edges.array("shore_dev", data=shore_dev)
    edges.array("dis_avg", data=dis_avg)
    
    log.info("Wrote Lake data for zones to zarr")
