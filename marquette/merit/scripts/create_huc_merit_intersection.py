import logging
import time
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pyproj import CRS
from tqdm import tqdm

log = logging.getLogger(__name__)


fiona_logger = logging.getLogger("fiona._env")
fiona_logger.setLevel(logging.CRITICAL)


rasterio_logger = logging.getLogger("rasterio._env")
rasterio_logger.setLevel(logging.ERROR)


def _traverse_flow_tree(comid, geodf, visited_comids):
    # Check if comid is 0 or already visited
    if comid == 0 or comid in visited_comids:
        return

    # Add COMID to visited list
    visited_comids.add(int(comid))

    # Extract the row corresponding to COMID
    row = geodf[geodf["COMID"] == comid]

    # Recurse for 'up1', 'up2', 'up3', and 'up4'
    for col in ["up1", "up2", "up3", "up4"]:
        next_comid = row.iloc[0][col]
        _traverse_flow_tree(next_comid, geodf, visited_comids)


def create_TM(cfg: DictConfig, gdf: gpd.GeoDataFrame) -> None:
    df_cols = []
    problem_hucs = []
    huc10_ids = gdf["huc10"].unique()
    merit_ids = gdf["COMID"].unique()
    huc10_ids.sort()
    merit_ids.sort()
    df = pd.DataFrame(index=huc10_ids, columns=merit_ids)
    df["HUC10"] = huc10_ids
    df = df.set_index("HUC10")
    for idx, id in enumerate(
        tqdm(
            huc10_ids,
            desc="creating TM",
            ncols=140,
            ascii=True,
        )
    ):
        merit_basins = gdf[gdf["huc10"] == id]
        total_area = sum([merit_basins.iloc[i].unitarea for i in range(merit_basins.shape[0])])
        for j, basin in merit_basins.iterrows():
            df_cols.append(basin.COMID)
            data = np.zeros([huc10_ids.shape[0]])
            data[idx] = basin.unitarea / total_area
            df[basin.COMID] = data
        percent_error = (total_area - float(merit_basins.areasqkm.iloc[0])) / total_area
        if abs(percent_error) > 0.05:
            log.info(
                f"HUC10: {merit_basins.huc10.iloc[0]} has bad clipping. Merit basins percent error = {percent_error}"
            )
            problem_hucs.append(merit_basins.huc10.iloc[0])
    log.info(f"Writing CSV to {cfg.output_TM}")
    df.to_csv(Path(cfg.output_TM), compression="gzip")
    log.info("Finished Data Extraction")


@hydra.main(version_base=None, config_path="../../conf/scripts/", config_name="create_srb")
def create_intersections(cfg: DictConfig) -> None:
    # Define the Albers Equal Area Conic projection parameters for CONUS
    albers_conus = CRS.from_proj4(
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
    )
    flow_file_path = Path(cfg.flowlines)
    flow_geodf = gpd.read_file(flow_file_path)
    visited_comids = set()
    start_comid = cfg.starting_edge
    _traverse_flow_tree(start_comid, flow_geodf, visited_comids)
    flowlines = flow_geodf[flow_geodf["COMID"].astype(int).isin(visited_comids)]
    flowline_path = Path(cfg.flowlines_output)
    flowline_path.mkdir(parents=True, exist_ok=True)
    flowlines.to_file(flowline_path)
    log.info("created list of merit IDs")
    directory_path = Path(cfg.merit_basin_dir)
    matching_rows = []
    for shp_path in tqdm(
        directory_path.glob(cfg.merit_basin_file),
        desc="Reading merit basins",
        ncols=140,
        ascii=True,
    ):
        current_gdf = gpd.read_file(Path(shp_path))
        rows = current_gdf[current_gdf["COMID"].astype(int).isin(visited_comids)]
        if len(rows) > 0:
            matching_rows.append(rows)
    merit_basins_subset = pd.concat(matching_rows)
    merit_basins_subset["COMID"] = merit_basins_subset["COMID"].astype(int)
    merit_basins_subset = merit_basins_subset.to_crs(albers_conus)  # to projected crs
    merit_basins_subset["geometry"] = merit_basins_subset.geometry.centroid
    huc10s = gpd.read_file(Path(cfg.huc_10s))
    huc10s = huc10s.to_crs(albers_conus)
    log.info("read huc10 basins")
    intersection_gdf = gpd.overlay(merit_basins_subset, huc10s, how="intersection")
    intersection_gdf = intersection_gdf.drop_duplicates(subset=["COMID"])
    log.info("intersected data, writing to disk")
    output_path = Path(cfg.output_file)
    output_path.mkdir(parents=True, exist_ok=True)
    intersection_gdf.to_file(output_path)
    log.info("Finished file creation")
    create_TM(cfg, intersection_gdf)

    # srb_basins = pd.concat(matching_rows)
    # srb_basins['COMID'] = srb_basins['COMID'].astype(int)
    # srb_basins.to_file(Path"H:/Tadd/02_formatted_data/merit_subsets"))


if __name__ == "__main__":
    start = time.perf_counter()
    create_intersections()
    end = time.perf_counter()
    log.info(f"Extracting data took : {(end - start):.6f} seconds")
