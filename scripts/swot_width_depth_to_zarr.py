import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Point
from tqdm import tqdm
import xarray as xr
import zarr

def parse_arguments():
    """Parsing the arguments for zone"""
    parser = argparse.ArgumentParser(description='Process HydroSWOT river width and depth obs for a specified zone')
    parser.add_argument('--zone', type=str, required=True, help='Zone identifier for MERIT data processing')
    return parser.parse_args()

def merge(zone: str):
    """Merging and writing the hydroswot streamflow points to the edges zarr store
    
    Parameters
    ----------
    zone: str
        the merit zone we're pulling obs from
    
    Note
    ----
    You will need to install the following pacakge for this script to work:
    `uv pip install openpyxl`
    """
    print("reading zarr store and excel file")
    root_path = Path(f"/projects/mhpi/data/MERIT/zarr/graph/CONUS/edges/{zone}")
    if root_path.exists() is False:
        raise FileNotFoundError("Cannot find your Zarr store")
    root = zarr.open_group(root_path)
    
    file_path = Path("/projects/mhpi/data/swot/SWOT_ADCP_Dataset.xlsx")
    if file_path.exists() == False:
        raise FileNotFoundError("Cannot find SWOT data")
    df = pd.read_excel(file_path)
    geometry = [Point(xy) for xy in zip(df["dec_long_va"], df["dec_lat_va"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print("reading river gdf and buffering lines")
    # basin_shp_file = Path(f"/projects/mhpi/data/MERIT/raw/basins/cat_pfaf_{zone}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
    riv_shp_file = Path(f"/projects/mhpi/data/MERIT/raw/flowlines/riv_pfaf_{zone}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
    # if basin_shp_file.exists() == False:
    #     raise FileNotFoundError("Cannot find MERIT basin COMID data")
    if riv_shp_file.exists() == False:
        raise FileNotFoundError("Cannot find MERIT flowlines COMID data")
    # basin_gdf = gpd.read_file(filename=basin_shp_file)    
    riv_gdf = gpd.read_file(filename=riv_shp_file).to_crs("EPSG:5070")  
    
    riv_gdf.geometry = riv_gdf.buffer(200)
    riv_gdf = riv_gdf.to_crs("EPSG:4326")
    
    print("Running spatial join")
    matched_gdf = gpd.sjoin(left_df=gdf, right_df=riv_gdf, how='inner', predicate='intersects')
    geometry = [Point(xy) for xy in zip(matched_gdf["dec_long_va"], matched_gdf["dec_lat_va"])]
    point_gdf = gpd.GeoDataFrame(matched_gdf, geometry=geometry, crs="EPSG:4326")
        
    if "mean_observed_swot_width" not in root:
        json_ = {
            "merit_COMID": point_gdf["COMID"].values,
            "drainage_area": point_gdf["drain_area_va"].values * 2.58999,  # converting from mi^2 to km^2
            "width": point_gdf["stream_wdth_va"].values * 0.3048,  # converting from feet to meters
        }
        df = pl.DataFrame(
            data=json_,
        ).filter(~pl.all_horizontal(pl.col("width").is_nan())).filter(~pl.all_horizontal(pl.col("drainage_area").is_nan()))
        
        drainage_area_avg = df.group_by("merit_COMID").agg(pl.col("drainage_area").mean()).sort("merit_COMID")
        width_avg = df.group_by("merit_COMID").agg(pl.col("width").mean()).sort("merit_COMID")
        comids = width_avg.select("merit_COMID").to_numpy().squeeze()
        basins = root.merit_basin[:]
        width_np = np.full_like(root.id[:], -1)  
        for comid in tqdm(comids, desc="writing comids"):
            idx = np.argwhere(basins == comid).squeeze()
            uparea = root.uparea[idx][-1]
            measured_width = width_avg.filter(pl.col("merit_COMID") == comid).select("width").to_numpy()[0][0]
            measured_da = drainage_area_avg.filter(pl.col("merit_COMID") == comid).select("drainage_area").to_numpy()[0][0]
            if uparea < 1000:
                threshold = 0.25
            else:
                threshold = 0.1
            if np.abs((measured_da - uparea) / uparea) < threshold:
                width_np[idx] = measured_width
    
        root.create_array(name="mean_observed_swot_width", data=width_np.astype(np.float32))
    else:
        print("width write has already been made")
        
    if "mean_observed_swot_depth" not in root:
        json_ = {
            "merit_COMID": point_gdf["COMID"].values,
            "drainage_area": point_gdf["drain_area_va"].values * 2.58999,  # converting from mi^2 to km^2
            "mean_depth": point_gdf["mean_depth_va"].values * 0.3048,  # converting from feet to meters
        }
        df = pl.DataFrame(
            data=json_,
        ).filter(~pl.all_horizontal(pl.col("mean_depth").is_nan())).filter(~pl.all_horizontal(pl.col("drainage_area").is_nan()))
        
        drainage_area_avg = df.group_by("merit_COMID").agg(pl.col("drainage_area").mean()).sort("merit_COMID")
        depth_avg = df.group_by("merit_COMID").agg(pl.col("mean_depth").mean()).sort("merit_COMID")
        
        comids = depth_avg.select("merit_COMID").to_numpy().squeeze()
        basins = root.merit_basin[:]
        depth_np = np.full_like(root.id[:], -1)  
        for comid in tqdm(comids, desc="writing comids"):
            idx = np.argwhere(basins == comid).squeeze()
            uparea = root.uparea[idx][-1]
            measured_depth = depth_avg.filter(pl.col("merit_COMID") == comid).select("mean_depth").to_numpy()[0][0]
            measured_da = drainage_area_avg.filter(pl.col("merit_COMID") == comid).select("drainage_area").to_numpy()[0][0]
            if uparea < 1000:
                threshold = 0.25
            else:
                threshold = 0.1
            if np.abs((measured_da - uparea) / uparea) < threshold:
                depth_np[idx] = measured_depth
    
        root.array(name="mean_observed_swot_depth", data=depth_np.astype(np.float32))
    else:
        print("depth write has already been made")
        
    print("Finished")

    
if __name__ == "__main__":
    args = parse_arguments()
    merge(args.zone)
    # zone = "74"
    # merge(zone)
