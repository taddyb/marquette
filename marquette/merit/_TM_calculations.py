import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import xarray as xr
import zarr

log = logging.getLogger(__name__)


def create_HUC_MERIT_TM(cfg: DictConfig, gdf: gpd.GeoDataFrame) -> zarr.hierarchy.Group:
    """
    Create a Transfer Matrix (TM) from GeoDataFrame.

    Args:
        cfg (DictConfig): Hydra configuration object containing settings.
        gdf (GeoDataFrame): GeoDataFrame containing geographical data.
    """
    gdf = gdf.dropna(subset=['HUC10'])
    huc10_ids = gdf["HUC10"].unique()
    merit_ids = gdf["COMID"].unique()
    huc10_ids.sort()
    merit_ids.sort()
    data_array = xr.DataArray(np.zeros((len(huc10_ids), len(merit_ids))),
                              dims=["HUC10", "COMID"],
                              coords={"HUC10": huc10_ids, "COMID": merit_ids})
    for idx, huc_id in enumerate(tqdm(huc10_ids, desc="creating TM")):
        merit_basins = gdf[gdf['HUC10'] == str(huc_id)]
        total_area = merit_basins.iloc[0]["area_new"]

        for j, basin in merit_basins.iterrows():
            unit_area = basin.unitarea / total_area
            data_array.loc[huc_id, basin.COMID] = unit_area
    xr_dataset = xr.Dataset(
        data_vars={"TM": data_array},
        coords={"HUC10": huc10_ids, "COMID": merit_ids},
        attrs={"description": "HUC10 -> MERIT Transition Matrix"}
    )
    print("Saving Zarr Data")
    zarr_path = Path(cfg.zarr.HUC_TM)
    xr_dataset.to_zarr(zarr_path, mode='w')
    zarr_hierarchy = zarr.open_group(Path(cfg.zarr.HUC_TM), mode='r')
    return zarr_hierarchy


def create_MERIT_FLOW_TM(
    cfg: DictConfig, edges: zarr.hierarchy.Group, huc_to_merit_TM: zarr.hierarchy.Group
) -> zarr.hierarchy.Group:
    """
    Creating a TM that maps MERIT basins to their reaches. Flow predictions are distributed
    based on reach length/ total merit reach length
    :param cfg:
    :param edges:
    :param huc_to_merit_TM:
    :return:
    """
    COMIDs = huc_to_merit_TM.COMID[:]
    river_graph_ids = edges.id[:]
    merit_basin = edges.merit_basin[:]
    river_graph_len = edges.len[:]
    proportion_array = np.zeros((len(COMIDs), len(river_graph_ids)))
    for i, basin_id in enumerate(tqdm(COMIDs, desc="Processing River flowlines")):
        indices = np.where(merit_basin == basin_id)[0]

        total_length = np.sum(river_graph_len[indices])
        if total_length == 0:
            print("Basin not found:", basin_id)
            continue
        proportions = river_graph_len[indices] / total_length
        for idx, proportion in zip(indices, proportions):
            column_index = np.where(river_graph_ids == river_graph_ids[idx])[0][0]
            proportion_array[i, column_index] = proportion

    data_array = xr.DataArray(
        data=proportion_array,
        dims=["COMID", "EDGEID"],  # Explicitly naming the dimensions
        coords={"COMID": COMIDs, "EDGEID": river_graph_ids}  # Adding coordinates
    )
    xr_dataset = xr.Dataset(
        data_vars={"TM": data_array},
        attrs={"description": "MERIT -> Edge Transition Matrix"}
    )
    zarr_path = Path(cfg.zarr.MERIT_TM)
    xr_dataset.to_zarr(zarr_path, mode='w')
    zarr_hierarchy = zarr.open_group(Path(cfg.zarr.MERIT_TM), mode='r')
    return zarr_hierarchy
    # zarr_group = zarr.open_group(Path(cfg.zarr.MERIT_TM), mode='w')
    # zarr_group.create_dataset('TM', data=proportion_array)
    # zarr_group.create_dataset('COMIDs', data=COMIDs)
    # zarr_group.create_dataset('EDGEIDs', data=river_graph_ids)


def join_geospatial_data(cfg: DictConfig) -> gpd.GeoDataFrame:
    """
    Joins two geospatial datasets based on the intersection of centroids of one dataset with the geometries of the other.

    Args:
    huc10_path (str): File path to the HUC10 shapefile.
    basins_path (str): File path to the basins shapefile.

    Returns:
    gpd.GeoDataFrame: The resulting joined GeoDataFrame.
    """
    huc10_gdf = gpd.read_file(Path(cfg.save_paths.huc10)).to_crs(epsg=4326)
    basins_gdf = gpd.read_file(Path(cfg.save_paths.basins))
    basins_gdf['centroid'] = basins_gdf.geometry.centroid
    joined_gdf = gpd.sjoin(basins_gdf.set_geometry('centroid'), huc10_gdf, how='left', predicate='intersects')
    joined_gdf.set_geometry('geometry', inplace=True)
    return joined_gdf


def plot_histogram(df: pd.DataFrame, num_bins: int = 100) -> None:
    """
    Creates and displays a histogram for the sum of values in each row of the provided DataFrame.

    Args:
    df (pd.DataFrame): A Pandas DataFrame whose row sums will be used for the histogram.
    num_bins (int, optional): The number of bins for the histogram. Defaults to 100.

    The function calculates the minimum, median, mean, and maximum values of the row sums
    and displays these as vertical lines on the histogram.
    """
    series = df.sum(axis=1)
    plt.figure(figsize=(10, 6))
    series.hist(bins=num_bins)
    plt.xlabel(r'Ratio of  $\sum$ MERIT basin area to HUC10 basin areas')
    plt.ylabel('Number of HUC10s')
    plt.title(r'Distribution of $\sum$ MERIT area / HUC10 basin area')
    min_val = series.min()
    median_val = series.median()
    mean_val = series.mean()
    max_val = series.max()
    plt.axvline(min_val, color='grey', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.3f}')
    plt.axvline(median_val, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.3f}')
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(max_val, color='green', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.3f}')
    plt.legend()
    plt.show()

