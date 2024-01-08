import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import xarray as xr
from xarray.backends import ZarrStore
import zarr

log = logging.getLogger(__name__)


def create_TM(cfg: DictConfig, gdf: gpd.GeoDataFrame) -> ZarrStore:
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
    xr_dataset = xr.Dataset({"TM": (["HUC10", "COMID"], data_array.data)})
    zarr_path = Path(cfg.zarr.TM)
    xr_dataset.to_zarr(zarr_path, mode='w')
    zarr_hierarchy = zarr.open_group(zarr_path, mode='r')
    # df = xr_dataset.to_dataframe().unstack('COMID')['TM']
    # df.to_csv(Path(cfg.csv.TM), compression="gzip")
    # log.info("Finished Data Extraction")
    return zarr_hierarchy


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
    joined_gdf = gpd.sjoin(basins_gdf.set_geometry('centroid'), huc10_gdf, how='left', op='intersects')
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
