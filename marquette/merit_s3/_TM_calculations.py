import logging
from pathlib import Path

import binsparse
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from omegaconf import DictConfig
from scipy import sparse
from tqdm import tqdm

log = logging.getLogger(__name__)


def create_HUC_MERIT_TM(
    cfg: DictConfig, edges: xr.Dataset, gdf: gpd.GeoDataFrame
) -> None:
    """
    Create a Transfer Matrix (TM) from GeoDataFrame.

    Args:
        cfg (DictConfig): Hydra configuration object containing settings.
        gdf (GeoDataFrame): GeoDataFrame containing geographical data.
    """
    gdf = gdf.dropna(subset=["HUC10"])
    huc10_ids = gdf["HUC10"].unique()
    huc10_ids.sort()
    merit_ids = np.unique(edges.merit_basin[:])  # already sorted
    data_array = xr.DataArray(
        np.zeros((len(huc10_ids), len(merit_ids))),
        dims=["HUC10", "COMID"],
        coords={"HUC10": huc10_ids, "COMID": merit_ids},
    )
    for idx, huc_id in enumerate(
        tqdm(
            huc10_ids,
            desc="creating TM",
            ncols=140,
            ascii=True,
        )
    ):
        merit_basins = gdf[gdf["HUC10"] == str(huc_id)]
        total_area = merit_basins.iloc[0]["area_new"]

        for j, basin in merit_basins.iterrows():
            unit_area = basin.unitarea / total_area
            data_array.loc[huc_id, basin.COMID] = unit_area
    xr_dataset = xr.Dataset(
        data_vars={"TM": data_array},
        coords={"HUC10": huc10_ids, "COMID": merit_ids},
        attrs={"description": "HUC10 -> MERIT Transition Matrix"},
    )
    print("Saving Zarr Data")
    zarr_path = Path(cfg.create_TMs.HUC.TM)
    xr_dataset.to_zarr(zarr_path, mode="w")


def format_pairs(gage_output: dict):
    pairs = []
    for comid, edge_id in zip(gage_output["comid_idx"], gage_output["edge_id_idx"]):
        for edge in edge_id:
            # Check if upstream is a list (multiple connections)
            if isinstance(edge, list):
                for _id in edge:
                    # Replace None with np.NaN for consistency
                    if _id is None:
                        _id = np.NaN
                    pairs.append((comid, _id))
            else:
                # Handle single connection (not a list)
                if edge is None:
                    edge = np.NaN
                pairs.append((comid, edge))

    return pairs


def create_coo_data(sparse_matrix, root: zarr.Group):
    """
    Creates coordinate format (COO) data from river graph output for a specific gage.

    This function processes the river graph data (specifically the 'ds' and 'up' arrays)
    to create a list of pairs representing connections in the graph. These pairs are then
    stored in a Zarr dataset within a group specific to a gage, identified by 'padded_gage_id'.

    Parameters:
    gage_output: The output from a river graph traversal, containing 'ds' and 'up' keys.
    padded_gage_id (str): The identifier for the gage, used to create a specific group in Zarr.
    root (zarr.Group): The root Zarr group where the dataset will be stored.

    """
    values = sparse_matrix["values"]
    pairs = format_pairs(sparse_matrix)

    # Create a Zarr dataset for this specific gage
    root.create_dataset("pairs", data=np.array(pairs), chunks=(10000,), dtype="int32")
    root.array("values", data=np.array(values), chunks=(10000,), dtype="float32")


def create_sparse_MERIT_FLOW_TM(
    cfg: DictConfig, edges: xr.Dataset
) -> zarr.Group:
    """
    Creating a sparse TM that maps MERIT basins to their reaches. Flow predictions are distributed
    based on reach length/ total merit reach length
    :param cfg:
    :param edges:
    :param huc_to_merit_TM:
    :return:
    """
    log.info("Using Edge COMIDs for TM")
    COMIDs = np.unique(edges.merit_basin[:])  # already sorted
    gage_coo_root = zarr.open_group(Path(cfg.create_TMs.MERIT.TM), mode="a")
    merit_basin = edges.merit_basin[:]
    river_graph_len = edges.len[:]
    river_graph = {"values": [], "comid_idx": [], "edge_id_idx": []}
    for comid_idx, basin_id in enumerate(
        tqdm(
            COMIDs,
            desc="Creating a sparse TM Mapping MERIT basins to their edges",
            ncols=140,
            ascii=True,
        )
    ):
        col_indices = np.where(merit_basin == basin_id)[0]
        total_length = np.sum(river_graph_len[col_indices])
        if total_length == 0:
            print("Basin not found:", basin_id)
            continue
        proportions = river_graph_len[col_indices] / total_length
        river_graph["comid_idx"].append(comid_idx)
        river_graph["edge_id_idx"].append(col_indices.tolist())
        river_graph["values"].extend(proportions.tolist())
    create_coo_data(river_graph, gage_coo_root)


def create_MERIT_FLOW_TM(
    cfg: DictConfig, edges: xr.Dataset
) -> zarr.Group:
    """
    Creating a TM that maps MERIT basins to their reaches. Flow predictions are distributed
    based on reach length/ total merit reach length
    :param cfg:
    :param edges:
    :param huc_to_merit_TM:
    :return:
    """
    # if cfg.create_TMs.MERIT.use_streamflow:
    #     log.info("Using Streamflow COMIDs for TM")
    #     streamflow_predictions_root = zarr.open(
    #         Path(cfg.create_streamflow.predictions), mode="r"
    #     )
    #     comids: np.ndarray = streamflow_predictions_root.COMID[:]
    #     sorted_indices = np.argsort(comids)
    #     COMIDs = comids[sorted_indices].astype(int)
    # else:
    log.info("Using Edge COMIDs for TM")
    COMIDs = np.unique(edges.merit_basin[:])  # already sorted
    river_graph_ids = edges.id[:]
    merit_basin = edges.merit_basin[:]
    river_graph_len = edges.len[:]

    # indices = np.zeros((len(COMIDs), len(river_graph_ids)), dtype=np.float64)
    # for i, basin_id in tqdm(enumerate(COMIDs), total=len(COMIDs), ncols=140, ascii=True, desc="reading idx"):
    #     mask = merit_basin == basin_id
    #     indices[i] = mask.astype(np.float64)

    # # Calculate the number of non-zero elements for each row in indices
    # num_non_zeros = indices.sum(axis=1)

    # proportions = np.transpose(indices) / num_non_zeros

    # tm = np.transpose(proportions)

    data_np = np.zeros((len(COMIDs), len(river_graph_ids)))
    for i, basin_id in enumerate(
        tqdm(
            COMIDs,
            desc="Processing River flowlines",
            ncols=140,
            ascii=True,
        )
    ):
        indices = np.where(merit_basin == basin_id)[0]

        total_length = np.sum(river_graph_len[indices])
        if total_length == 0:
            print("Basin not found:", basin_id)
            continue
        proportions = river_graph_len[indices] / total_length
        for idx, proportion in zip(indices, proportions):
            column_index = np.where(river_graph_ids == river_graph_ids[idx])[0][0]
            data_np[i][column_index] = proportion

    if cfg.create_TMs.MERIT.save_sparse:
        log.info("Writing to sparse matrix")
        gage_coo_root = zarr.open_group(Path(cfg.create_TMs.MERIT.TM), mode="a")
        matrix = sparse.csr_matrix(data_np)
        binsparse.write(gage_coo_root, "TM", matrix)
        log.info("Sparse matrix written")
    else:
        data_array = xr.DataArray(
            data=data_np,
            dims=["COMID", "EDGEID"],  # Explicitly naming the dimensions
            coords={"COMID": COMIDs, "EDGEID": river_graph_ids},  # Adding coordinates
        )
        xr_dataset = xr.Dataset(
            data_vars={"TM": data_array},
            attrs={"description": "MERIT -> Edge Transition Matrix"},
        )
        log.info("Writing MERIT TM to zarr store")
        zarr_path = Path(cfg.create_TMs.MERIT.TM)
        xr_dataset.to_zarr(zarr_path, mode="w")
        # zarr_hierarchy = zarr.open_group(Path(cfg.create_TMs.MERIT.TM), mode="r")


def join_geospatial_data(cfg: DictConfig) -> gpd.GeoDataFrame:
    """
    Joins two geospatial datasets based on the intersection of centroids of one dataset with the geometries of the other.

    Args:
    huc10_path (str): File path to the HUC10 shapefile.
    basins_path (str): File path to the basins shapefile.

    Returns:
    gpd.GeoDataFrame: The resulting joined GeoDataFrame.
    """
    huc10_gdf = gpd.read_file(Path(cfg.create_TMs.HUC.shp_files)).to_crs(epsg=4326)
    basins_gdf = gpd.read_file(Path(cfg.create_TMs.MERIT.shp_files))
    basins_gdf["centroid"] = basins_gdf.geometry.centroid
    joined_gdf = gpd.sjoin(
        basins_gdf.set_geometry("centroid"),
        huc10_gdf,
        how="left",
        predicate="intersects",
    )
    joined_gdf.set_geometry("geometry", inplace=True)
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
    plt.xlabel(r"Ratio of  $\sum$ MERIT basin area to HUC10 basin areas")
    plt.ylabel("Number of HUC10s")
    plt.title(r"Distribution of $\sum$ MERIT area / HUC10 basin area")
    min_val = series.min()
    median_val = series.median()
    mean_val = series.mean()
    max_val = series.max()
    plt.axvline(
        min_val,
        color="grey",
        linestyle="dashed",
        linewidth=2,
        label=f"Min: {min_val:.3f}",
    )
    plt.axvline(
        median_val,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median_val:.3f}",
    )
    plt.axvline(
        mean_val,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_val:.3f}",
    )
    plt.axvline(
        max_val,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Max: {max_val:.3f}",
    )
    plt.legend()
    plt.show()
