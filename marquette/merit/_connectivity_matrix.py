import ast
from collections import defaultdict
import logging
from pathlib import Path
from typing import List, Tuple, Any


import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import torch
import xarray as xr
import zarr


log = logging.getLogger(__name__)


def left_pad_number(number):
    """
    Left pads a number with '0' if it has 7 digits to make it 8 digits long.

    Parameters:
    number (int or str): The number to be left-padded.

    Returns:
    str: The left-padded number as a string.
    """
    number_str = str(number)
    if len(number_str) == 7:
        return "0" + number_str
    return number_str


def map_gages_to_zone(cfg: DictConfig, edges: zarr.Group) -> gpd.GeoDataFrame:
    def choose_row_to_keep(group_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Selects the row where 'uparea' is closest to 'DRAIN_SQKM' without going below it.
        If no such row exists, selects the row with the closest 'uparea' under 'DRAIN_SQKM'.

        Parameters:
        group_df (DataFrame): DataFrame representing all rows of a particular group.

        Returns:
        DataFrame: A single row which has the closest 'uparea' to 'DRAIN_SQKM'.
        """
        group_df["diff"] = group_df["uparea"] - group_df["DRAIN_SQKM"]
        valid_rows = group_df[group_df["diff"] >= 0]

        if not valid_rows.empty:
            idx = valid_rows["diff"].idxmin()
        else:
            idx = group_df["diff"].abs().idxmin()

        return group_df.loc[[idx]].drop(columns=["diff"])

    def filter_by_comid_prefix(gdf: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
        """
        Filters a GeoDataFrame to include only rows where the first two characters
        of the 'COMID' column match the given prefix.

        Parameters:
        gdf (GeoDataFrame): The GeoDataFrame to filter.
        prefix (str): The two-character prefix to match.

        Returns:
        GeoDataFrame: Filtered GeoDataFrame.
        """
        gdf["MERIT_ZONE"] = gdf["COMID"].astype(str)
        filtered_gdf = gdf[gdf["MERIT_ZONE"].str[:2] == prefix]
        return filtered_gdf

    def find_closest_edge(
        row: gpd.GeoSeries,
        zone_edge_ids: np.ndarray,
        zone_merit_basin_ids: np.ndarray,
        zone_upstream_areas: np.ndarray,
    ) -> Tuple[np.float64, int, np.float64, np.float64, np.float64]:
        """
        Finds details of the edge with the upstream area closest to the DRAIN_SQKM value in absolute terms
        and calculates the percent error between DRAIN_SQKM and the matched zone_edge_uparea.

        Parameters:
        row (GeoSeries): A row from the result_df GeoDataFrame.
        zone_edge_ids (ndarray): Array of edge IDs.
        zone_merit_basin_ids (ndarray): Array of merit basin IDs corresponding to edge IDs.
        zone_upstream_areas (ndarray): Array of upstream areas corresponding to edge IDs.

        Returns:
        Tuple[np.float64, int, np.float64, np.float64, np.float64]: Contains edge ID, index of the matching edge, upstream area of the matching edge,
               the difference in catchment area, and the percent error between DRAIN_SQKM and zone_edge_uparea.
        """
        COMID = row["COMID"]
        DRAIN_SQKM = row["DRAIN_SQKM"]
        matching_indices = np.where(zone_merit_basin_ids == COMID)[0]

        if len(matching_indices) == 0:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # Return NaNs if no matching edges are found

        matching_upstream_areas = zone_upstream_areas[matching_indices]
        differences = np.abs(matching_upstream_areas - DRAIN_SQKM)
        min_diff_idx = np.argmin(differences)
        percent_error = (
            (differences[min_diff_idx] / DRAIN_SQKM) if DRAIN_SQKM != 0 else np.nan
        )

        return (
            zone_edge_ids[matching_indices[min_diff_idx]],
            matching_indices[min_diff_idx],
            matching_upstream_areas[min_diff_idx],
            differences[min_diff_idx],
            percent_error,
        )

    gdf = gpd.read_file(Path(cfg.save_paths.usgs_flowline_intersections))
    if cfg.filter:
        # filter based on large-list of gage_locations
        gage_locations_df = pd.read_csv(cfg.save_paths.gage_locations)
        gage_ids = gage_locations_df["id"].astype(str).apply(lambda x: x.zfill(8))
        gdf = gdf[gdf["STAID"].isin(gage_ids)]
    gdf["COMID"] = gdf["COMID"].astype(int)
    filtered_gdf = filter_by_comid_prefix(gdf, cfg.zone)
    grouped = filtered_gdf.groupby("STAID")
    unique_gdf = grouped.apply(choose_row_to_keep).reset_index(drop=True)
    zone_edge_ids = edges.id[:]
    zone_merit_basin_ids = edges.merit_basin[:]
    zone_upstream_areas = edges.uparea[:]
    edge_info = unique_gdf.apply(
        lambda row: find_closest_edge(
            row, zone_edge_ids, zone_merit_basin_ids, zone_upstream_areas
        ),
        axis=1,
        result_type="expand",
    )
    unique_gdf["edge_intersection"] = edge_info[0]
    unique_gdf["zone_edge_id"] = edge_info[1]
    unique_gdf["zone_edge_uparea"] = edge_info[2]
    unique_gdf["zone_edge_vs_gage_area_difference"] = edge_info[3]
    unique_gdf["drainage_area_percent_error"] = edge_info[4]

    result_df = unique_gdf[
        unique_gdf["drainage_area_percent_error"] <= cfg.drainage_area_treshold
    ]

    columns = [
        "STAID",
        "STANAME",
        "MERIT_ZONE",
        "HUC02",
        "DRAIN_SQKM",
        "LAT_GAGE",
        "LNG_GAGE",
        "STATE",
        "COMID",
        "edge_intersection",
        "zone_edge_id",
        "zone_edge_uparea",
        "zone_edge_vs_gage_area_difference",
        "drainage_area_percent_error",
    ]
    result = result_df[columns]
    result = result.dropna()
    result["STAID"] = result["STAID"].astype(int)
    result.to_csv(Path(cfg.csv.zone_gage_information), index=False)

    # combining this zone df to the master df that has all zonal information
    try:
        df = pd.read_csv(Path(cfg.csv.gage_information))

        try:
            combined_df = pd.concat([df, result], ignore_index=True)
            sorted_df = combined_df.sort_values(by=["STAID"])
            sorted_df.to_csv(Path(cfg.csv.gage_information), index=False)
        except pd.errors.InvalidIndexError:
            log.info(
                "Not merging your file with the master list as it seems like your gages are already included"
            )
    except FileNotFoundError:
        result.to_csv(Path(cfg.csv.gage_information), index=False)
    return result


def create_gage_connectivity(
    edges: zarr.hierarchy.Group,
    gage_coo_root: zarr.hierarchy.Group,
    zone_csv: gpd.GeoDataFrame,
) -> None:
    def stack_traversal(gage_id: str, id_: str, idx: int, merit_flowlines: zarr.Group):
        """
        Performs a stack-based traversal on a graph of river flowlines.

        This function uses a depth-first search approach to traverse through the graph
        represented by river flowlines. Starting from a given node (identified by 'id_' and 'idx'),
        it explores the upstream flowlines, constructing a graph structure that includes
        information about each node and its upstream connections.

        Parameters:
        id_ (str): The identifier of the starting node in the river graph.
        idx (int): The index of the starting node in the 'merit_flowlines' dataset.
        merit_flowlines (zarr.Group): A Zarr group containing river flowline data.
                                      Expected to have 'up' and 'id' arrays for upstream connections
                                      and node identifiers, respectively.

        Returns:
        dict: A dictionary representing the traversed river graph. It contains three keys:
              'ID' - a list of node identifiers,
              'ds' - a list of indices corresponding to the downstream node for each ID,
              'up' - a list of lists, where each sublist contains indices of upstream nodes.

        Note:
        The function prints the stack size after every 1000 iterations for tracking the traversal progress.
        """
        river_graph = {"ID": [], "ds": [], "up": []}
        stack = [(id_, idx)]
        visited = set()
        iteration_count = 1

        while stack:
            if iteration_count % 1000 == 0:
                print(
                    f"Gage{gage_id}, ID, IDX {id_, idx}: Stack size is {len(stack)}, river graph len: {len(river_graph['ID'])}"
                )

            current_id, current_idx = stack.pop()
            if current_id in visited:
                iteration_count += 1
                continue
            visited.add(current_id)

            up_ids = ast.literal_eval(merit_flowlines.up[current_idx])
            up_idx_list = []

            for up_id in up_ids:
                up_idx = np.where(merit_flowlines.id[:] == up_id)[0][0]
                if up_id not in visited:
                    stack.append((up_id, up_idx))
                up_idx_list.append(up_idx)

            river_graph["ID"].append(current_id)
            river_graph["ds"].append(current_idx)
            river_graph["up"].append(up_idx_list if up_idx_list else [None])

            iteration_count += 1

        return river_graph

    def create_coo_data(
        gage_output, padded_gage_id: str, root: zarr.Group
    ) -> List[Tuple[Any, Any]]:
        """
        Creates coordinate format (COO) data from river graph output for a specific gage.

        This function processes the river graph data (specifically the 'ds' and 'up' arrays)
        to create a list of pairs representing connections in the graph. These pairs are then
        stored in a Zarr dataset within a group specific to a gage, identified by 'padded_gage_id'.

        Parameters:
        gage_output: The output from a river graph traversal, containing 'ds' and 'up' keys.
        padded_gage_id (str): The identifier for the gage, used to create a specific group in Zarr.
        root (zarr.Group): The root Zarr group where the dataset will be stored.

        Returns:
        List[Tuple[Any, Any]]: A list of tuples, each representing a pair of connected nodes in the graph.
        """
        pairs = []

        # Iterate over downstream and upstream nodes to create pairs
        for ds, ups in zip(gage_output["ds"], gage_output["up"]):
            for up in ups:
                # Check if upstream is a list (multiple connections)
                if isinstance(up, list):
                    for _id in up:
                        # Replace None with np.NaN for consistency
                        if _id is None:
                            _id = np.NaN
                        pairs.append((ds, _id))
                else:
                    # Handle single connection (not a list)
                    if up is None:
                        up = np.NaN
                    pairs.append((ds, up))

        # Create a Zarr dataset for this specific gage
        single_gage_csr_data = root.require_group(padded_gage_id)
        single_gage_csr_data.create_dataset(
            "pairs", data=np.array(pairs), chunks=(10000,), dtype="float32"
        )

        return pairs

    def find_connections(row, coo_root, zone_attributes):
        gage_id = str(row["STAID"]).zfill(8)
        edge_id = row["edge_intersection"]
        zone_edge_id = row["zone_edge_id"]
        if gage_id not in coo_root:
            gage_output = stack_traversal(
                gage_id, edge_id, zone_edge_id, zone_attributes
            )
            create_coo_data(gage_output, gage_id, coo_root)

    def apply_find_connections(row, gage_coo_root, edges):
        return find_connections(row, gage_coo_root, edges)

    dask_df = dd.from_pandas(zone_csv, npartitions=10)
    result = dask_df.apply(
        apply_find_connections,
        args=(gage_coo_root, edges),
        axis=1,
        meta=(None, "object"),
    )
    with ProgressBar():
        _ = result.compute()
