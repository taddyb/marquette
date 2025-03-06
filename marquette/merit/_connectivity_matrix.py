import ast
import logging
from pathlib import Path
from typing import Any, List, Tuple

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def format_pairs(gage_output: dict):
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

    return pairs


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
        gdf["MERIT_ZONE"] = gdf["COMID"].astype(str).str[:2]
        filtered_gdf = gdf[gdf["MERIT_ZONE"] == str(prefix)]
        return filtered_gdf
    
    def find_upstream_das(zone_gdf, edges) -> tuple[np.ndarray, np.ndarray]:
        """Finds the upstream drainage area at the flow-entry to each COMID"""
        upstream_da = []
        downstream_comid_da = []
        for _, row in zone_gdf.iterrows():
            upstream_comids = [
                int(row["up1"]),
                int(row["up2"]),
                int(row["up3"]),
                int(row["up4"]),
            ]
            down_id = int(row["NextDownID"])
            upareas = []
            for comid in upstream_comids:
                if comid != 0:
                    idx = np.where(edges.merit_basin[:] == comid)[0]
                    upareas.append(np.min(edges.uparea[idx]))
            idx = np.where(edges.merit_basin[:] == down_id)[0]
            down_uparea = edges.uparea[idx]
            if down_uparea.shape[0] == 0:
                downstream_comid_da.append(-1)
            else:
                downstream_comid_da.append(np.min(down_uparea))
            upstream_da.append(np.max(np.array(upareas)) if len(upareas) > 0 else 0)
        
        return np.array(upstream_da), np.array(downstream_comid_da)
    
    def filter_by_das(zone_gdf, upstream_comid_da, downstream_comid_da):
        """Filters out drainage areas based on if the gauge is in the correct COMID
        """
        zone_gdf = zone_gdf.copy()  # Creating a new reference
        pour_point_da = zone_gdf["uparea"].values
        zone_gdf["upstream_comid_da"] = upstream_comid_da
        ds_mask = downstream_comid_da == -1
        downstream_comid_da[ds_mask] = pour_point_da[ds_mask]  # setting the downstream comid da to the pour point if no DS values
        zone_gdf["downstream_comid_da"] = downstream_comid_da
        zone_gdf["up_epsilon"] = 100
        zone_gdf["ds_epsilon"] = 100 
        mask_lb = zone_gdf["DRAIN_SQKM"] >= upstream_comid_da - zone_gdf["up_epsilon"] 
        mask_ub = zone_gdf["DRAIN_SQKM"] <= downstream_comid_da + zone_gdf["ds_epsilon"]
        mask = np.logical_and(mask_lb, mask_ub)
        return zone_gdf[mask]
                    

    def find_closest_edge(
        row: gpd.GeoSeries,
        zone_edge_ids: np.ndarray,
        zone_merit_basin_ids: np.ndarray,
        zone_upstream_areas: np.ndarray,
    ):
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
                np.nan,
            )  # Return NaNs if no matching edges are found

        matching_upstream_areas = zone_upstream_areas[matching_indices]
        differences = np.abs(matching_upstream_areas - DRAIN_SQKM)
        min_diff_idx = np.argmin(differences)
        percent_error = (
            (differences[min_diff_idx] / DRAIN_SQKM) if DRAIN_SQKM != 0 else np.nan
        )
        ratio = np.abs(matching_upstream_areas / DRAIN_SQKM)

        return (
            zone_edge_ids[matching_indices[min_diff_idx]],
            matching_indices[min_diff_idx],
            matching_upstream_areas[min_diff_idx],
            differences[min_diff_idx],
            percent_error,
            ratio,
        )

    gdf = gpd.read_file(Path(cfg.create_N.gage_buffered_flowline_intersections))
    if cfg.create_N.filter_based_on_dataset:
        # filter based on large-list of gage_locations
        gage_locations_df = pd.read_csv(cfg.create_N.obs_dataset)
        try:
            if cfg.create_N.pad_gage_id:
                gage_ids = (
                    gage_locations_df["id"].astype(str).apply(lambda x: x.zfill(8))
                )
            else:
                gage_ids = gage_locations_df["id"]
        except KeyError:
            if cfg.create_N.pad_gage_id:
                gage_ids = (
                    gage_locations_df["STAT_ID"].astype(str).apply(lambda x: x.zfill(8))
                )
            else:
                gage_ids = gage_locations_df["STAT_ID"]
        gdf = gdf[gdf["STAID"].isin(gage_ids)]
        
    # Step 1: filter by Zone
    gdf["COMID"] = gdf["COMID"].astype(int)
    zone_gdf = filter_by_comid_prefix(gdf, cfg.zone)
    if len(zone_gdf) == 0:
        log.info("No MERIT BASINS within this region")
        return False
    
    # Step 2: find matching COMID
    grouped = zone_gdf.groupby("STAID")
    unique_gdf = grouped.apply(choose_row_to_keep).reset_index(drop=True)
    
    # Step 3: ensure the DA of the gauge is located inside of the COMID
    upstream_comid_da, downstream_comid_da= find_upstream_das(unique_gdf, edges)
    filtered_gdf = filter_by_das(unique_gdf, upstream_comid_da, downstream_comid_da)
    
    # Step 4: match to a pour point
    zone_edge_ids = edges.id[:]
    zone_merit_basin_ids = edges.merit_basin[:]
    zone_upstream_areas = edges.uparea[:]
    # test_row = filtered_gdf.iloc[0]
    # edge_info = find_closest_edge(test_row, zone_edge_ids, zone_merit_basin_ids, zone_upstream_areas)
    edge_info = filtered_gdf.apply(
        lambda row: find_closest_edge(
            row, zone_edge_ids, zone_merit_basin_ids, zone_upstream_areas
        ),
        axis=1,
        result_type="expand",
    )
    filtered_gdf["edge_intersection"] = edge_info[0]  # TODO check if the data is empty
    filtered_gdf["zone_edge_id"] = edge_info[1]
    filtered_gdf["zone_edge_uparea"] = edge_info[2]
    filtered_gdf["zone_edge_vs_gage_area_difference"] = edge_info[3]
    filtered_gdf["drainage_area_percent_error"] = edge_info[4]
    filtered_gdf["a_merit_a_usgs_ratio"] = edge_info[5]
    
    # result_df = filtered_gdf[
    #     filtered_gdf["drainage_area_percent_error"] <= cfg.create_N.drainage_area_treshold
    # ]

    try:
        filtered_gdf["LNG_GAGE"] = filtered_gdf["LON_GAGE"]
        filtered_gdf["LAT_GAGE"] = filtered_gdf["Latitude"]
    except KeyError:
        pass
    if "HUC02" not in filtered_gdf.columns:
        filtered_gdf["HUC02"] = 0

    columns = [
        "STAID",
        # "STANAME",
        # "MERIT_ZONE",
        "HUC02",
        "DRAIN_SQKM",
        "LAT_GAGE",
        "LNG_GAGE",
        # "STATE",
        "COMID",
        "edge_intersection",
        "zone_edge_id",
        "zone_edge_uparea",
        "zone_edge_vs_gage_area_difference",
        "drainage_area_percent_error",
        "a_merit_a_usgs_ratio",
    ]
    result = filtered_gdf[columns]
    result = result.dropna()
    result["STAID"] = result["STAID"].astype(int)
    result.to_csv(Path(cfg.create_N.zone_obs_dataset), index=False)

    # combining this zone df to the master df that has all zonal information
    try:
        df = pd.read_csv(Path(cfg.create_N.obs_dataset_output))

        try:
            combined_df = pd.concat([df, result], ignore_index=True)
            sorted_df = combined_df.sort_values(by=["STAID"])
            sorted_df.to_csv(Path(cfg.create_N.obs_dataset_output), index=False)
        except pd.errors.InvalidIndexError:
            log.info(
                "Not merging your file with the master list as it seems like your gages are already included"
            )
    except FileNotFoundError:
        result.to_csv(Path(cfg.create_N.obs_dataset_output), index=False)
    return result


def create_gage_connectivity(
    cfg: DictConfig,
    edges: zarr.Group,
    gage_coo_root: zarr.Group,
    zone_csv: pd.DataFrame | gpd.GeoDataFrame,
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
        gage_output, _gage_id: str, root: zarr.Group
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
        pairs = format_pairs(gage_output)

        # Create a Zarr dataset for this specific gage
        single_gage_csr_data = root.require_group(_gage_id)
        single_gage_csr_data.create_dataset(
            "pairs", data=np.array(pairs), chunks=(10000,), dtype="float32"
        )

    def find_connections(row, coo_root, zone_attributes, _pad_gage_id=True):
        if _pad_gage_id:
            _gage_id = str(row["STAID"]).zfill(8)
        else:
            _gage_id = str(object=row["STAID"])
        edge_id = row["edge_intersection"]
        zone_edge_id = row["zone_edge_id"]
        if _gage_id not in coo_root:
            gage_output = stack_traversal(
                _gage_id, edge_id, zone_edge_id, zone_attributes
            )
            create_coo_data(gage_output, _gage_id, coo_root)

    def apply_find_connections(row, gage_coo_root, edges, pad_gage_id):
        return find_connections(row, gage_coo_root, edges, pad_gage_id)

    pad_gage_id = cfg.create_N.pad_gage_id
    dask_df = dd.from_pandas(zone_csv, npartitions=16)
    result = dask_df.apply(
        apply_find_connections,
        args=(gage_coo_root, edges, pad_gage_id),
        axis=1,
        meta=(None, "object"),
    )
    with ProgressBar():
        _ = result.compute()


def new_zone_connectivity(
    cfg: DictConfig,
    edges: zarr.Group,
    full_zone_root: zarr.Group,
) -> None:
    def find_connection(edges: zarr.Group, mapping: dict) -> dict:
        """
        Performs a traversal on a graph of river flowlines, represented by a NumPy array of edges.
        This function iterates over each edge and explores its upstream connections, constructing a graph structure.

        Parameters:
        gage_id (str): Identifier for the gage being processed.
        edges (np.ndarray): A NumPy array containing river flowline data. Each element is expected to have 'id' and 'up' attributes.
        mapping (dict): A dictionary mapping node IDs to their indices in the edges array.

        Returns:
        dict: A dictionary representing the traversed river graph for each edge. It contains three keys:
              'ID' - a list of node identifiers,
              'idx' - a list of indices corresponding to the nodes,
              'up' - a list of lists, where each sublist contains indices of upstream nodes.
        """
        river_graph = {"ds": [], "up": []}
        for idx in tqdm(
            range(edges.id.shape[0]),
            desc="Looping through edges",
            ascii=True,
            ncols=140,
        ):
            river_graph["ds"].append(idx)

            # Decode upstream ids and convert to indices
            upstream_ids = ast.literal_eval(edges.up[idx])
            if len(upstream_ids) == 0:
                river_graph["up"].append([None])
            else:
                upstream_indices = [mapping.get(up_id, None) for up_id in upstream_ids]
                river_graph["up"].append(upstream_indices)
        return river_graph

    mapping = {id_: i for i, id_ in enumerate(edges.id[:])}
    # bag = db.from_sequence(range(edges.id.shape[0]), npartitions=100)  # Adjust the number of partitions as needed
    # results = bag.map(lambda idx: find_connection(idx, edges, mapping))
    # with ProgressBar():
    #     traversed_graphs = results.compute()
    river_graph = find_connection(edges, mapping)

    # combined_graph = {"ds": [], "up": []}
    # for graph in traversed_graphs:
    #     combined_graph["ds"].extend(graph["ds"])
    #     combined_graph["up"].extend(graph["up"])

    pairs = format_pairs(river_graph)

    full_zone_root.create_dataset(
        "pairs", data=np.array(pairs), chunks=(5000, 5000), dtype="float32"
    )
