import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def calculate_drainage_area_for_all_edges(edges, segment_das):
    num_edges = len(edges)
    up_ids = edges[0]["up"]
    if up_ids:
        for idx, edge in enumerate(edges):
            try:
                prev_up_area = sum(segment_das[seg] for seg in edge["up_merit"])
            except KeyError:
                edge["up_merit"] = ast.literal_eval(edge["up_merit"])
                prev_up_area = sum(segment_das[seg] for seg in edge["up_merit"])
            area_difference = edge["uparea"] - prev_up_area
            even_distribution = area_difference / num_edges
            edge["uparea"] = prev_up_area + even_distribution * (idx + 1)
    else:
        total_uparea = edges[0]["uparea"]
        even_distribution = total_uparea / num_edges
        for idx, edge in enumerate(edges):
            edge["uparea"] = even_distribution * (idx + 1)
    return edges


def calculate_num_edges(length: float, dx: float, buffer: float) -> Tuple:
    """
    Calculate the number of edges and the length of each edge for a given segment.

    This function determines the number of edges a segment should be divided into,
    based on its length, a desired edge length (dx), and a tolerance (buffer).
    The function adjusts the number of edges to ensure that the deviation of the
    actual edge length from dx is within the specified buffer.

    Parameters
    ----------
    length : float
        The length of the segment for which to calculate the number of edges.
    dx : float
        The desired length of each edge.
    buffer : float
        The acceptable deviation from the desired edge length (dx).

    Returns
    -------
    tuple
        A tuple containing two elements:
            - The first element is an integer representing the number of edges.
            - The second element is a float representing the actual length of each edge.

    Examples
    --------
    >> calculate_num_edges(100, 30, 5)
    (3, 33.333333333333336)

    >> calculate_num_edges(100, 25, 2)
    (4, 25.0)
    """
    num_edges = length // dx
    if num_edges == 0:
        num_edges = 1
        if dx - length < buffer:
            edge_len = length
        else:
            edge_len = dx
    else:
        edge_len = length / num_edges
        buf_dev = edge_len - dx
        while abs(buf_dev) > buffer:
            if buf_dev > dx:
                num_edges -= 1
            else:
                num_edges += 1
            edge_len = length / num_edges
            buf_dev = edge_len - dx
    return (int(num_edges), edge_len)


def create_edge_json(segment_row: pd.Series, up=None, ds=None, edge_id=None) -> Dict[str, Any]:
    """
    Create a JSON representation of an edge based on segment data.

    Parameters
    ----------
    segment_row : pandas.Series
        A series representing a row from the segment DataFrame.
    up : list, optional
        List of upstream segment IDs.
    ds : str, optional
        Downstream segment ID.
    edge_id : str, optional
        Unique identifier for the edge.

    Returns
    -------
    dict
        Dictionary representing the edge with various attributes.
    """
    edge = {
        "id": edge_id,
        "merit_basin": segment_row["id"],
        "segment_sorting_index": segment_row["index"],
        "order": segment_row["order"],
        "len": segment_row["len"],
        "len_dir": segment_row["len_dir"],
        "ds": ds,
        "up": up,
        "up_merit": segment_row["up"],
        "slope": segment_row["slope"],
        "sinuosity": segment_row["sinuosity"],
        "stream_drop": segment_row["stream_drop"],
        "uparea": segment_row["uparea"],
        "coords": segment_row["coords"],
        "crs": segment_row["crs"],
    }
    return edge


def create_segment(row: pd.Series, crs: Any, dx: int, buffer: float) -> Dict[str, Any]:
    """
    Create a dictionary representation of a segment using its row data.

    This function is a wrapper that calls 'create_segment_dict' by passing the
    geometry of the segment along with other attributes. It simplifies the creation
    of a segment dictionary from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A series representing a row from a DataFrame containing segment data.
    crs : Any
        Coordinate reference system of the segment.
    dx : int
        Desired length of each edge in the segment (used in further calculations).
    buffer : float
        Buffer tolerance for edge length calculation.

    Returns
    -------
    dict
        Dictionary containing segment attributes.
    """
    return dict(create_segment_dict(row, row.geometry, crs, dx, buffer))


def string_to_dict_builder(input_str, crs_info):
    """
    Convert a string representation of a dictionary to an actual dictionary
    using a modular approach for different sections.

    Parameters:
    input_str (str): The string to be converted.
    crs_info (str): The CRS information to be inserted.

    Returns:
    dict: The resulting dictionary.
    """

    def handle_list(section):
        """
        Handles the parsing of list structures.
        """
        try:
            return ast.literal_eval(section.strip())
        except Exception as e:
            return f"Error parsing list: {e}"

    result_dict = {}
    # Using regex to extract key-value pairs
    pattern = r"'([^']+)':\s*((?:\[.*?\]|<.*?>|'.*?'|[^,]+)*)"
    matches = re.findall(pattern, input_str)

    for key, value in matches:
        key = key.strip()
        if key == "crs":
            result_dict[key] = crs_info
        elif key == "coords":
            result_dict[key] = value.strip().strip("'")
        else:
            result_dict[key] = handle_list(value)

    return result_dict


def create_segment_dict(
    row: pd.Series,
    segment_coords: List[Tuple[float, float]],
    crs: Any,
    dx: int,
    buffer: float,
) -> Dict[str, Any]:
    """
    Create a dictionary representation of a segment with various attributes.

    This function constructs a dictionary for a river segment based on provided
    attributes. It includes details such as segment ID, order, length, downstream
    ID, slope, sinuosity, stream drop, upstream area, coordinates, and CRS.

    Parameters
    ----------
    row : pandas.Series
        A series representing a row from a DataFrame containing segment data.
    segment_coords : List[Tuple[float, float]]
        List of tuples representing coordinates of the segment.
    crs : Any
        Coordinate reference system of the segment.
    dx : int
        Desired length of each edge in the segment (used in further calculations).
    buffer : float
        Buffer tolerance for edge length calculation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing segment attributes.
    """
    segment_dict = {
        "id": row["COMID"],
        "order": row["order"],
        "len": row["lengthkm"] * 1000,  # to meters
        "len_dir": row["lengthdir"] * 1000,  # to meters
        "ds": row["NextDownID"],
        # 'is_headwater': False,
        "up": [row[key] for key in ["up1", "up2", "up3", "up4"] if row[key] != 0]
        if row["maxup"] > 0
        else ([] if row["order"] == 1 else []),
        "slope": row["slope"],
        "sinuosity": row["sinuosity"],
        "stream_drop": row["strmDrop_t"],
        "uparea": row["uparea"],
        "coords": segment_coords,
        "crs": crs,
    }

    return segment_dict


def get_upstream_ids(row: pd.Series, edge_info: dict):
    """
    Generate upstream IDs for a segment.

    Parameters
    ----------
    row : pandas.Series
        A series representing a row from the segment DataFrame.
    edge_info : dict
        The number of edges associated with the segment.

    Returns
    -------
    list
        List of upstream segment IDs.
    """
    up_ids = []
    if row["up"] is None:
        return up_ids
    try:
        if isinstance(row["up"], str):
            row["up"] = ast.literal_eval(row["up"])
        for id in row["up"]:
            num_edges, _ = edge_info[id]
            up_ids.append(f"{id}_{num_edges - 1}")
    except KeyError:
        log.error(f"KeyError with segment {row['id']}")
        return []
    return up_ids


def find_flowlines(cfg: DictConfig) -> Path:
    """
    A function to find the correct flowline of all MERIT basins using glob

    Parameters
    ----------
    cfg : DictConfig
        The cfg object

    Returns
    -------
    Path
        The file that we're going to create flowline connectivity for

    Raises
    ------
    IndexError
        Raised if no flowlines are found with your MERIT region code
    """
    flowline_path = Path(cfg.create_edges.flowlines)
    region_id = f"_{cfg.zone}_"
    matching_file = flowline_path.glob(f"*{region_id}*.shp")
    try:
        found_file = [file for file in matching_file][0]
        return found_file
    except IndexError:
        raise IndexError(f"No flowlines found using: *{region_id}*.shp")


def many_segment_to_edge_partition(
    df: pd.DataFrame,
    edge_info: Dict[str, Any],
    num_edge_dict: Dict[str, Any],
    segment_das: Dict[str, float],
) -> pd.DataFrame:
    """
    Process a DataFrame partition to create edges for segments with multiple edges.

    This function iterates over each segment in the DataFrame partition, computes
    the edge length, upstream IDs, and creates a JSON representation for each edge.
    It is specifically designed for segments that have multiple edges.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame partition containing segment data.
    edge_info : dict
        Dictionary containing information about the number of edges and edge length
        for each segment.
    segment_das : dict
        Dictionary containing drainage area data for each segment.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing edge data for all segments in the partition.
    """
    all_edges = []
    for _, segment in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing Segments",
        ncols=140,
        ascii=True,
    ):
        all_segment_edges = []
        num_edges, edge_len = edge_info[segment["id"]]
        up_ids = get_upstream_ids(segment, num_edge_dict)
        for i in range(num_edges):
            if i == 0:
                edge = create_edge_json(
                    segment,
                    up=up_ids,
                    ds=f"{segment['id']}_{i + 1}",
                    edge_id=f"{segment['id']}_{i}",
                )
            else:
                edge = create_edge_json(
                    segment,
                    up=[f"{segment['id']}_{i - 1}"],
                    ds=f"{segment['id']}_{i + 1}" if i < num_edges - 1 else f"{segment['ds']}_0",
                    edge_id=f"{segment['id']}_{i}",
                )
            edge["len"] = edge_len
            edge["len_dir"] = edge_len / segment["sinuosity"]
            all_segment_edges.append(edge)
        all_segment_edges = calculate_drainage_area_for_all_edges(all_segment_edges, segment_das)
        for edge in all_segment_edges:
            all_edges.append(edge)
    return pd.DataFrame(all_edges)


def singular_segment_to_edge_partition(
    df: pd.DataFrame,
    edge_info: Dict[str, Any],
    num_edge_dict: Dict[str, Any],
    segment_das: Dict[str, float],
) -> pd.DataFrame:
    """
    Process a DataFrame partition to create edges for each segment.

    This function iterates over each segment in the DataFrame, computes the edge
    length, upstream IDs, and creates JSON representation of each edge. It handles
    segments that are associated with only one edge.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame partition containing segment data.
    edge_info : dict
        Dictionary containing edge information for each segment.
    segment_das : dict
        Dictionary containing drainage area data for each segment.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing edge data for all segments in the partition.
    """
    all_edges = []
    for _, segment in tqdm(
        df.iterrows(),
        total=len(df),
        ncols=140,
        ascii=True,
    ):
        __, edge_len = edge_info[segment["id"]]
        up_ids = get_upstream_ids(segment, num_edge_dict)
        edge = create_edge_json(
            segment,
            up=up_ids,
            ds=f"{segment['ds']}_0",
            edge_id=f"{segment['id']}_0",
        )
        edge["len"] = edge_len
        edge["len_dir"] = edge_len / segment["sinuosity"]
        all_edges.append(edge)
    return pd.DataFrame(all_edges)


def _plot_gdf(gdf: gpd.GeoDataFrame) -> None:
    """
    A function to find the correct flowline of all MERIT basins using glob

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geodataframe you want to plot

    Returns
    -------
    None

    Raises
    ------
    None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax)
    ax.set_title("Polyline Plot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def sort_based_on_keys(array_to_sort, keys, segment_sorted_index):
    """
    Sort 'array_to_sort' based on the order defined in 'keys'.
    For each key, find rows in 'segment_sorted_index' where this value occurs.
    If there are multiple occurrences, sort these rows further by ID.

    Args:
    array_to_sort: The array to be sorted.
    keys: The array of keys to sort by.
    segment_sorted_index: The index array to match keys against.

    Returns:
    A sorted version of 'array_to_sort'.
    """
    sorted_array = []
    for key in tqdm(
        keys,
        ncols=140,
        ascii=True,
    ):
        matching_indices = np.where(segment_sorted_index == key)[0]
        if len(matching_indices) > 1:
            sorted_indices = np.sort(matching_indices)
        else:
            sorted_indices = matching_indices
        sorted_array.extend(array_to_sort[sorted_indices])
    return np.array(sorted_array)


def sort_xarray_dataarray(da, keys, segment_sorted_index):
    sorted_data = sort_based_on_keys(da.values, keys, segment_sorted_index)
    return xr.DataArray(sorted_data, dims=da.dims, coords=da.coords)
