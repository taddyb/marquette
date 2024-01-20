import logging
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import torch
import xarray as xr
import zarr

log = logging.getLogger(__name__)

from marquette.merit._edge_calculations import (
    calculate_num_edges,
    create_segment,
    find_flowlines,
    many_segment_to_edge_partition,
    singular_segment_to_edge_partition,
    string_to_dict_builder,
    sort_xarray_dataarray,
)

from marquette.merit._connectivity_matrix import downstream_map

from marquette.merit._TM_calculations import (
    create_HUC_MERIT_TM,
    create_MERIT_FLOW_TM,
    join_geospatial_data,
)
from marquette.merit._streamflow_conversion_functions import (
    calculate_from_qr_files,
    calculate_from_individual_files,
)


def write_streamflow(cfg: DictConfig) -> None:
    """
    Process and write streamflow data to a Zarr store.

    This function reads streamflow data from CSV files, processes it according to
    the provided configuration, and writes the results to a Zarr store. It handles
    the conversion of streamflow units, sorting of data into bins, and management of
    missing data. The function creates two Zarr datasets: one for the streamflow
    predictions and another for the corresponding HUC keys.

    Parameters:
    cfg (DictConfig): A Hydra DictConfig object containing configuration settings.
                      The configuration should include paths for streamflow files,
                      attribute files, Zarr store locations, and unit settings.

    Raises:
    IndexError: If an index error occurs while processing the streamflow data. This
                can happen if there's a mismatch between the data in the CSV files
                and the expected structure or if a specific HUC is missing in the
                attributes file.

    Notes:
    - The function assumes the streamflow data is in CSV files located in a specified
      directory and that the HUC IDs are present in an attributes CSV file.
    - The function handles unit conversion (e.g., from mm/day to m³/s) based on the
      configuration settings.
    - Data is sorted and binned based on HUC IDs, and missing data for any HUC is
      handled by inserting zeros.
    - The processed data is stored in a Zarr store with separate datasets for streamflow
      predictions and HUC keys.

    Example Usage:
    --------------
    cfg = DictConfig({
        'save_paths': {
            'attributes': 'path/to/attributes.csv',
            'streamflow_files': 'path/to/streamflow/files'
        },
        'zarr': {
            'HUC_TM': 'path/to/huc_tm.zarr',
            'streamflow': 'path/to/streamflow.zarr',
            'streamflow_keys': 'path/to/streamflow_keys.zarr'
        },
        'units': 'mm/day'
    })
    write_streamflow(cfg)
    """
    streamflow_path = Path(cfg.zarr.streamflow)
    if streamflow_path.exists() is False:
        if cfg.individual_streamflow_files:
            """Expecting to read from individual files"""
            calculate_from_individual_files(cfg)
        else:
            """Expecting to read data from QR files"""
            calculate_from_qr_files(cfg)
    else:
        log.info("Streamflow data already exists")


def create_edges(cfg: DictConfig) -> zarr.hierarchy.Group:
    root = zarr.open_group(Path(cfg.zarr.edges), mode="a")
    group_name = f"{cfg.continent}{cfg.area}"
    if group_name in root:
        log.info("Edge data already exists in zarr format")
        edges = root.require_group(group_name)
    else:
        flowline_file: Path = find_flowlines(cfg)
        polyline_gdf: gpd.GeoDataFrame = gpd.read_file(flowline_file)
        dx: int = cfg.dx  # Unit: Meters
        buffer: float = cfg.buffer * dx  # Unit: Meters
        for col in [
            "COMID",
            "NextDownID",
            "up1",
            "up2",
            "up3",
            "up4",
            "maxup",
            "order",
        ]:
            polyline_gdf[col] = polyline_gdf[col].astype(int)
        computed_series = polyline_gdf.apply(
            lambda df: create_segment(df, polyline_gdf.crs, dx, buffer), axis=1
        )
        segments_dict = computed_series.to_dict()
        segment_das = {
            segment["id"]: segment["uparea"] for segment in segments_dict.values()
        }
        sorted_keys = sorted(
            segments_dict, key=lambda key: segments_dict[key]["uparea"]
        )
        num_edges_dict = {
            _segment["id"]: calculate_num_edges(_segment["len"], dx, buffer)
            for _, _segment in tqdm(
                segments_dict.items(), desc="Processing Number of Edges"
            )
        }
        one_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(), desc="Filtering Segments == 1"
            )
            if edge_info[0] == 1
        }
        many_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(), desc="Filtering Segments > 1"
            )
            if edge_info[0] > 1
        }
        segments_with_more_than_one_edge = {}
        segments_with_one_edge = {}
        for i, segment in segments_dict.items():
            segment_id = segment["id"]
            segment["index"] = i

            if segment_id in many_edge_segment:
                segments_with_more_than_one_edge[segment_id] = segment
            elif segment_id in one_edge_segment:
                segments_with_one_edge[segment_id] = segment
            else:
                print(f"MISSING ID: {segment_id}")

        df_one = pd.DataFrame.from_dict(segments_with_one_edge, orient="index")
        df_many = pd.DataFrame.from_dict(
            segments_with_more_than_one_edge, orient="index"
        )
        ddf_one = dd.from_pandas(df_one, npartitions=cfg.num_partitions)
        ddf_many = dd.from_pandas(df_many, npartitions=cfg.num_partitions)

        meta = pd.DataFrame(
            {
                "id": pd.Series(dtype="str"),
                "merit_basin": pd.Series(dtype="int"),
                "segment_sorting_index": pd.Series(dtype="int"),
                "order": pd.Series(dtype="int"),
                "len": pd.Series(dtype="float"),
                "len_dir": pd.Series(dtype="float"),
                "ds": pd.Series(dtype="str"),
                "up": pd.Series(dtype="object"),  # List or array
                "up_merit": pd.Series(dtype="object"),  # List or array
                "slope": pd.Series(dtype="float"),
                "sinuosity": pd.Series(dtype="float"),
                "stream_drop": pd.Series(dtype="float"),
                "uparea": pd.Series(dtype="float"),
                "coords": pd.Series(dtype="str"),
                "crs": pd.Series(dtype="object"),  # CRS object
            }
        )

        edges_results_one = ddf_one.map_partitions(
            singular_segment_to_edge_partition,
            edge_info=one_edge_segment,
            num_edge_dict=num_edges_dict,
            segment_das=segment_das,
            meta=meta,
        )
        edges_results_many = ddf_many.map_partitions(
            many_segment_to_edge_partition,
            edge_info=many_edge_segment,
            num_edge_dict=num_edges_dict,
            segment_das=segment_das,
            meta=meta,
        )
        edges_results_one_df = edges_results_one.compute()
        edges_results_many_df = edges_results_many.compute()
        merged_df = pd.concat([edges_results_one_df, edges_results_many_df])
        for col in ["id", "ds", "up", "coords", "up_merit", "crs"]:
            merged_df[col] = merged_df[col].astype(str)
        xr_dataset = xr.Dataset.from_dataframe(merged_df)
        sorted_keys_array = np.array(sorted_keys)
        sorted_edges = xr.Dataset()
        edges = root.create_group(group_name)
        for var_name in xr_dataset.data_vars:
            sorted_edges[var_name] = sort_xarray_dataarray(
                xr_dataset[var_name],
                sorted_keys_array,
                xr_dataset["segment_sorting_index"].values,
            )
            shape = sorted_edges[var_name].shape
            dtype = sorted_edges[var_name].dtype
            tmp = edges.zeros(var_name, shape=shape, chunks=1000, dtype=dtype)
            tmp[:] = sorted_edges[var_name].values
        tmp = edges.zeros(
            "sorted_keys",
            shape=sorted_keys_array.shape,
            chunks=1000,
            dtype=sorted_keys_array.dtype,
        )
        tmp[:] = sorted_keys_array
    return edges


def create_N(cfg: DictConfig, edges: zarr.hierarchy.Group) -> None:
    root = zarr.open_group(Path(cfg.zarr.csr_matrix), mode="a")
    group_name = f"{cfg.continent}{cfg.area}"
    if group_name in root:
        log.info("Connectivity Matrix (N) already exists")
    else:
        id_to_index = {id_val: idx for idx, id_val in enumerate(edges.id[:])}
        rows, cols, data, visited = [], [], [], set()

        for id_index in tqdm(range(len(edges.id)), desc="Mapping Downstream"):
            downstream_map(id_index, edges, rows, cols, data, id_to_index, visited)

        rows_tensor = torch.tensor(rows, dtype=torch.int64)
        cols_tensor = torch.tensor(cols, dtype=torch.int64)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        csr_data = root.create_group(cfg.save_name)
        csr_data.create_dataset(
            "rows", data=rows_tensor.numpy(), chunks=(10000,), dtype="i8"
        )
        csr_data.create_dataset(
            "cols", data=cols_tensor.numpy(), chunks=(10000,), dtype="i8"
        )
        csr_data.create_dataset(
            "data", data=data_tensor.numpy(), chunks=(10000,), dtype="f4"
        )


def create_TMs(cfg: DictConfig, edges: zarr.hierarchy.Group) -> None:
    huc_to_merit_path = Path(cfg.zarr.HUC_TM)
    if huc_to_merit_path.exists():
        log.info("HUC -> MERIT data already exists in zarr format")
        huc_to_merit_TM = zarr.open(huc_to_merit_path, mode="r")
    else:
        log.info(f"Creating HUC10 -> MERIT TM")
        overlayed_merit_basins = join_geospatial_data(cfg)
        huc_to_merit_TM = create_HUC_MERIT_TM(cfg, overlayed_merit_basins)
    merit_to_river_graph_path = Path(cfg.zarr.MERIT_TM)
    if merit_to_river_graph_path.exists():
        log.info("MERIT -> FLOWLINE data already exists in zarr format")
        merit_to_river_graph_TM = zarr.open(merit_to_river_graph_path, mode="r")
    else:
        log.info(f"Creating MERIT -> FLOWLINE TM")
        merit_to_river_graph_TM = create_MERIT_FLOW_TM(cfg, edges, huc_to_merit_TM)
    return huc_to_merit_TM
