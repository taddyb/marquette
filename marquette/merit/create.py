from collections import defaultdict
import logging
from pathlib import Path

import dask.dataframe as dd
import fiona
import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
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

from marquette.merit._connectivity_matrix import (
    create_gage_connectivity,
    map_gages_to_zone,
    new_zone_connectivity,
)

from marquette.merit._TM_calculations import (
    create_HUC_MERIT_TM,
    create_MERIT_FLOW_TM,
    join_geospatial_data,
)
from marquette.merit._streamflow_conversion_functions import (
    calculate_from_individual_files,
    separate_basins,
)


def write_streamflow(cfg: DictConfig) -> None:
    """
    Process and write streamflow data to a Zarr store.
    """
    streamflow_path = Path(cfg.create_streamflow.data_store)
    version = cfg.create_streamflow.version.lower()
    split_output = ["dpl_v1", "dpl_v2", "dpl_v2.5", "dpl_v3-pre"]
    if streamflow_path.exists() is False:
        if (version in split_output):
            """Expecting to read from individual files"""
            streamflow_files_path = separate_basins(cfg)
        else:
            streamflow_files_path = Path(cfg.save_paths.streamflow_files)
        calculate_from_individual_files(cfg, streamflow_files_path)
    else:
        log.info("Streamflow data already exists")


def create_edges(cfg: DictConfig) -> zarr.hierarchy.Group:
    root = zarr.open_group(Path(cfg.create_edges.edges), mode="a")
    group_name = f"{cfg.zone}"
    if group_name in root:
        log.info("Edge data already exists in zarr format")
        edges = root.require_group(group_name)
    else:
        flowline_file = find_flowlines(cfg)
        polyline_gdf = gpd.read_file(flowline_file)
        dx = cfg.create_edges.dx  # Unit: Meters
        buffer = cfg.create_edges.buffer * dx  # Unit: Meters
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
                segments_dict.items(), desc="Processing Number of Edges", ncols=140, ascii=True,
            )
        }
        one_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(), desc="Filtering Segments == 1", ncols=140, ascii=True,
            )
            if edge_info[0] == 1
        }
        many_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(), desc="Filtering Segments > 1", ncols=140, ascii=True,
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
        ddf_one = dd.from_pandas(df_one, npartitions=64)
        ddf_many = dd.from_pandas(df_many, npartitions=64)

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
    gage_coo_root = zarr.open_group(Path(cfg.create_N.gage_coo_indices), mode="a")
    zone_root = gage_coo_root.require_group(cfg.zone)
    if cfg.create_N.run_whole_zone:
        if "full_zone" in zone_root:
            log.info("Full zone already exists")
        else:
            full_zone_root = zone_root.require_group("full_zone")
            new_zone_connectivity(cfg, edges, full_zone_root)
        log.info("Full zone sparse Matrix created")
    else:
        zone_csv_path = Path(cfg.create_N.zone_obs_dataset)
        if zone_csv_path.exists():
            zone_csv = pd.read_csv(zone_csv_path)
        else:
            zone_csv = map_gages_to_zone(cfg, edges)
        if zone_csv is not False:
            create_gage_connectivity(cfg, edges, zone_root, zone_csv)
            log.info("All sparse gage matrices are created")


def create_TMs(cfg: DictConfig, edges: zarr.hierarchy.Group) -> None:
    if "HUC" in cfg.create_TMs:
        huc_to_merit_path = Path(cfg.create_TMs.HUC.TM)
        if huc_to_merit_path.exists():
            log.info("HUC -> MERIT data already exists in zarr format")
        else:
            log.info(f"Creating HUC10 -> MERIT TM")
            overlayed_merit_basins = join_geospatial_data(cfg)
            create_HUC_MERIT_TM(cfg, edges, overlayed_merit_basins)
    merit_to_river_graph_path = Path(cfg.create_TMs.MERIT.TM)
    if merit_to_river_graph_path.exists():
        log.info("MERIT -> FLOWLINE data already exists in zarr format")
    else:
        log.info(f"Creating MERIT -> FLOWLINE TM")
        create_MERIT_FLOW_TM(cfg, edges)
