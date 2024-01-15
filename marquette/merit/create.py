import logging
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask_geopandas as dg
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
    sort_xarray_dataarray,
)
from marquette.merit._TM_calculations import (
    create_HUC_MERIT_TM,
    create_MERIT_FLOW_TM,
    join_geospatial_data,
)
from marquette.merit._streamflow_conversion_functions import (
    extract_numbers,
    _sort_into_bins,
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
    streamflow_nc_path = Path(cfg.netcdf.streamflow)
    if streamflow_nc_path.exists() is False:
        attrs_df = pd.read_csv(cfg.save_paths.attributes)
        huc10_ids = attrs_df["gage_ID"].values.astype("str")
        huc_to_merit_TM = zarr.open(Path(cfg.zarr.HUC_TM), mode="r")
        huc_10_list = huc_to_merit_TM.HUC10[:]
        bins_size = 1000
        bins = [
            huc10_ids[i : i + bins_size] for i in range(0, len(huc10_ids), bins_size)
        ]
        basin_hucs = huc_10_list
        basin_indexes = _sort_into_bins(basin_hucs, bins)
        streamflow_data = []
        columns = []
        folder = Path(cfg.save_paths.streamflow_files)
        file_paths = [file for file in folder.glob("*") if file.is_file()]
        file_paths.sort(key=extract_numbers)
        iterable = basin_indexes.keys()
        pbar = tqdm(iterable)
        for i, key in enumerate(pbar):
            pbar.set_description(f"Processing Qr files")
            values = basin_indexes[key]
            if values:
                file = file_paths[i]
                df = pd.read_csv(file, dtype=np.float32, header=None)
                for val in values:
                    id = list(val.keys())[0]
                    columns.append(id)
                    row = attrs_df[attrs_df["gage_ID"] == id]
                    try:
                        attr_idx = row.index[0]
                        try:
                            row_idx = attr_idx - (
                                key * 1000
                            )  # taking only the back three numbers
                            _streamflow = df.iloc[row_idx].values
                        except IndexError as e:
                            raise e
                        if cfg.units.lower() == "mm/day":
                            # converting from mm/day to m3/s
                            area = row["area"].values[0]
                            _streamflow = _streamflow * area * 1000 / 86400
                        streamflow_data.append(_streamflow)
                    except IndexError:
                        log.info(f"HUC10 {id} is missing from the attributes file.")
                        no_pred = np.zeros([14610])
                        streamflow_data.append(no_pred)
                        continue
        array = np.array(streamflow_data).T
        column_keys = np.array(columns)
        date_range = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
        ds = xr.Dataset(
            {"streamflow": (["time", "location"], array)},
            coords={"time": date_range, "location": column_keys},
        )
        ds_interpolated = ds.interp(
            time=pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="H"),
            method="linear",
        )
        ds_interpolated.to_netcdf(Path(cfg.netcdf.streamflow))
    else:
        log.info("Streamflow data already exists in netcdf format")


def convert_streamflow(cfg: DictConfig) -> None:
    """
    Convert streamflow data from CSV files to a Zarr group format.

    This function reads streamflow data from multiple CSV files located in a specified
    directory, converts each file to a NumPy array, and then stores each array as a
    dataset in a Zarr group. The function creates the Zarr group if it does not
    already exist. Each dataset within the Zarr group is named after the corresponding
    file.

    Parameters:
    cfg (DictConfig): A Hydra DictConfig configuration object. The configuration
                      should contain the following keys:
                      - zarr.streamflow: The path where the Zarr group will be created.
                      - save_paths.streamflow_files: The directory containing the CSV
                                                     files with streamflow data.

    Returns:
    None: This function does not return anything. It writes the converted data to
          disk in Zarr group format.

    Raises:
    FileNotFoundError: If the specified directory for streamflow CSV files does not exist.
    IOError: If there is an issue reading the CSV files or writing to the Zarr group.

    Example usage:
    ```
    cfg = DictConfig({'zarr': {'streamflow': '/path/to/zarr/output'},
                      'save_paths': {'streamflow_files': '/path/to/csv/files'}})
    convert_streamflow(cfg)
    ```
    """
    try:
        streamflow_output = Path(cfg.zarr.streamflow)
        if not streamflow_output.exists():
            folder = Path(cfg.save_paths.streamflow_files)
            if not folder.exists():
                raise FileNotFoundError(f"Specified directory does not exist: {folder}")
            file_paths = [file for file in folder.glob("*") if file.is_file()]
            file_paths.sort(key=extract_numbers)
            zarr_group = zarr.open_group(streamflow_output, mode="w")
            for file in file_paths:
                try:
                    array = pd.read_csv(file, dtype=np.float32, header=None).to_numpy()
                    zarr_group.create_dataset(file.name, data=array)
                    log.info(f"Wrote {file.name} to disk")
                except IOError as e:
                    log.info(f"Error processing file {file}: {e}")
        else:
            log.info(f"Zarr group already exists: {streamflow_output}")

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
    except IOError as e:
        log.error(f"I/O error occurred: {e}")


def create_edges(cfg: DictConfig) -> zarr.hierarchy.Group:
    edges_file = Path(cfg.zarr.edges)
    if edges_file.exists():
        log.info("Edge data already exists in zarr format")
        edges = zarr.open(edges_file, mode="r")
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
        crs = polyline_gdf.crs
        dask_gdf = dg.from_geopandas(polyline_gdf, npartitions=cfg.num_partitions)
        meta = pd.Series([], dtype=object)
        with ProgressBar():
            computed_series: dd.Series = dask_gdf.map_partitions(
                lambda df: df.apply(create_segment, args=(crs, dx, buffer), axis=1),
                meta=meta,
            ).compute()

        segments_dict = computed_series.to_dict()
        sorted_keys = sorted(
            segments_dict, key=lambda key: segments_dict[key]["uparea"]
        )
        segment_das = {
            segment["id"]: segment["uparea"] for segment in segments_dict.values()
        }
        num_edges_dict = {
            segment_["id"]: calculate_num_edges(segment_["len"], dx, buffer)
            for seg_id, segment_ in tqdm(
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
                "coords": gpd.GeoSeries(
                    dtype="geometry"
                ),  # Assuming this is a geometry column
                "crs": pd.Series(dtype="object"),  # CRS object
            }
        )

        edges_results_one = ddf_one.map_partitions(
            singular_segment_to_edge_partition,
            edge_info=one_edge_segment,
            segment_das=segment_das,
            meta=meta,
        )
        edges_results_many = ddf_many.map_partitions(
            many_segment_to_edge_partition,
            edge_info=many_edge_segment,
            segment_das=segment_das,
            meta=meta,
        )
        for i, segment in segments_dict.items():
            segment_id = segment["id"]
            segment["index"] = i
        edges_results_one_df = edges_results_one.compute()
        edges_results_many_df = edges_results_many.compute()
        merged_df = pd.concat([edges_results_one_df, edges_results_many_df])
        for col in ["id", "ds", "up", "coords", "up_merit", "crs"]:
            merged_df[col] = merged_df[col].astype(str)
        xr_dataset = xr.Dataset.from_dataframe(merged_df)
        sorted_keys_array = np.array(sorted_keys)
        sorted_edges = xr.Dataset()
        for var_name in xr_dataset.data_vars:
            sorted_edges[var_name] = sort_xarray_dataarray(
                xr_dataset[var_name],
                sorted_keys_array,
                xr_dataset["segment_sorting_index"].values,
            )
        sorted_edges.to_zarr(Path(cfg.zarr.edges), mode="w")
        edges = zarr.open_group(Path(cfg.zarr.edges), mode="r")
        zarr.save(cfg.zarr.sorted_edges_keys, sorted_keys_array)
    return edges


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
