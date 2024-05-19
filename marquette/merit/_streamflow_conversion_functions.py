import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from omegaconf import DictConfig
from tqdm import tqdm, trange

log = logging.getLogger(__name__)


def calculate_huc10_flow_from_individual_files(cfg: DictConfig) -> None:
    """Converts huc10 flow from the provided indivudual files to m3/s and saves to zarr

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    """
    warnings.warn("This function is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)
    qr_folder = Path(cfg.create_streamflow.predictions)
    streamflow_files_path = qr_folder / "basin_split/"

    attrs_df = pd.read_csv(cfg.create_streamflow.obs_attributes)
    attrs_df["gage_ID"] = (
        attrs_df["gage_ID"].astype(str).str.zfill(10)
    )  # Left padding a 0 to make sure that all gages can be read
    id_to_area = attrs_df.set_index("gage_ID")["area"].to_dict()

    huc_to_merit_TM = zarr.open(Path(cfg.create_TMs.HUC.TM).__str__(), mode="r")
    huc_10_list = huc_to_merit_TM.HUC10[:] # type: ignore
    date_range = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="D",
    )
    streamflow_data = np.zeros((len(date_range), len(huc_10_list)))

    for i, huc_id in enumerate(
        tqdm(
            huc_10_list,
            desc="Processing River flowlines",
            ncols=140,
            ascii=True,
        )
    ):
        try:
            file_path = streamflow_files_path / f"{huc_id}.npy"
            data = np.load(file_path)
            file_id = file_path.stem
            area = id_to_area.get(file_id)  # defaulting to mean area if there is no area for the HUC10
            # CONVERTING FROM MM/DAY TO M3/S
            data = data * area * 1000 / 86400
            streamflow_data[:, i] = data
        except FileNotFoundError:
            log.info(f"No Predictions found for {huc_id}")
        except KeyError:
            log.info(f"{huc_id} has no area")

    data_array = xr.DataArray(
        data=streamflow_data,
        dims=["time", "HUC10"],  # Explicitly naming the dimensions
        coords={"time": date_range, "HUC10": huc_10_list},  # Adding coordinates
    )
    xr_dataset = xr.Dataset(
        data_vars={"streamflow": data_array},
        attrs={"description": "Streamflow -> HUC Predictions"},
    )
    streamflow_path = Path(cfg.create_streamflow.data_store)
    xr_dataset.to_zarr(streamflow_path, mode="w")
    # zarr_hierarchy = zarr.open_group(streamflow_path, mode="r")


def separate_basins(cfg: DictConfig) -> None:
    """Code provided by Yalan Song to separate the basin predictions into individual files

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    """
    warnings.warn("This function is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)
    qr_folder = Path(cfg.create_streamflow.predictions)
    data_split_folder = qr_folder / "basin_split/"
    if data_split_folder.exists() is False:
        data_split_folder.mkdir(parents=True, exist_ok=True)
        attrs_df = pd.read_csv(Path(cfg.create_streamflow.obs_attributes))
        basin_ids = attrs_df.gage_ID.values
        batch_size = 1000
        start_idx = np.arange(0, len(basin_ids), batch_size)
        end_idx = np.append(start_idx[1:], len(basin_ids))
        for idx in trange(len(start_idx), desc="reading files"):
            try:
                basin_ids_np = pd.read_csv(
                    qr_folder / f"Qr_{start_idx[idx]}_{end_idx[idx]}",
                    dtype=np.float32,
                    header=None,
                ).values
            except FileNotFoundError:
                basin_ids_np = pd.read_csv(
                    qr_folder / f"out0_{start_idx[idx]}_{end_idx[idx]}",
                    dtype=np.float32,
                    header=None,
                ).values
            attribute_batch_df = pd.read_csv(
                qr_folder / "attributes" / f"attributes_{start_idx[idx]}_{end_idx[idx]}.csv"
            )
            attribute_batch_ids = attribute_batch_df.gage_ID.values
            for idx, _id in enumerate(
                tqdm(
                    attribute_batch_ids,
                    desc="saving predictions separately",
                    ncols=140,
                    ascii=True,
                )
            ):
                formatted_id = str(int(_id)).zfill(10)
                qr = basin_ids_np[idx : idx + 1, :]
                np.save(data_split_folder / f"{formatted_id}.npy", qr)


def calculate_merit_flow(cfg: DictConfig, edges: zarr.Group) -> None:
    """Calculates the flow for the MERIT dataset, converts to m3/s, and saves it to zarr

    Parameters
    ----------
    cfg : DictConfig
        _description_
    edges : zarr.Group
        _description_

    Raises
    ------
    KeyError
        Cannot find the specified COMID within the defined Areas
    """
    attrs_df = pd.read_csv(Path(cfg.create_streamflow.obs_attributes) / f"COMID_{str(cfg.zone)[0]}.csv")
    id_to_area = attrs_df.set_index("COMID")["unitarea"].to_dict()

    edge_comids = np.unique(edges.merit_basin[:])  # type: ignore # already sorted

    streamflow_predictions_root: zarr.array = zarr.open(Path(cfg.create_streamflow.predictions).__str__(), mode="r") # type: ignore

    # Different merit forwards have different save outputs. Specifying here to handle the different versions
    version = int(cfg.create_streamflow.version.lower().split("_v")[1][0])  # getting the version number
    if version >= 3:
        log.info(msg="Reading Zarr Store")
        zone_keys = [key for key in streamflow_predictions_root.keys() if str(cfg.zone) in key]
        zone_comids = []
        zone_runoff = []
        for key in zone_keys:
            zone_comids.append(streamflow_predictions_root[key].COMID[:])
            zone_runoff.append(streamflow_predictions_root[key].Qr[:])
        streamflow_comids = np.concatenate(zone_comids).astype(int)
        file_runoff = np.transpose(np.concatenate(zone_runoff))
        del zone_comids
        del zone_runoff

    else:
        log.info("Reading Zarr Store")
        file_runoff = np.transpose(streamflow_predictions_root.Runoff) # type: ignore

        streamflow_comids: np.ndarray = streamflow_predictions_root.COMID[:].astype(int) # type: ignore

    log.info("Mapping predictions to zone COMIDs")
    runoff_full_zone = np.zeros((file_runoff.shape[0], edge_comids.shape[0]))
    indices = np.searchsorted(edge_comids, streamflow_comids)
    runoff_full_zone[:, indices] = file_runoff

    log.info("Creating areas areas_array")
    areas = np.zeros_like(edge_comids, dtype=np.float64)
    for idx, comid in enumerate(edge_comids):
        try:
            areas[idx] = id_to_area[comid]
        except KeyError as e:
            log.error(f"problem finding {comid} in Areas Dictionary")
            raise e
    areas_array = areas * 1000 / 86400

    log.info("Converting runoff data, setting NaN and 0 to 1e-6")
    streamflow_m3_s_data = runoff_full_zone * areas_array
    streamflow_m3_s_data = np.nan_to_num(streamflow_m3_s_data, nan=1e-6, posinf=1e-6, neginf=1e-6)
    mask = streamflow_m3_s_data == 0
    streamflow_m3_s_data[mask] = 1e-6

    date_range = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="D",
    )
    data_array = xr.DataArray(
        data=streamflow_m3_s_data,
        dims=["time", "COMID"],  # Explicitly naming the dimensions
        coords={"time": date_range, "COMID": edge_comids},  # Adding coordinates
    )
    xr_dataset = xr.Dataset(
        data_vars={"streamflow": data_array},
        attrs={"description": "Streamflow -> MERIT Predictions"},
    )
    streamflow_path = Path(cfg.create_streamflow.data_store)
    xr_dataset.to_zarr(streamflow_path, mode="w")
    # zarr_hierarchy = zarr.open_group(streamflow_path, mode="r")
