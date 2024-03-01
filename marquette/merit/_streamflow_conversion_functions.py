import logging
from pathlib import Path

from omegaconf import DictConfig
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import xarray as xr
import zarr


log = logging.getLogger(__name__)


def calculate_huc10_flow_from_individual_files(cfg: DictConfig) -> None:
    qr_folder = Path(cfg.create_streamflow.predictions)
    streamflow_files_path = qr_folder / "basin_split/"

    attrs_df = pd.read_csv(cfg.create_streamflow.obs_attributes)
    attrs_df["gage_ID"] = (
        attrs_df["gage_ID"].astype(str).str.zfill(10)
    )  # Left padding a 0 to make sure that all gages can be read
    id_to_area = attrs_df.set_index("gage_ID")["area"].to_dict()

    huc_to_merit_TM = zarr.open(Path(cfg.create_TMs.HUC.TM), mode="r")
    huc_10_list = huc_to_merit_TM.HUC10[:]
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
            area = id_to_area.get(
                file_id
            )  # defaulting to mean area if there is no area for the HUC10
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
    """
    Code provided by Yalan Song
    :param cfg:
    :return:
    """
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
                qr_folder
                / "attributes"
                / f"attributes_{start_idx[idx]}_{end_idx[idx]}.csv"
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


def calculate_merit_flow(cfg: DictConfig) -> None:
    attrs_df = pd.read_csv(
        Path(cfg.create_streamflow.obs_attributes) / f"COMID_{str(cfg.zone)[0]}.csv"
    )
    id_to_area = attrs_df.set_index("COMID")["unitarea"].to_dict()

    streamflow_predictions_root = zarr.open(
        Path(cfg.create_streamflow.predictions), mode="r"
    )
    log.info("Reading Zarr Store")
    runoff = np.transpose(streamflow_predictions_root.Qr[:])
    # runoff = streamflow_predictions_root.Runoff[:]

    log.info("Creating areas areas_array")
    comids = streamflow_predictions_root.COMID[:]
    # comids = streamflow_predictions_root.rivid[:]
    areas = np.zeros_like(comids, dtype=np.float64)
    for idx, comid in enumerate(comids):
        try:
            areas[idx] = id_to_area[comid]
        except KeyError as e:
            msg = f"problem finding {comid} in Areas Dictionary"
            log.exception(msg=msg)
            raise KeyError(msg)
    areas_array = areas * 1000 / 86400

    log.info("Converting runoff data")
    streamflow_m3_s_data = runoff * areas_array
    streamflow_m3_s_data = np.nan_to_num(
        streamflow_m3_s_data, nan=1e-6, posinf=1e-6, neginf=1e-6
    )

    date_range = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="D",
    )
    data_array = xr.DataArray(
        data=streamflow_m3_s_data,
        dims=["time", "COMID"],  # Explicitly naming the dimensions
        coords={"time": date_range, "COMID": comids},  # Adding coordinates
    )
    xr_dataset = xr.Dataset(
        data_vars={"streamflow": data_array},
        attrs={"description": "Streamflow -> MERIT Predictions"},
    )
    streamflow_path = Path(cfg.create_streamflow.data_store)
    xr_dataset.to_zarr(streamflow_path, mode="w")
    # zarr_hierarchy = zarr.open_group(streamflow_path, mode="r")
