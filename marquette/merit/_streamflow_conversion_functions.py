import logging
from pathlib import Path
import re
from typing import List, Tuple

from omegaconf import DictConfig
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import xarray as xr
import zarr

log = logging.getLogger(__name__)


def extract_numbers(filename: str) -> Tuple[int, int]:
    """
    Extracts numerical values from a filename and returns them as a tuple of integers.

    This function searches for the first occurrence of one or two groups of digits in the filename,
    separated by an underscore, and returns the extracted numbers as a tuple of integers. If the
    expected pattern is not found, it returns (0, 0). This function is typically used for sorting
    filenames based on numerical values embedded in their names.

    Parameters:
    filename (str or Path-like): The filename or path from which to extract the numbers. The
                                 filename is expected to contain numbers in the format 'xxxx_yyyy'.

    Returns:
    tuple: A tuple of two integers representing the extracted numerical values. If the pattern is
           not found, returns (0, 0).

    Example:
    --------
    >> extract_numbers("Qr_12000_13000")
    (12000, 13000)

    >> extract_numbers("file_123.txt")
    (123, 0)

    >> extract_numbers("no_numbers_here")
    (0, 0)

    Notes:
    ------
    - The function uses regular expressions to find the numbers.
    - If only one group of digits is found, the second element of the returned tuple will be 0.
    """
    match = re.search(r"(\d+)_(\d+)", str(filename))
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0)


def _sort_into_bins(ids: np.ndarray, bins: List[np.ndarray]):
    """
    Sorts a list of IDs into specified bins.

    This function takes an array of IDs and sorts them into predefined bins. Each bin is a
    range of values, and the function determines in which bin each ID belongs. The function
    returns a dictionary where each key corresponds to a bin index and the value is a list of
    dictionaries, each containing the ID and its original index in the `ids` array.

    Parameters:
    ids (np.ndarray): An array of integer IDs to be sorted into bins.
    bins (List[np.ndarray]): A list of NumPy arrays, where each array represents a bin. Each
                             bin is a sorted array of integers defining the range of IDs it
                             contains.

    Returns:
    dict: A dictionary where keys are integers representing bin indices. Each value is a list
          of dictionaries, with each dictionary containing an ID and its original index.

    Example:
    --------
    ids = np.array([101, 102, 103, 104])
    bins = [np.array([100, 101]), np.array([102, 103, 104])]
    sorted_bins = _sort_into_bins(ids, bins)
    # sorted_bins will be:
    # {0: [{101: 0}], 1: [{102: 1}, {103: 2}, {104: 3}]}

    Notes:
    ------
    - The function uses a binary search algorithm to efficiently find the appropriate bin for
      each ID.
    - If an ID does not fit into any bin, it is not included in the returned dictionary.
    """
    def find_list_of_str(target: int, sorted_lists: List[np.ndarray]):
        left, right = 0, len(sorted_lists) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_list = sorted_lists[mid]
            if mid_list.size > 0:
                first_element = int(mid_list[0])
                last_element = int(mid_list[-1])
                if target < first_element:
                    right = mid - 1
                elif target > last_element:
                    left = mid + 1
                else:
                    return mid
            else:
                left += 1
        return None

    keys = list(range(0, 16, 1))
    grouped_values = {key: [] for key in keys}
    for idx, value in enumerate(ids):
        id = int(ids[idx])
        _key = find_list_of_str(id, bins)
        try:
            grouped_values[_key].append({id: idx})
        except KeyError:
            log.debug("ignoring this key")
    return grouped_values


def calculate_from_individual_files(cfg: DictConfig,  streamflow_files_path: Path) -> None:
    attrs_df = pd.read_csv(cfg.save_paths.attributes)
    attrs_df['gage_ID'] = attrs_df['gage_ID'].astype(str).str.zfill(10)  # Left padding a 0 to make sure that all gages can be read
    id_to_area = attrs_df.set_index('gage_ID')['area'].to_dict()

    huc_to_merit_TM = zarr.open(Path(cfg.zarr.HUC_TM), mode="r")
    huc_10_list = huc_to_merit_TM.HUC10[:]
    date_range = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
    streamflow_data = np.zeros((len(date_range), len(huc_10_list)))

    for i, huc_id in enumerate(tqdm(huc_10_list, desc="Processing River flowlines")):
        try:
            file_path = streamflow_files_path / f"{huc_id}.npy"
            data = np.load(file_path)
            file_id = file_path.stem
            area = id_to_area.get(file_id)  # defaulting to mean area if there is no area for the HUC10
            data = data * area * 1000 / 86400
            streamflow_data[:, i] = data
        except FileNotFoundError:
            log.info(f"No Predictions found for {huc_id}")
        except KeyError:
            log.info(f"{huc_id} has no area")

    data_array = xr.DataArray(
        data=streamflow_data,
        dims=["time", "HUC10"],  # Explicitly naming the dimensions
        coords={"time": date_range, "HUC10": huc_10_list}  # Adding coordinates
    )
    xr_dataset = xr.Dataset(
        data_vars={"streamflow": data_array},
        attrs={"description": "Streamflow -> HUC Predictions"}
    )
    streamflow_path = Path(cfg.zarr.streamflow)
    xr_dataset.to_zarr(streamflow_path, mode='w')
    zarr_hierarchy = zarr.open_group(streamflow_path, mode='r')


def calculate_from_qr_files(cfg: DictConfig) -> None:
    attrs_df = pd.read_csv(cfg.save_paths.attributes)
    huc10_ids = attrs_df["gage_ID"].values.astype("str")
    huc_to_merit_TM = zarr.open(Path(cfg.zarr.HUC_TM), mode="r")
    huc_10_list = huc_to_merit_TM.HUC10[:]
    bins_size = 1000
    bins = [
        huc10_ids[i: i + bins_size] for i in range(0, len(huc10_ids), bins_size)
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
                _id = list(val.keys())[0]
                columns.append(_id)
                row = attrs_df[attrs_df["gage_ID"] == _id]
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
                    no_pred = np.zeros([14610])  # the len of the array
                    streamflow_data.append(no_pred)
                    continue
    array = np.array(streamflow_data).T
    column_keys = np.array(columns)
    date_range = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
    ds = xr.Dataset(
        {"streamflow": (["time", "HUC10"], array)},
        coords={"time": date_range, "HUC10": column_keys},
    )
    ds.to_zarr(Path(cfg.zarr.streamflow))


def separate_basins(cfg: DictConfig) -> Path:
    """
    Code provided by Yalan Song
    :param cfg:
    :return:
    """
    qr_folder = Path(cfg.save_paths.streamflow_files)
    data_split_folder = qr_folder / f"basin_split/"
    if data_split_folder.exists() is False:
        data_split_folder.mkdir(parents=True, exist_ok=True)
        attrs_df = pd.read_csv(Path(cfg.save_paths.attributes))
        basin_ids = attrs_df.gage_ID.values
        batch_size = 1000
        start_idx = np.arange(0, len(basin_ids), batch_size)
        end_idx = np.append(start_idx[1:], len(basin_ids))
        for idx in trange(len(start_idx), desc="reading files"):
            basin_ids_np = pd.read_csv(qr_folder / f"Qr_{start_idx[idx]}_{end_idx[idx]}", dtype=np.float32, header=None).values
            attribute_batch_df = pd.read_csv(qr_folder / "attributes" / f"attributes_{start_idx[idx]}_{end_idx[idx]}.csv")
            attribute_batch_ids = attribute_batch_df.gage_ID.values
            for idx, _id in enumerate(tqdm(attribute_batch_ids, desc="saving predictions separately")):
                formatted_id = str(int(_id)).zfill(10)
                qr = basin_ids_np[idx:idx + 1, :]
                np.save(data_split_folder / f"{formatted_id}.npy", qr)
    return data_split_folder
