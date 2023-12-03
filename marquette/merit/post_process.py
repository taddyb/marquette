import logging
from pathlib import Path
import time

import gzip
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


# @hydra.main(
#     version_base=None,
#     config_path="../conf/",
#     config_name="config",
# )
def post_process(cfg: DictConfig) -> None:
    """
    Main function for running experiments.

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    folder = Path(cfg.csv.mapped_streamflow_dir)
    file_name = f"*_{cfg.basin}_mapped_streamflow.csv.gz"
    file_paths = [file for file in folder.glob(file_name) if file.is_file()]
    file_paths.sort()

    for i in tqdm(
        range(0, len(file_paths) - 1), desc=f"interpolating missing {cfg.basin} data"
    ):
        try:
            _process_files(file_paths[i], file_paths[i + 1])
        except gzip.BadGzipFile:
            log.info(f"Bad GZIP file. Skipping the file: {file_paths[i]}")


def _fix_date_format(date_str):
    if len(date_str) == 10:  # Only date, no time
        return date_str + " 00:00:00"
    return date_str


def _process_files(file1, file2):
    df1 = pd.read_csv(file1, compression="gzip")
    if df1.shape[0] != 8760:
        df2 = pd.read_csv(file2, compression="gzip", nrows=1)
        df_combined = pd.concat([df1, df2], ignore_index=True)

        df_combined["dates"] = df_combined["dates"].apply(_fix_date_format)
        df_combined["dates"] = pd.to_datetime(df_combined["dates"])
        df_combined.set_index("dates", inplace=True)
        df_combined = df_combined.resample("H").asfreq()
        df_combined = df_combined.interpolate(method="linear")
        df_reset = df_combined.reset_index()[:-1]

        df_reset.to_csv(file1, index=False, compression="gzip")


if __name__ == "__main__":
    post_process()
