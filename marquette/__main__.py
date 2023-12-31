import logging
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig
import pandas as pd

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """
    Main function for running experiments.

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    if cfg.name.lower() == "hydrofabric":
        extract_hydrofabric(cfg)
    elif cfg.name.lower() == "merit":
        extract_merit(cfg)


def extract_hydrofabric(cfg: DictConfig) -> None:
    log.error("NOT SUPPORTED AT THIS MOMENT")


def _missing_files(cfg: DictConfig) -> bool:
    import pandas as pd
    mapped_files_dir = Path(cfg.csv.mapped_streamflow_dir)
    try:
        file_count = sum(1 for item in mapped_files_dir.iterdir() if item.is_file())
        start_date = pd.Timestamp(cfg.start_date)
        end_date = pd.Timestamp(cfg.end_date)
        num_years = (end_date.year - start_date.year) + 1  # need to include the first value
        return file_count != num_years
    except FileNotFoundError:
        return True  # No predictions. Return True


def extract_merit(cfg: DictConfig) -> None:
    from marquette.merit.map import create_graph, map_streamflow_to_river_graph
    from marquette.merit.post_process import post_process

    start = time.perf_counter()
    edges_file = Path(cfg.csv.edges)
    if edges_file.exists():
        edges = pd.read_csv(edges_file, compression="gzip")
    else:
        log.info(f"Creating {cfg.basin} River Graph")
        edges = create_graph(cfg)
    log.info(f"Mapping {cfg.basin} Streamflow to Nodes/Edges")
    if _missing_files(cfg):
        map_streamflow_to_river_graph(cfg, edges)
    log.info(f"Running post-processing")
    post_process(cfg)
    end = time.perf_counter()
    log.info(f"Extracting data took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
