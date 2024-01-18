import logging
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig
import pandas as pd
import zarr

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
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
        raise ImportError("Hydrofabric functionality not yet supported")
    elif cfg.name.lower() == "merit":
        from marquette.merit.create import create_edges, create_TMs, write_streamflow
    start = time.perf_counter()
    log.info(f"Creating MERIT {cfg.continent}{cfg.area} River Graph")
    edges = create_edges(cfg)
    log.info(f"Mapping MERIT {cfg.continent}{cfg.area} Streamflow to Nodes/Edges")
    create_TMs(cfg, edges)
    log.info(f"Converting Streamflow to zarr")
    write_streamflow(cfg)
    end = time.perf_counter()
    log.info(f"Extracting data took : {(end - start):.6f} seconds")


def _missing_files(cfg: DictConfig) -> bool:
    import pandas as pd

    mapped_files_dir = Path(cfg.csv.mapped_streamflow_dir)
    try:
        file_count = sum(1 for item in mapped_files_dir.iterdir() if item.is_file())
        start_date = pd.Timestamp(cfg.start_date)
        end_date = pd.Timestamp(cfg.end_date)
        num_years = (
            end_date.year - start_date.year
        ) + 1  # need to include the first value
        return file_count != num_years
    except FileNotFoundError:
        return True  # No predictions. Return True


if __name__ == "__main__":
    main()
