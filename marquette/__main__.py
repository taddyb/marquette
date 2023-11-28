import logging
from pathlib import Path

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
    from marquette.hydrofabric.map import create_graph, create_network
    log.info(f"Creating River Graph")
    edges = create_graph(cfg)
    log.info(f"Connecting Nodes/Edges")
    create_network(cfg, edges)
    log.info(f"Done!")


def extract_merit(cfg: DictConfig) -> None:
    from marquette.merit.map import create_graph, map_streamflow_to_river_graph
    edges_file = Path(cfg.csv.edges)
    if edges_file.exists():
        edges = pd.read_csv(edges_file)
    else:
        log.info(f"Creating River Graph")
        edges = create_graph(cfg)
    log.info(f"Mapping Streamflow to Nodes/Edges")
    map_streamflow_to_river_graph(cfg, edges)
    log.info(f"Done!")


if __name__ == "__main__":
    main()
