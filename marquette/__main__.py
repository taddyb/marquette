import logging

import hydra
from omegaconf import DictConfig

from marquette.map import create_graph, create_network

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
    log.info(f"Creating River Graph")
    edges = create_graph(cfg)
    log.info(f"Connecting Noes/Edges")
    create_network(cfg, edges)
    log.info(f"Done!")


if __name__ == "__main__":
    main()
