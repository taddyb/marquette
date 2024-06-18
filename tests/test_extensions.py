import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def test_graph(sample_gage_cfg: DictConfig, q_prime_data: np.ndarray) -> None:
    """Testing if the river graph created by extensions.calculate_q_prime_summation is correct for a specific case
    Gauge: 01563500
    Correct Number of 2km edges within the river graph: 473
    
    Parameters:
    ----------
    sample_gage_cfg: DictConfig
        The configuration object.
        
    q_prime_data: np.ndarray 
        The q_prime data for the specific case.
    """
    df_path = Path(f"{sample_gage_cfg.create_edges.edges}").parent / f"{sample_gage_cfg.zone}_graph_df.csv"
    if not df_path.exists():
        pytest.skip(
            f"Skipping graph test as this code has yet to be run. Please run the code to generate the graph."
        )
    df = pd.read_csv(df_path)
    G = nx.from_pandas_edgelist(df=df, create_using=nx.DiGraph(),)
    ancestors = nx.ancestors(G, source=89905)
    ancestors.add(89905)
    assert len(ancestors) == q_prime_data.shape[1], "There are an incorrect number of edges in your river graph"
    
    
# def test_q_prime(sample_gage_cfg: DictConfig, q_prime_data: np.ndarray) -> None:
#     """Testing if the q_prime data created by extensions.calculate_q_prime_summation is correct for a specific case
#     Gauge: 01563500
#     Time: 1987/05/19 - 1988/05/18
    
#     Parameters:
#     ----------
#     sample_gage_cfg: DictConfig
#         The configuration object.
        
#     q_prime_data: np.ndarray 
#         The q_prime data for the specific case.
#     """
#     root = zarr.open(sample_gage_cfg.create_edges.edges, mode="r")
#     zone_root = root[sample_gage_cfg.zone.__str__()]
#     try:
#         # Dividing the Summed_q_prime data by the number of COMIDs in that edge
#         summed_q_prime_data : np.ndarray = zone_root.summed_q_prime[2695:3060, 6742] / 4  # type: ignore 
#         correct_q_prime_data = np.sum(q_prime_data, axis=1)
#         assert np.allclose(summed_q_prime_data, correct_q_prime_data, atol=1e-6)
#     except AttributeError:
#         pytest.skip(
#             f"Skipping Q_prime test as this code has yet to be run. Please run the code to generate the graph."
#         )
