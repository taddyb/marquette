import logging

import geopandas as gpd
import torch
import numpy as np
import pandas as pd

from marquette._graph import data_to_csv, get_edge_counts, segments_to_edges, Segment

log = logging.getLogger(__name__)


def create_graph(cfg):
    flowline_file = cfg.save_paths.flow_lines
    polyline_gdf = gpd.read_file(flowline_file)

    # Convert multiple columns to int type
    for col in [
        "COMID",
        "NextDownID",
        "up1",
        "up2",
        "up3",
        "up4",
        "maxup",
        "order_",
    ]:
        polyline_gdf[col] = polyline_gdf[col].astype(int)

    crs = polyline_gdf.crs

    # Generate segments using list comprehension
    segments = [
        Segment(row, row.geometry, crs) for _, row in polyline_gdf.iterrows()
    ]

    dx = cfg.dx  # Unit: Meters
    buffer = cfg.buffer * dx  # Unit: Meters
    sorted_segments = sorted(
        segments, key=lambda segment: segment.uparea, reverse=False
    )
    segment_das = {
        segment.id: segment.uparea for segment in segments
    }  # Simplified with dict comprehension
    # TODO see why this is taking so long
    edge_counts = get_edge_counts(sorted_segments, dx, buffer)
    edges_ = [
        edge
        for segment in sorted_segments
        for edge in segments_to_edges(
            segment, edge_counts, segment_das
        )  # returns many edges
    ]
    edges = data_to_csv(edges_)

    # Saving data to csv
    # data_to_csv(segments).to_csv(
    #     f"{cfg.save_path}segment_info.csv",
    # )
    edges.to_csv(cfg.csv.edges, index=False)

    return edges


def create_network(cfg, edges):
    graph_map = {row[0]: i for i, row in enumerate(edges.itertuples(index=False))}
    upstream_lsts = edges["up"].to_numpy()
    max_len = max(len(lst) for lst in upstream_lsts)
    connectivity = np.array(
        [lst + [None] * (max_len - len(lst)) for lst in upstream_lsts],
        dtype=object,
    )

    network = torch.zeros(
        [connectivity.shape[0], connectivity.shape[0]], 
    )
    for i, row in enumerate(connectivity):
        values = [
            graph_map[key] if key in graph_map.keys() else None for key in row
        ]
        if any(value is not None for value in values):
            for j, value in enumerate(values):
                if value is not None:
                    network[i, value] = 1

    # Outputting results to csv for debugging. The index of 0 is the COMID field
    network_df = pd.DataFrame(
        network.cpu().numpy(),
        index=edges.iloc[:, 0].to_numpy(),
        columns=edges.iloc[:, 0].to_numpy(),
    )
    network_df.to_csv(cfg.csv.network)
    return network
