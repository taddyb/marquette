import logging

log = logging.getLogger(__name__)


def downstream_map(id_index, merit_flowlines, rows, cols, data, id_to_index, visited):
    """
    Perform a recursive downstream mapping starting from a given node ID and record the downstream connectivity.

    Parameters
    ----------
    id_index : int
        The index of the starting node ID for the mapping.
    merit_flowlines : zarr.core.Array
        The zarr array containing the merit flowlines data.
    rows : list
        List to store row indices for the COO matrix.
    cols : list
        List to store column indices for the COO matrix.
    data : list
        List to store data values (connectivity) for the COO matrix.
    id_to_index : dict
        Mapping from string IDs to numerical indices.
    visited : set
        A set of visited node indices.
    """
    if id_index in visited:
        return
    visited.add(id_index)

    ds_id = merit_flowlines.ds[id_index]
    if ds_id == '0_0':
        return  # Stop mapping when the river ends

    if ds_id in id_to_index:
        ds_index = id_to_index[ds_id]
        rows.append(id_index)
        cols.append(ds_index)
        data.append(1)  # Connectivity
        downstream_map(ds_index, merit_flowlines, rows, cols, data, id_to_index, visited)