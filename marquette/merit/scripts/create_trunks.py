"""
Functions for processing and filtering zone pairs based on subzone indices.

This module provides functionality to take indices from subzones and create saved 
coordinate (coo) pairs based on the missing indices. It handles memory-efficient 
processing of large arrays using chunking and GPU acceleration via CuPy.

This function creates the pairs that are used in the Network object of dMC

Notes
-----
The main workflow consists of:
1. Loading full zone and subzone pairs
2. Finding and removing upstream connections
3. Saving the filtered pairs to a new Zarr array
"""
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from tqdm import trange

def _find_upstream_mask(full_pairs: np.ndarray, sub_pairs: np.ndarray, chunk_size=1000) -> cp.ndarray:
    """
    Find indices where upstream connections exist in the full-zone connections.

    Notes
    -----
    Algorithm steps:
    1. Create broadcasted comparison
    2. Handle NaN equality separately
    3. Combine regular equality and NaN equality
    4. Check if both elements in each pair match (axis=2)
    5. Check if any pair in sub_pairs matches (axis=1)

    Parameters
    ----------
    full_pairs : cp.ndarray
        The pairs intersections [to, from] for the full zone
    sub_pairs : cp.ndarray
        The pairs intersections [to, from] for the sub zone we're masking out
    chunk_size : int, optional
        Size of chunks to process at once, by default 1000

    Returns
    -------
    cp.ndarray
        Boolean mask indicating which pairs in full_pairs have upstream connections
    """
    n_rows = full_pairs.shape[0]
    final_mask = np.zeros(n_rows, dtype=bool)
    
    for start_idx in trange(0, n_rows, chunk_size, desc="Processing chunks for subpairs"):
        end_idx = min(start_idx + chunk_size, n_rows)
        chunk = full_pairs[start_idx:end_idx]
        
        regular_equal = chunk[:, None] == sub_pairs
        nan_equal = cp.isnan(chunk)[:, None] & cp.isnan(sub_pairs)
        equal_or_both_nan = regular_equal | nan_equal
        pairs_match = cp.all(equal_or_both_nan, axis=2)
        chunk_mask = pairs_match.any(axis=1)

        final_mask[start_idx:end_idx] = cp.asnumpy(chunk_mask)
    return final_mask


def create_trunks(coo_path: zarr.Group, subzones: list[str], gage_id: str = "full_zone") -> cp.ndarray:
    """
    Create trunk pairs by removing subzone connections from full zone pairs.

    Parameters
    ----------
    coo_path : zarr.Group
        Zarr group containing the coordinate pairs for full zone and subzones
    subzones : list[str]
        List of subzone names to process and remove from full zone
    gage_id : str, optional
        Gage ID to use for finding the larger array, by default "full_zone"

    Returns
    -------
    cp.ndarray
        Filtered pairs array with subzone connections removed

    Notes
    -----
    This function:
    1. Loads full zone pairs into GPU memory
    2. Iteratively processes each subzone
    3. Removes matching pairs from full zone
    4. Manages GPU memory by freeing unused arrays
    """
    full_zone_pairs = cp.array(coo_path[gage_id].pairs[:])
    mempool = cp.get_default_memory_pool()
    start_time = time.perf_counter()
    for _subzone in subzones:
        subzone_pairs = cp.array(coo_path[_subzone].pairs[:])
        mask = _find_upstream_mask(full_zone_pairs, subzone_pairs)
        full_zone_pairs = full_zone_pairs[~mask]
        del mask 
        del subzone_pairs
        mempool.free_all_blocks()
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    return full_zone_pairs


if __name__ == "__main__":
    """
    Main execution block for processing and saving trunk pairs.

    This script:
    1. Loads coordinate pairs from a specified zone
    2. Removes connections from specified subzones
    3. Saves the resulting filtered pairs to a new Zarr array
    """
    cp.cuda.runtime.setDevice(1)
    zone = "74"
    coo_path = zarr.open_group(Path("/projects/mhpi/data/MERIT/zarr/gage_coo_indices") / zone)
    subzones = [
        "arkansas",
        "missouri",
        "ohio",
        "tennessee",
        "upper_mississippi",
    ]
    gage_id = '4127800'
    save_name = f"{gage_id}_without_{'_'.join(sorted(subzones))}"
    pairs = create_trunks(coo_path, subzones, gage_id)
    root = zarr.group(Path("/projects/mhpi/data/MERIT/zarr/gage_coo_indices") / zone / save_name)
    root.create_dataset(
        "pairs", data=cp.asnumpy(pairs), chunks=(5000, 5000), dtype="float32"
    )
    print(f"Saved: {save_name}")
