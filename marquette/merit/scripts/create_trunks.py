"""Takes the indices from subzones, and creates saved coo pairs based on the missing idx"""
from pathlib import Path

import cupy as cp
import numpy as np
import zarr

def _find_upstream_mask(full_pairs: np.ndarray, sub_pairs: np.ndarray) -> np.ndarray:
    """Finds the indices where an upstream connection is in the full-zone connections

    Notes
    -----
    Below are the steps that are used in this algorithm
    1. Create broadcasted comparison
    2. Handle NaN equality separately
    3. Combine regular equality and NaN equality
    4. Check if both elements in each pair match (axis=2)
    5. Check if any pair in sub_pairs matches (axis=1)

    Parameters
    ----------
    full_pairs : np.ndarray
        The pairs intersections [to, from] for the full zone
    sub_pairs : np.ndarray
        The pairs intersections [to, from] for the sub zone we're masking out

    Returns
    -------
    np.ndarray
        The mask of the upstream connections
    """
    regular_equal = full_pairs[:, None] == sub_pairs
    nan_equal = np.isnan(full_pairs)[:, None] & np.isnan(sub_pairs)
    equal_or_both_nan = regular_equal | nan_equal
    pairs_match = np.all(equal_or_both_nan, axis=2)
    mask = pairs_match.any(axis=1)
    return mask

def remove_dup_nan_pairs(
    all_subzone_pairs: cp.ndarray, 
    full_zone_pairs: cp.ndarray,
) -> cp.ndarray:
    nan_mask = cp.isnan(full_zone_pairs[:, 1])
    all_subzone_pairs_nan_mask = cp.isnan(all_subzone_pairs[:, 1])
    full_pairs_row = full_zone_pairs[nan_mask][:, 0]
    sub_pairs_row = all_subzone_pairs[all_subzone_pairs_nan_mask][:, 0]
    matches = cp.isin(full_pairs_row, sub_pairs_row)
    nan_intersection_mask = cp.argwhere(matches,).squeeze()
    filtered_nan_mask = cp.zeros_like(nan_mask, dtype=bool)
    filtered_nan_mask[nan_intersection_mask] = True
    return filtered_nan_mask

def remove_intersection_pairs(
    all_subzone_pairs: cp.ndarray, 
    filtered_full_zone_pairs: cp.ndarray,
) -> cp.ndarray:
    nan_mask = cp.isnan(filtered_full_zone_pairs[:, 1])
    all_subzone_pairs_nan_mask = cp.isnan(all_subzone_pairs[:, 1])
    subzone_no_nans = all_subzone_pairs[~all_subzone_pairs_nan_mask][:, None, :]
    full_no_nans = filtered_full_zone_pairs[~nan_mask][:, None, :]
    _mask = cp.isin(full_no_nans, subzone_no_nans).squeeze()
    mask = _mask.sum(axis=1) == 2  # Both elements in the pair match
    
    nan_indices = cp.where(~nan_mask)[0]
    filtered_mask = cp.zeros_like(nan_mask, dtype=bool)
    intersected_idx = nan_indices[mask]
    filtered_mask[intersected_idx] = True
    return filtered_mask

def create_trunks(zone: str, coo_path: zarr.Group, subzones: list[str]) -> np.ndarray:
    all_pairs = []
    full_zone_pairs = cp.array(coo_path["full_zone"].pairs[:])
    for _subzone in subzones:
        all_pairs.append(cp.array(coo_path[_subzone].pairs[:]))
    all_subzone_pairs = cp.concatenate(all_pairs)
    filtered_nan_mask = remove_dup_nan_pairs(
        all_subzone_pairs, 
        full_zone_pairs,
    )
    filtered_full_zone_pairs = full_zone_pairs[~filtered_nan_mask]
    filtered_mask = remove_intersection_pairs(
        all_subzone_pairs, 
        filtered_full_zone_pairs,
    )
    trunk_full_zone_pairs = full_zone_pairs[~filtered_mask]
    return trunk_full_zone_pairs


if __name__ == "__main__":
    zone = "74"
    coo_path = zarr.open_group(Path("/projects/mhpi/data/MERIT/zarr/gage_coo_indices") / zone)
    subzones = [
        # "arkansas",
        # "missouri",
        # "ohio",
        "tennessee",
        # "upper_mississippi",
    ]
    save_name = "main_mississippi_trunk"
    pairs = create_trunks(zone, coo_path, subzones)
    root = zarr.group(Path("/projects/mhpi/data/MERIT/zarr/gage_coo_indices") / zone / save_name)
    root.create_dataset(
        "pairs", data=pairs, chunks=(5000, 5000), dtype="float32"
    )
    print(f"Saved: {save_name}")

