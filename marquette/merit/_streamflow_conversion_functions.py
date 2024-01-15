import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def extract_numbers(filename: str) -> Tuple[int, int]:
    """
    Extracts numerical values from a filename and returns them as a tuple of integers.

    This function searches for the first occurrence of one or two groups of digits in the filename,
    separated by an underscore, and returns the extracted numbers as a tuple of integers. If the
    expected pattern is not found, it returns (0, 0). This function is typically used for sorting
    filenames based on numerical values embedded in their names.

    Parameters:
    filename (str or Path-like): The filename or path from which to extract the numbers. The
                                 filename is expected to contain numbers in the format 'xxxx_yyyy'.

    Returns:
    tuple: A tuple of two integers representing the extracted numerical values. If the pattern is
           not found, returns (0, 0).

    Example:
    --------
    >> extract_numbers("Qr_12000_13000")
    (12000, 13000)

    >> extract_numbers("file_123.txt")
    (123, 0)

    >> extract_numbers("no_numbers_here")
    (0, 0)

    Notes:
    ------
    - The function uses regular expressions to find the numbers.
    - If only one group of digits is found, the second element of the returned tuple will be 0.
    """
    match = re.search(r"(\d+)_(\d+)", str(filename))
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0)


def _sort_into_bins(ids: np.ndarray, bins: List[np.ndarray]):
    """
    Sorts a list of IDs into specified bins.

    This function takes an array of IDs and sorts them into predefined bins. Each bin is a
    range of values, and the function determines in which bin each ID belongs. The function
    returns a dictionary where each key corresponds to a bin index and the value is a list of
    dictionaries, each containing the ID and its original index in the `ids` array.

    Parameters:
    ids (np.ndarray): An array of integer IDs to be sorted into bins.
    bins (List[np.ndarray]): A list of NumPy arrays, where each array represents a bin. Each
                             bin is a sorted array of integers defining the range of IDs it
                             contains.

    Returns:
    dict: A dictionary where keys are integers representing bin indices. Each value is a list
          of dictionaries, with each dictionary containing an ID and its original index.

    Example:
    --------
    ids = np.array([101, 102, 103, 104])
    bins = [np.array([100, 101]), np.array([102, 103, 104])]
    sorted_bins = _sort_into_bins(ids, bins)
    # sorted_bins will be:
    # {0: [{101: 0}], 1: [{102: 1}, {103: 2}, {104: 3}]}

    Notes:
    ------
    - The function uses a binary search algorithm to efficiently find the appropriate bin for
      each ID.
    - If an ID does not fit into any bin, it is not included in the returned dictionary.
    """
    def find_list_of_str(target: int, sorted_lists: List[np.ndarray]):
        left, right = 0, len(sorted_lists) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_list = sorted_lists[mid]
            if mid_list.size > 0:
                first_element = int(mid_list[0])
                last_element = int(mid_list[-1])
                if target < first_element:
                    right = mid - 1
                elif target > last_element:
                    left = mid + 1
                else:
                    return mid
            else:
                left += 1
        return None

    keys = list(range(0, 16, 1))
    grouped_values = {key: [] for key in keys}
    for idx, value in enumerate(ids):
        id = int(ids[idx])
        _key = find_list_of_str(id, bins)
        grouped_values[_key].append({id: idx})

    return grouped_values


def interpolate_chunk(data_chunk, date_index):
    df = pd.DataFrame(data_chunk, index=date_index)
    df = df.resample('H').asfreq()
    return df.interpolate(method='linear').values