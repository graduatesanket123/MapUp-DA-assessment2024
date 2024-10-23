from typing import Dict, List
import pandas as pd
import re
from itertools import permutations
from datetime import datetime, timedelta
import numpy as np

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        for item in reversed(group):
            result.append(item)
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    def flatten(d, parent_key=''):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.update(flatten({f"[{i}]": item}, new_key))
            else:
                items[new_key] = v
        return items
    
    return flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    return [list(p) for p in set(permutations(nums))]

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    dates = []
    for pattern in date_patterns:
        found_dates = re.findall(pattern, text)
        dates.extend(found_dates)
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    import polyline

    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    
    # Haversine formula to calculate distance between two coordinates
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of Earth in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    for i in range(1, len(df)):
        df.at[i, 'distance'] = haversine(df.at[i-1, 'latitude'], df.at[i-1, 'longitude'],
                                          df.at[i, 'latitude'], df.at[i, 'longitude'])

    return df

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    """
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]
    
    return final_matrix

def time_check(df) -> pd.Series:
    """
    Use shared dataset-1 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period
    """
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    time_check_results = []

    for (id, id_2), group in df.groupby(['id', 'id_2']):
        time_range = pd.date_range(start=group['start'].min(), end=group['end'].max(), freq='H')
        is_full_24h = (len(time_range) >= 24)
        is_full_week = (group['start'].dt.dayofweek.nunique() == 7)

        time_check_results.append((id, id_2, not (is_full_24h and is_full_week)))

    result_df = pd.DataFrame(time_check_results, columns=['id', 'id_2', 'incorrect'])
    result_df.set_index(['id', 'id_2'], inplace=True)

    return result_df['incorrect']
