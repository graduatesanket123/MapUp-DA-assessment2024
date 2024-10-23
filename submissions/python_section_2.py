import pandas as pd
import numpy as np
from datetime import time, timedelta

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    ids = sorted(pd.unique(df[['id_start', 'id_end']].values.ravel('K')))
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)

    # Fill in the known distances
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Set diagonal to zero
    np.fill_diagonal(distance_matrix.values, 0)

    # Compute cumulative distances using Floyd-Warshall algorithm
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                unrolled.append({'id_start': id_start, 'id_end': id_end, 'distance': df.at[id_start, id_end]})

    return pd.DataFrame(unrolled)

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    ids_within_threshold = df[(df['id_start'] != reference_id) & 
                               (df['distance'] >= lower_bound) & 
                               (df['distance'] <= upper_bound)]['id_start'].unique()

    return pd.DataFrame({'ids': ids_within_threshold})

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    toll_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in toll_rates.items():
        df[vehicle] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_intervals = [
        (time(0, 0), time(10, 0), 0.8),  # Weekdays: 00:00 to 10:00
        (time(10, 0), time(18, 0), 1.2),  # Weekdays: 10:00 to 18:00
        (time(18, 0), time(23, 59), 0.8),  # Weekdays: 18:00 to 23:59
    ]

    results = []
    
    for id_start, id_end in zip(df['id_start'], df['id_end']):
        for day in days:
            for start_time, end_time, discount in time_intervals:
                for hour in range(24):
                    start_dt = time(hour, 0)
                    end_dt = time(hour + 1, 0)
                    if start_time <= start_dt < end_time:
                        rates = {
                            'id_start': id_start,
                            'id_end': id_end,
                            'start_day': day,
                            'start_time': start_dt,
                            'end_day': day,
                            'end_time': end_dt,
                        }
                        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                            rates[vehicle] = df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), vehicle].values[0] * discount
                        results.append(rates)

    return pd.DataFrame(results)
