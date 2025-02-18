import numpy as np
import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df

def total_distance(df: pd.DataFrame) -> pd.DataFrame:
    df["total_distance"]=df["driver_distance"]+df["trip_distance"]
    return df

def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame,participant_past_perf: pd.DataFrame, participant_distance_preference: pd.DataFrame) -> pd.DataFrame:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'],errors='coerce', utc=True)
    participant_past_perf['event_timestamp'] = pd.to_datetime(participant_past_perf['event_timestamp'],errors='coerce', utc=True)
    participant_distance_preference['event_timestamp'] = pd.to_datetime(participant_distance_preference['event_timestamp'],errors='coerce', utc=True)
    df=df.sort_values('event_timestamp').reset_index(drop=True)
    participant_past_perf=participant_past_perf.sort_values('event_timestamp').reset_index(drop=True)
    participant_distance_preference=participant_distance_preference.sort_values('event_timestamp').reset_index(drop=True)
    df=pd.merge_asof(df, participant_past_perf, 
                          on="event_timestamp", 
                          by="driver_id",
                          direction="backward")

    
    df=pd.merge_asof(df, participant_distance_preference, 
                          on="event_timestamp", 
                          by="driver_id",
                          direction="backward")


    df=df.fillna(0)

    return df