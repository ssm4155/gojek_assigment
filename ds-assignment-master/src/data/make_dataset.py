import pandas as pd

from src.utils.config import load_config
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()
    config = load_config()

    booking_df = store.get_raw("booking_log.csv")
    booking_df = clean_booking_df(booking_df)

    participant_df = store.get_raw("participant_log.csv")
    participant_df = clean_participant_df(participant_df)
    
    participant_df = create_target(participant_df, config["target"])
    dataset = merge_dataset(booking_df, participant_df)
    participant_distance_preference=distance_preference(dataset)

    participant_history = participant_past_performance(participant_df)

    store.put_processed("dataset.csv", dataset)
    store.put_processed("participant_past_performance.csv", participant_history)
    store.put_processed("participant_distance_preference.csv", participant_distance_preference)

def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_columns = [
        "order_id",
        "trip_distance",
        "pickup_latitude",
        "pickup_longitude",
    ]
    df = df.drop_duplicates(subset=unique_columns)
    return df[unique_columns]

def filter_orders(group):
    if "ACCEPTED" in group["participant_status"].values:
        return group[group["participant_status"] == "ACCEPTED"]
    else:
        return group[group["participant_status"] == "CREATED"]

def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.groupby(["experiment_key","driver_id"]).apply(filter_orders).reset_index(drop=True)
    return df


def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(participants, bookings, on="order_id", how="left")
    return df

def distance_preference(df: pd.DataFrame) -> pd.DataFrame:
    df=df[df["is_completed"]==1]
    df=df.sort_values(["driver_id","event_timestamp"]).reset_index(drop=True)
    df["last_5_ride_avg_distance"] = df.groupby("driver_id",group_keys=False)["trip_distance"].apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)
    return df[["event_timestamp","driver_id","last_5_ride_avg_distance"]]

def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = df["participant_status"].apply(lambda x: int(x == "ACCEPTED"))
    return df

def participant_past_performance(df: pd.DataFrame) -> pd.DataFrame:
    df=df.sort_values(by=["driver_id","event_timestamp"])
    df["complete_track_record"] = df.groupby("driver_id",group_keys=False)["is_completed"].apply(lambda x: x.shift(1).expanding().mean()).fillna(0)
    df["n_requests"]=df.groupby("driver_id",group_keys=False)["is_completed"].apply(lambda x: x.shift(1).expanding().count()).fillna(0)
    df["last_10_track_record"] = df.groupby("driver_id",group_keys=False)["is_completed"].apply(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()).fillna(0)
    df["last_5_track_record"] = df.groupby("driver_id",group_keys=False)["is_completed"].apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)
    return df[["event_timestamp","driver_id","complete_track_record","n_requests","last_10_track_record","last_5_track_record"]]


if __name__ == "__main__":
    main()
