import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    driver_distance_to_pickup,
    total_distance,
    driver_historical_completed_bookings,
    hour_of_day
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")
    participant_past_perf = store.get_processed("participant_past_performance.csv")
    participant_distance_preference = store.get_processed("participant_distance_preference.csv")
    dataset = apply_feature_engineering(dataset,participant_past_perf,participant_distance_preference)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame,participant_past_perf: pd.DataFrame,participant_distance_preference: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(total_distance)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings,participant_past_perf,participant_distance_preference)
    )


if __name__ == "__main__":
    main()
