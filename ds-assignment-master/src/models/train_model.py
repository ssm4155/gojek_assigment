import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier,Pool
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    class_counts = np.bincount(df_train[config["target"]])
    total_samples = len(df_train)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    class_weights[1]=class_weights[1]*1.5

    rf_estimator = CatBoostClassifier(**config["catboost"],class_weights=class_weights)
    model = SklearnClassifier(rf_estimator, config["catboost_features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
