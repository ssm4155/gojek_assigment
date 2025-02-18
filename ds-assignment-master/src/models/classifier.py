from abc import ABC, abstractmethod
from typing import Dict, List
from catboost import Pool
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
)

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        print(df_train[self.features].isna().sum())
        X_train_pooled=Pool(data=df_train[self.features], label=df_train[self.target])
        self.clf.fit(X_train_pooled)

    def evaluate(self, df_test: pd.DataFrame):
        metrics_dict={}
        df_test['preds']=self.clf.predict(df_test[self.features],prediction_type='Class')
        metrics_dict["accuracy"] = accuracy_score(df_test[self.target], df_test['preds'])
        metrics_dict["precision_weighted"] = precision_score(df_test[self.target], df_test['preds'], average="weighted")
        metrics_dict["recall_weighted"] = recall_score(df_test[self.target], df_test['preds'], average="weighted")
        metrics_dict["f1_weighted"] = f1_score(df_test[self.target], df_test['preds'], average="weighted")
        metrics_dict["roc_auc"] = roc_auc_score(df_test[self.target], df_test['preds'])
        
        return metrics_dict

    def predict(self, df: pd.DataFrame):
        print(df.shape)
        print(df.isna().sum())
        print(df[df["n_requests"].isna()].head())
        return self.clf.predict(df[self.features],prediction_type='Probability')[:, 1]
