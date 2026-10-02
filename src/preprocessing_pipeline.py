import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class DataFramePreprocessor:
    def __init__(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])
        self.feature_names = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names)


def create_preprocessing_pipeline(X=None) -> DataFramePreprocessor:
    return DataFramePreprocessor()
