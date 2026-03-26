import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.file_handler import open_csv, save_model, load_model


@dataclass
class DataTransformConfig:
    raw_train_data: str = os.path.join("data", "processed", "train.csv")
    raw_test_data: str = os.path.join("data", "processed", "test.csv")
    transformed_train: str = os.path.join("data", "artifacts", "train_transformed.csv")
    transformed_test: str = os.path.join("data", "artifacts", "test_transformed.csv")
    label_encoder_path: str = os.path.join("models", "data_transform_model", "label_encoder.pkl")
    standard_scaler_path: str = os.path.join("models", "data_transform_model", "standard_scaler.pkl")


class DataTransform:
    def __init__(self, config: DataTransformConfig = DataTransformConfig()):
        self.config = config

    def transform(self):
        try:
            logging.info("Opening training and testing data")
            train_data = open_csv(self.config.raw_train_data)
            test_data = open_csv(self.config.raw_test_data)
            logging.info("Train and test data opened successfully")

            X_train = train_data.drop("label", axis=1)
            y_train = train_data["label"]

            X_test = test_data.drop("label", axis=1)
            y_test = test_data["label"]

            num_cols = X_train.columns.tolist()

            logging.info("Creating StandardScaler")
            sc = StandardScaler()

            logging.info("Scaling training data")
            X_train_transformed = sc.fit_transform(X_train)

            logging.info("Scaling testing data")
            X_test_transformed = sc.transform(X_test)

            logging.info("Encoding target labels")
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            logging.info("Converting transformed arrays into DataFrames")
            X_train_df = pd.DataFrame(X_train_transformed, columns=num_cols)
            X_test_df = pd.DataFrame(X_test_transformed, columns=num_cols)

            X_train_df["label"] = y_train_encoded
            X_test_df["label"] = y_test_encoded

            os.makedirs(os.path.dirname(self.config.transformed_train), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.label_encoder_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.standard_scaler_path), exist_ok=True)

            logging.info("Saving transformed train and test datasets")
            X_train_df.to_csv(self.config.transformed_train, index=False)
            X_test_df.to_csv(self.config.transformed_test, index=False)
            logging.info("Transformed datasets saved successfully")

            logging.info("Saving scaler and label encoder objects")
            save_model(self.config.label_encoder_path, le)
            save_model(self.config.standard_scaler_path, sc)
            logging.info("Scaler and encoder saved successfully")

            return (
                self.config.transformed_train,
                self.config.transformed_test,
                self.config.label_encoder_path,
                self.config.standard_scaler_path
            )

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)

    def transform_new_data(self, new_data):
        """
        Transforms new data using saved scaler.
        Expects:
        - pandas DataFrame
        - WITHOUT target column 'label'
        """
        try:
            logging.info("Starting transformation for new data")

            if not isinstance(new_data, pd.DataFrame):
                raise ValueError("new_data must be a pandas DataFrame")

            df = new_data.copy()

            if "label" in df.columns:
                df = df.drop("label", axis=1)

            logging.info("Loading saved scaler")
            sc = load_model(self.config.standard_scaler_path)

            logging.info("Transforming new data using saved scaler")
            transformed_array = sc.transform(df)

            transformed_df = pd.DataFrame(transformed_array, columns=df.columns)

            logging.info("New data transformation completed successfully")
            return transformed_df

        except Exception as e:
            logging.error("Error during transformation of new data")
            raise CustomException(e, sys)

    def decode_prediction(self, prediction):
        try:
            logging.info("Loading label encoder for decoding prediction")
            le = load_model(self.config.label_encoder_path)

            decoded = le.inverse_transform(prediction)
            return decoded

        except Exception as e:
            logging.error("Error while decoding prediction")
            raise CustomException(e, sys)