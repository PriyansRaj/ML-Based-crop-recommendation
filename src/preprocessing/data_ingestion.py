import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.file_handler import open_csv


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "processed", "crop_processed.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")
    prediction_data_path: str = os.path.join("data", "prediction", "new_data.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def ingest_training_data(self):
        try:
            logging.info("Starting training data ingestion")

            df = open_csv(self.config.raw_data_path)
            logging.info("Raw dataset loaded successfully")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            logging.info("Splitting the dataset into train and test dataset")
            train_data, test_data = train_test_split(
                df,
                test_size=0.3,
                random_state=42
            )

            train_data.to_csv(self.config.train_data_path, index=False)
            logging.info(f"Train data saved successfully at {self.config.train_data_path}")

            test_data.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Test data saved successfully at {self.config.test_data_path}")

            return train_data, test_data

        except Exception as e:
            logging.error("Error during training data ingestion")
            raise CustomException(e, sys)

    def ingest_new_data(self, new_data=None):
    
        try:
            logging.info("Starting new data ingestion")

            if new_data is None:
                logging.info(f"Loading new data from {self.config.prediction_data_path}")
                df = open_csv(self.config.prediction_data_path)

            elif isinstance(new_data, dict):
                logging.info("Converting dictionary input to DataFrame")
                df = pd.DataFrame([new_data])

            elif isinstance(new_data, pd.DataFrame):
                logging.info("New data received as DataFrame")
                df = new_data.copy()

            elif isinstance(new_data, str):
                logging.info(f"Loading new data from provided path: {new_data}")
                df = open_csv(new_data)

            else:
                raise ValueError(
                    "new_data must be None, dict, pandas DataFrame, or CSV file path"
                )

            logging.info("New data ingestion completed successfully")
            return df

        except Exception as e:
            logging.error("Error during new data ingestion")
            raise CustomException(e, sys)