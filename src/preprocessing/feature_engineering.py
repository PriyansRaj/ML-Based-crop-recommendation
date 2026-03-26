import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.file_handler import open_csv


@dataclass
class FeatureEngineeringConfig:
    raw_data_path: str = os.path.join("data", "raw", "Crop_recommendation.csv")
    processed_data_path: str = os.path.join("data", "processed", "crop_processed.csv")


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.config = config

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        logging.info("Applying log transform to rainfall column")
        df["rainfall_log"] = np.log1p(df["rainfall"])

        logging.info("Creating N, P, K ratio features")
        df["N_P_ratio"] = df["N"] / (df["P"] + 1)
        df["N_K_ratio"] = df["N"] / (df["K"] + 1)
        df["P_K_ratio"] = df["P"] / (df["K"] + 1)

        logging.info("Calculating total nutrients")
        df["total_nutrients"] = df["N"] + df["P"] + df["K"]

        logging.info("Calculating climate index")
        df["climate_index"] = (
            df["temperature"] * df["humidity"]
        ) / (df["rainfall"] + 1)

        logging.info("Creating interaction feature: temp_humidity")
        df["temp_humidity"] = df["temperature"] * df["humidity"]

        logging.info("Creating interaction feature: rain_temp")
        df["rain_temp"] = df["rainfall"] * df["temperature"]

        return df

    def transform(self):
        try:
            logging.info("Starting feature engineering for training data")

            df = open_csv(self.config.raw_data_path)
            logging.info("Raw dataset loaded successfully")

            df = self._apply_feature_engineering(df)

            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            df.to_csv(self.config.processed_data_path, index=False)

            logging.info(f"Processed data saved at: {self.config.processed_data_path}")
            return self.config.processed_data_path

        except Exception as e:
            logging.error("Error occurred during feature engineering")
            raise CustomException(e, sys)

    def transform_new_data(self, new_data):
    
        try:
            logging.info("Starting feature engineering for new data")

            if isinstance(new_data, dict):
                df = pd.DataFrame([new_data])
            elif isinstance(new_data, pd.DataFrame):
                df = new_data.copy()
            else:
                raise ValueError("new_data must be a dict or pandas DataFrame")

            df = self._apply_feature_engineering(df)

            logging.info("Feature engineering for new data completed successfully")
            return df

        except Exception as e:
            logging.error("Error occurred during feature engineering for new data")
            raise CustomException(e, sys)