import sys

from src.utils.logger import logging
from src.utils.exception import CustomException

from src.preprocessing.feature_engineering import (
    FeatureEngineering,
    FeatureEngineeringConfig
)
from src.preprocessing.data_ingestion import (
    DataIngestion,
    DataIngestionConfig
)
from src.preprocessing.data_transformation import (
    DataTransform,
    DataTransformConfig
)


class PreprocessingPipeline:
    def __init__(self):
        self.fe_config = FeatureEngineeringConfig()
        self.di_config = DataIngestionConfig()
        self.dt_config = DataTransformConfig()

        self.fe = FeatureEngineering(self.fe_config)
        self.di = DataIngestion(self.di_config)
        self.dt = DataTransform(self.dt_config)

    def run_pipeline(self):
        try:
            logging.info("Starting preprocessing pipeline for training")

            logging.info("Running feature engineering step")
            processed_data_path = self.fe.transform()
            logging.info(f"Feature engineering completed: {processed_data_path}")

            logging.info("Running data ingestion step")
            train_path, test_path = self.di.ingest_training_data()
            logging.info(f"Data ingestion completed: train={train_path}, test={test_path}")

            logging.info("Running data transformation step")
            transformed_train, transformed_test, label_encoder_path, scaler_path = self.dt.transform()
            logging.info("Data transformation completed successfully")

            logging.info("Preprocessing pipeline completed successfully")

            return {
                "processed_data_path": processed_data_path,
                "train_data_path": train_path,
                "test_data_path": test_path,
                "transformed_train_path": transformed_train,
                "transformed_test_path": transformed_test,
                "label_encoder_path": label_encoder_path,
                "scaler_path": scaler_path
            }

        except Exception as e:
            logging.error("Error in preprocessing pipeline")
            raise CustomException(e, sys)

    def prepare_new_data(self, new_data):
        try:
            logging.info("Preparing new data for prediction")

            logging.info("Running new data ingestion")
            df = self.di.ingest_new_data(new_data)

            logging.info("Running feature engineering on new data")
            df_engineered = self.fe.transform_new_data(df)

            logging.info("Running transformation on new data")
            df_transformed = self.dt.transform_new_data(df_engineered)

            logging.info("New data prepared successfully")
            return df_transformed

        except Exception as e:
            logging.error("Error while preparing new data")
            raise CustomException(e, sys)