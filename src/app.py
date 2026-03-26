import os
import sys
import numpy as np
from collections import Counter

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.file_handler import load_model,open_csv
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer, TrainerConfig
from src.preprocessing.processing import PreprocessingPipeline


class Main:
    def __init__(self):
        self.processing = PreprocessingPipeline()
        self.trainer = Trainer()
        self.trainer_config = TrainerConfig()
        self.evaluator = Evaluator()

    def _preprocessing_artifacts_exist(self):
        files_to_check = [
            self.processing.fe_config.processed_data_path,
            self.processing.di_config.train_data_path,
            self.processing.di_config.test_data_path,
            self.processing.dt_config.transformed_train,
            self.processing.dt_config.transformed_test,
            self.processing.dt_config.label_encoder_path,
            self.processing.dt_config.standard_scaler_path,
        ]

        missing_files = [file for file in files_to_check if not os.path.exists(file)]

        if missing_files:
            logging.info(f"Missing preprocessing artifacts: {missing_files}")
            return False

        logging.info("All preprocessing artifacts are available")
        return True

    def _trained_models_exist(self):
        model_dir = self.trainer_config.model_path

        if not os.path.exists(model_dir):
            logging.info("Model directory does not exist")
            return False

        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

        if not model_files:
            logging.info("No saved model files found")
            return False

        logging.info(f"Saved models found: {model_files}")
        return True

    def _load_all_models(self):
        try:
            logging.info("Loading all saved models")

            model_dir = self.trainer_config.model_path
            models = {}

            for file_name in os.listdir(model_dir):
                if file_name.endswith(".pkl"):
                    model_name = file_name.replace(".pkl", "")
                    model_path = os.path.join(model_dir, file_name)
                    models[model_name] = load_model(model_path)

            if not models:
                raise ValueError("No saved models found in model directory")

            logging.info(f"Loaded {len(models)} models successfully")
            return models

        except Exception as e:
            logging.error("Error while loading models")
            raise CustomException(e, sys)

    def _majority_vote(self, predictions):
        try:
            model_preds = list(predictions.values())
            n_samples = len(model_preds[0])

            final_predictions = []

            for i in range(n_samples):
                votes = [pred[i] for pred in model_preds]
                voted_class = Counter(votes).most_common(1)[0][0]
                final_predictions.append(voted_class)

            return np.array(final_predictions)

        except Exception as e:
            logging.error("Error during majority voting")
            raise CustomException(e, sys)

    def run(self, new_data=None, decode=True):
        try:
            logging.info("Starting main pipeline")

            if not self._preprocessing_artifacts_exist():
                logging.info("Preprocessing artifacts not complete. Running preprocessing pipeline")
                self.processing.run_pipeline()
            else:
                logging.info("Skipping preprocessing")

            if not self._trained_models_exist():
                logging.info("Trained models not found. Running model training")
                self.trainer.run()
            else:
                logging.info("Skipping model training")

            if new_data is None:
                return {
                    "message": "Pipeline is ready. Preprocessing and models already available."
                }

            logging.info("Preparing new data for prediction")
            transformed_data = self.processing.prepare_new_data(new_data)

            models = self._load_all_models()

            all_predictions = {}
            for model_name, model in models.items():
                logging.info(f"Predicting with {model_name}")
                all_predictions[model_name] = model.predict(transformed_data)

            final_prediction = self._majority_vote(all_predictions)

            if decode:
                decoded_individual_predictions = {}
                for model_name, pred in all_predictions.items():
                    decoded_individual_predictions[model_name] = (
                        self.processing.dt.decode_prediction(pred)
                    )

                final_prediction = self.processing.dt.decode_prediction(final_prediction)

                return {
                    "individual_predictions": decoded_individual_predictions,
                    "final_prediction": final_prediction,
                }

            return {
                "individual_predictions": all_predictions,
                "final_prediction": final_prediction,
            }

        except Exception as e:
            logging.error("Error in main pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    main_obj = Main()

    sample = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }

    result = main_obj.run(new_data=sample)
    print(result)