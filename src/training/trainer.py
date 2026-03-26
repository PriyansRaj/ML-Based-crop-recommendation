import sys
import os
from dataclasses import dataclass

from src.utils.file_handler import save_model, load_model, open_csv
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.training.model_initializer import ModelInitializer
from src.preprocessing.data_transformation import DataTransformConfig
from src.training.evaluator import Evaluator


@dataclass
class TrainerConfig:
    model_path: str = os.path.join("models", "prediction_models")


class Trainer:
    def __init__(self):
        self.trainer_config = TrainerConfig()
        self.dt_config = DataTransformConfig()
        self.models_initializer = ModelInitializer()
        self.evaluator = Evaluator()

    def _load_training_data(self):
        try:
            logging.info("Loading transformed training data")

            train_df = open_csv(self.dt_config.transformed_train)

            X_train = train_df.drop("label", axis=1)
            y_train = train_df["label"]

            logging.info("Training data loaded successfully")
            return X_train, y_train

        except Exception as e:
            logging.error("Error while loading training data")
            raise CustomException(e, sys)

    def fit(self):
        logging.info("Training the models")
        try:
            os.makedirs(self.trainer_config.model_path, exist_ok=True)

            X_train, y_train = self._load_training_data()

            models = self.models_initializer.get_models()
            fit_models = {}

            for name, model in models.items():
                model_file_name = f"{name}.pkl"
                model_file_path = os.path.join(
                    self.trainer_config.model_path,
                    model_file_name
                )

                print(f"Fitting {name}...\n")
                logging.info(f"Fitting {name} model")

                model.fit(X_train, y_train)
                fit_models[name] = model

                logging.info(f"{name} training completed")
                logging.info(f"Saving {name} model")

                save_model(model_file_path, model)

            logging.info("Training completed")
            return fit_models

        except Exception as e:
            logging.error("Error while training the model")
            raise CustomException(e, sys)

    def predict(self, test):
        logging.info("Getting fitted models")
        result = {}

        try:
            # Safety: if label column is accidentally passed, remove it
            if hasattr(test, "columns") and "label" in test.columns:
                test = test.drop("label", axis=1)

            fit_models = self.fit()

            for name, model in fit_models.items():
                logging.info(f"Testing {name} model")
                pred = model.predict(test)
                result[name] = pred

            logging.info("Testing completed")
            return result

        except Exception as e:
            logging.error("Error while testing")
            raise CustomException(e, sys)

    def predict_from_saved_models(self, test):
        logging.info("Loading saved models for prediction")
        result = {}

        try:
            if hasattr(test, "columns") and "label" in test.columns:
                test = test.drop("label", axis=1)

            if not os.path.exists(self.trainer_config.model_path):
                raise FileNotFoundError(
                    f"Model path does not exist: {self.trainer_config.model_path}"
                )

            model_files = os.listdir(self.trainer_config.model_path)

            for file_name in model_files:
                if file_name.endswith(".pkl"):
                    model_name = file_name.replace(".pkl", "")
                    model_path = os.path.join(self.trainer_config.model_path, file_name)

                    logging.info(f"Loading model {model_name}")
                    model = load_model(model_path)

                    logging.info(f"Predicting with {model_name}")
                    pred = model.predict(test)
                    result[model_name] = pred

            logging.info("Prediction using saved models completed")
            return result

        except Exception as e:
            logging.error("Error while predicting using saved models")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Running fit function")
            self.fit()

            logging.info("Running testing")
            test_csv = open_csv(self.dt_config.transformed_test)

            x_test = test_csv.drop("label", axis=1)
            y_test = test_csv["label"]

    
            res = self.predict(x_test)

            print("Predicted Result:", res)

            logging.info("Evaluating models")

            for model_name, pred in res.items():
                print(f"\nEvaluating {model_name}...\n")

                ac, ps, f1, rc, cm = self.evaluator.evaluate(y_test, pred)
                

                print(f"Model: {model_name}")
                print(f"Accuracy score: {ac}")
                print(f"Precision score: {ps}")
                print(f"F1 score: {f1}")
                print(f"Recall score: {rc}")
                print(f"Confusion matrix:\n{cm}")

                self.evaluator.get_evaluation_report(y_test, pred,f"{model_name}.pdf")

            logging.info("Running completed")

        except Exception as e:
            logging.error("Error while running trainer")
            raise CustomException(e, sys)