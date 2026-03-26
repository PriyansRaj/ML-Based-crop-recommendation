import pandas as pd
import pickle
from src.utils.logger import logging 
import sys

from src.utils.exception import CustomException

def open_csv(file_name: str)->pd.DataFrame:
    logging.info("Opening datafile")
    try:
        df = pd.read_csv(file_name)
        logging.info("Datafile opened")
        return df
    except Exception as e:
        logging.error("Error while opening dataframe")
        raise CustomException(e,sys)


def save_model(path:str,obj):
    logging.info("Saving the model")
    try:
        pickle.dump(obj,open(path,'wb'))
        logging.info("Model saved")
    except Exception as e:
        logging.error("Error while saving model")
        raise CustomException(e,sys)


def load_model(path: str):
    logging.info("Loading the model")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error("Error while loading model")
        raise CustomException(e, sys)