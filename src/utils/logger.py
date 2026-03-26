import logging
import os
from datetime import datetime

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = f'{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log'
LOG_PATH = os.path.join(LOG_DIR,LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[ %(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s]"
)

console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)