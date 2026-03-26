import sys
import os
from fpdf import FPDF
from src.utils.logger import logging
from src.utils.exception import CustomException

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class Evaluator:
    def __init__(self):
        self.report_path = "reports"

    def evaluate(self, original, prediction):
        try:
            logging.info("Starting evaluation")

            logging.info("Calculating accuracy score")
            ac = accuracy_score(original, prediction)

            logging.info("Calculating precision score")
            ps = precision_score(original, prediction, average="weighted")

            logging.info("Calculating F1 score")
            f1 = f1_score(original, prediction, average="weighted")

            logging.info("Calculating confusion matrix")
            cm = confusion_matrix(original, prediction)

            logging.info("Calculating recall score")
            rc = recall_score(original, prediction, average="weighted")

            return ac, ps, f1, rc, cm

        except Exception as e:
            logging.error("Error during evaluation")
            raise CustomException(e, sys)

    def get_evaluation_report(self, original, prediction, filename="evaluation_report.pdf"):
        try:
            ac, ps, f1, rc, cm = self.evaluate(original, prediction)

            os.makedirs(self.report_path, exist_ok=True)

            pdf = FPDF()   
            pdf.add_page()
            pdf.set_font("Times", size=20)
            pdf.cell(200, 10, txt="Evaluation Report", ln=True, align="C")

            pdf.ln(10)
            pdf.set_font("Times", size=14)

            pdf.cell(200, 10, txt=f"Accuracy Score  : {ac:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Precision Score : {ps:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Recall Score    : {rc:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"F1 Score        : {f1:.4f}", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)

            for row in cm:
                pdf.cell(200, 10, txt="  ".join(map(str, row)), ln=True)

            file_path = os.path.join(self.report_path, filename)
            pdf.output(file_path)

            logging.info(f"Evaluation report saved at {file_path}")
            return file_path

        except Exception as e:
            logging.error("Error while generating evaluation report")
            raise CustomException(e, sys)