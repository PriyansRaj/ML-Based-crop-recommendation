from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier
)
from xgboost import XGBClassifier


class ModelInitializer:
    def __init__(self):
        self.models = self._initialize_models()

    def _initialize_models(self):
        models = {
            "perceptron": Perceptron(),
            "svm": SVC(probability=True),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "adaboost": AdaBoostClassifier(),
            "xgboost": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss"
            ),

         
            "voting_classifier": VotingClassifier(
                estimators=[
                    ("svm", SVC(probability=True)),
                    ("knn", KNeighborsClassifier()),
                    ("dt", DecisionTreeClassifier()),
                    ("rf", RandomForestClassifier())
                ],
                voting="soft"
            ),

           
            "bagging_knn": BaggingClassifier(
                estimator=KNeighborsClassifier(),
                n_estimators=10,
                random_state=42
            ),

        
            "bagging_svm": BaggingClassifier(
                estimator=SVC(probability=True),
                n_estimators=10,
                random_state=42
            )
        }
        return models

    def get_models(self):
        return self.models

    def get_model(self, model_name):
        return self.models.get(model_name, None)