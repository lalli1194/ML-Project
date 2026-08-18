import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models
):
    try:
        report = {}

        for model_name, model in models.items():

            logging.info(
                f"Training model: {model_name}"
            )

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R2 scores
            train_score = r2_score(
                y_train,
                y_train_pred
            )

            test_score = r2_score(
                y_test,
                y_test_pred
            )

            logging.info(
                f"{model_name} - "
                f"Train R2: {train_score:.4f}, "
                f"Test R2: {test_score:.4f}"
            )

            # Store test R2
            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and test input data"
            )

            # Features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Regression models
            models = {

                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "Linear Regression": LinearRegression(),

                "K-Neighbors Regressor": KNeighborsRegressor(),

                "XGBoost Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1
                ),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                ),

                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                )
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Find best R2 score
            best_model_score = max(
                model_report.values()
            )

            # Find best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(best_model_score)
            ]

            # Get best model
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}"
            )

            logging.info(
                f"Best model R2 score: "
                f"{best_model_score:.4f}"
            )

            # Minimum acceptable score
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with R2 score >= 0.6",
                    sys
                )

            # Save best model
            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            # Final prediction
            predicted = best_model.predict(X_test)

            # Final R2 score
            r2_square = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"Final R2 Score: {r2_square:.4f}"
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            