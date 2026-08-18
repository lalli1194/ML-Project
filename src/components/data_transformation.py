import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformation_object(self):

        """
        This function is responsible for
        data transformation based on different
        types of data.
        """

        try:

            numerical_columns = [
                "writing score",
                "reading score"
            ]

            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            logging.info(
                "Numerical pipeline completed"
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler(
                            with_mean=False
                        )
                    )
                ]
            )

            logging.info(
                "Categorical pipeline completed"
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            logging.info(
                "Preprocessor object created successfully"
            )

            return preprocessor

        except Exception as e:

            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # Read train and test data
            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)

            logging.info(
                "Read train and test data completed"
            )

            # Get preprocessing object
            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformation_object()
            )

            # Target column
            target_column_name = "math score"

            # Separate input and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logging.info(
                "Applying preprocessing object on "
                "training and testing dataframe"
            )

            # Fit and transform training data
            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Transform testing data
            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(
                "Saving preprocessing object"
            )

            # Save preprocessor
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                "Preprocessor saved successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:

            raise CustomException(e, sys)