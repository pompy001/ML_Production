import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Explanation of above code
import os: This line imports the built-in os module, which provides a way to interact with the operating system in a platform-independent way.

import sys: This line imports the built-in sys module, which provides access to some variables and functions that interact with the Python interpreter.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))): This line adds a directory to the Python module search path (sys.path) so that Python can find and import modules from that directory. Here's a breakdown of the different parts of this line:

os.path.dirname(__file__): This returns the directory name of the current script (__file__ is a special variable that contains the path of the currently executing script).

os.path.join(os.path.dirname(__file__), ".."): This joins the directory name of the current script with ".." (which means the parent directory).

os.path.abspath(os.path.join(os.path.dirname(__file__), "..")): This gets the absolute path of the parent directory of the current script.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))): This inserts the absolute path of the parent directory of the current script at the beginning of the sys.path list, which means that Python will look for modules in that directory before looking in any other directories.
"""


import sys 
from dataclasses import dataclass
from exception import ProjectException
from logger import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from utils import save_object
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns =['writing_score','reading_score']
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            
                ]
            )

            return preprocessor

        except Exception as e:
            raise ProjectException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns =['writing_score','reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]


            #applying preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise ProjectException(e,sys)









