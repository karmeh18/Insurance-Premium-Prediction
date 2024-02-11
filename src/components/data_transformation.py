import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

from src.utils import save_obj
from src.exception import Custom_Exception
from src.logger import logging

@dataclass
class Data_Transformation:
    def get_data_transformation_object(self):
        
        """
        This function is reponsible for Data Transformation
        """
        try:
            num_cols=['age', 'bmi', 'children']
            cat_cols=['gender',
                      'smoker',
                      'region',
                      'medical_history',
                      'family_medical_history',
                      'exercise_frequency',
                      'occupation',
                      'coverage_level']
    
            num_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='mean')),
                    ("Standardization",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical pipeline has been created for the columns {}".format(num_cols))
            cat_pipeline=Pipeline(
                steps=[
                    ('TargetEncoder',TargetEncoder(cols=cat_cols)),
                    ("StandardScaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical pipeline has been created for the columns {}".format(cat_cols))

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
            ])
            return preprocessor            

        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,raw_data_path):
        try:
            self.data_transformation_config=os.path.join('artifacts','preprocessor.pkl')
            self.train_data_path=os.path.join('artifacts','train_data.csv')
            self.test_data_path=os.path.join('artifacts','test_data.csv')
            raw_data=pd.read_csv(raw_data_path)
            X=raw_data.drop("charges",axis=1)
            y=raw_data['charges']

            logging.info("Data has been splitted in")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            logging.info("Exporting Training data and test data")
            export_train_data=pd.concat([X_train,y_train],axis=1)
            export_test_data=pd.concat([X_test,y_test],axis=1)
            export_train_data.to_csv(self.train_data_path,index=False,header=True)
            export_test_data.to_csv(self.test_data_path,index=False,header=True)

            logging.info("Applying preprocess object to Training and Testing Data")

            preprocessor_obj=self.get_data_transformation_object()
            input_features_train_arr=preprocessor_obj.fit_transform(X_train,y_train)
            input_feature_test_arr=preprocessor_obj.fit_transform(X_test,y_test)

            logging.info("Applied preprocesor object to training and test data")

            logging.info("Combining preprocessed data output, Input variables and target variable in both training and test data")
            train_arr=np.c_[input_features_train_arr,np.array(y_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(y_test)]

            logging.info('Data points have been preprocessed and combined in both testing and training data points')

            save_obj(file_path=self.data_transformation_config,obj=preprocessor_obj)
            return (train_arr,
                    test_arr)
        
        except Exception as e:
            raise Custom_Exception(e,sys)

