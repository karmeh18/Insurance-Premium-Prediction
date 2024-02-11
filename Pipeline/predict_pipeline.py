import sys
import os
import pandas as pd

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_obj

class PredictPipeline:
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info('Model and Preprocessor object has been loaded')

            model=load_obj(model_path)
            preprocessor=load_obj(preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)
        

class CustomData:
    def __init__(self,
                 gender,
                 smoker,
                 region,
                 medical_history,
                 family_medical_history,
                 exercise_frequency,
                 occupation,
                 coverage_level,
                 age,
                 bmi,
                 children):
        
        self.gender=gender
        self.smoker=smoker
        self.region=region
        self.medical_history=medical_history
        self.family_medical_history=family_medical_history
        self.exercise_frequency=exercise_frequency
        self.occupation=occupation
        self.coverage_level=coverage_level
        self.age=age
        self.bmi=bmi
        self.children=children

    #'gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'
    def get_data_as_dataframe(self):
        try:
            custom_data_input={
                'gender':[self.gender],
                'smoker':[self.smoker],
                'region':[self.region],
                'medical_history':[self.medical_history],
                'family_medical_history':[self.family_medical_history],
                'exercise_frequency':[self.exercise_frequency],
                'occupation':[self.occupation],
                'coverage_level':[self.coverage_level],
                'age':[self.age],
                'bmi':[self.bmi],
                'children':[self.children]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise Custom_Exception(e,sys)