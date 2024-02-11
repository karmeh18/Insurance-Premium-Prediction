import sys
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import save_obj
from dataclasses import dataclass
from src.exception import Custom_Exception
from src.logger import logging
import numpy as np

@dataclass
class ModelTrainer:
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            self.model_trainer_config=os.path.join('artifacts','model.pkl')
            logging.info('Model Trainer class has been initiated')
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,:-1],
                                           test_arr[:,-1])
            lr=LinearRegression()
            logging.info("Linear Regression has been initiated")
            model=lr.fit(X_train,y_train)
            pred=model.predict(X_test)
            score=r2_score(y_test,pred)
            logging.info("Model Training has been completed and the accuracy is {}".format(score))
            independent_variable=np.vstack((X_train,X_test))
            dependent_variable=np.vstack((y_train.reshape(-1,1),y_test.reshape(-1,1)))
            complete_model=lr.fit(independent_variable,dependent_variable)
            save_obj(self.model_trainer_config,complete_model)
            logging.info("Model has been saved that has been trained on combined data of Training data and testing data")
            return score
        except Exception as e:
            raise Custom_Exception(e,sys)
        
