import os
import sys
import pandas as pd

from src.exception import Custom_Exception
from src.logger import logging
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import ModelTrainer


class DataIngestion:
    def initiate_data_ingestion(self):
        self.data_path=os.path.join('artifacts','data.csv')
        logging.info('Data Ingestion has been initiated')
        try:
            df=pd.read_csv("notebook\data\insurance_dataset.csv")
            logging.info("Data has been imported")

            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            logging.info("Raw Data directory has been created and a csv file has been exported to the path '{}'".format(self.data_path))

            return self.data_path
        except Exception as e:
            raise Custom_Exception(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()

    transformation=Data_Transformation()
    train_arr,test_arr=transformation.initiate_data_transformation(raw_data_path)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))