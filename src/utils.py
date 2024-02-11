import os
import sys

import numpy as np
import pandas as pd
import pickle

from src.exception import Custom_Exception

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)