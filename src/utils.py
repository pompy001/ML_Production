import os
import sys
import pandas as pd
import numpy as np
from exception import ProjectException
import dill

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as  file_onj:
            dill.dump(obj,file_onj)

    except Exception as e:
        raise ProjectException(e,sys)