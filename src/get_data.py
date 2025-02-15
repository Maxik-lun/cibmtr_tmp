import pandas as pd
import numpy as np
from . import constants as ct

def prepare_data(train_path = None, test_path = None, data_info = None):
    train_path = train_path or ct.KAGGLE_TRAIN
    test_path = test_path or ct.KAGGLE_TEST
    data_info = data_info or ct.KAGGLE_DESCRIPTION
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    data_dictionary = pd.read_csv(data_info)
    CATS = data_dictionary[data_dictionary['type'] == 'Categorical']['variable'].tolist()
    CATS = [c for c in CATS if not c in ct.RMV]
    return train, test, CATS


