import pandas as pd
import numpy as np
from . import constants as ct
from itertools import combinations


def add_features(data, NUMS):
    df = data.copy()
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail'])
    df['dri_score']=df['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
    for col in ['diabetes','pulm_moderate','cardiac']:
        df.loc[df[col].isna(),col]='Not done'
    # hla feature gen
    HLA_COLS = [col for col in NUMS if 'hla_' in col]
    hla_df = {}
    for operator, col1, col2 in ct.HLA_GEN_EXPRS:
        if operator == 'diff':
            hla_df[f"{operator}_{col1}_{col2}"] = df.eval(f"{col1} - {col2}")
        elif operator == 'ad':
            hla_df[f"{operator}_{col1}_{col2}"] = df.eval(f"{col1} - {col2}").abs()
        elif operator == 'prod':
            hla_df[f"prod_{col1}_{col2}"] = df.eval(f"{col1}*{col2}")
    hla_df = pd.DataFrame(hla_df, index=df.index)
    df = pd.concat([df, hla_df], axis=1)
    df['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
    df['age_gap'] = np.abs(df['age_at_hct'] - df['donor_age'])
    return df

def combine_train_test(train, test):
    combined = pd.concat([train.assign(df_kind='train'),
                          test.assign(df_kind='test')],
                          axis=0,ignore_index=True)
    return combined

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


