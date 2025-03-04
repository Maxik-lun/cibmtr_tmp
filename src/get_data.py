import pandas as pd
import numpy as np
from . import constants as ct

def recalculate_hla_sums(data):
    df = data.copy()
    # Calculate new columns by summing existing columns after filling NaNs with 0
    df["hla_nmdp_6"] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df["hla_low_res_6"] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_low"].fillna(0)
    df["hla_high_res_6"] = df["hla_match_a_high"].fillna(0) + df["hla_match_b_high"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df["hla_low_res_8"] = (df["hla_match_a_low"].fillna(0) + 
                           df["hla_match_b_low"].fillna(0) + 
                           df["hla_match_c_low"].fillna(0) + 
                           df["hla_match_drb1_low"].fillna(0))
    df["hla_high_res_8"] = (df["hla_match_a_high"].fillna(0) + 
                            df["hla_match_b_high"].fillna(0) + 
                            df["hla_match_c_high"].fillna(0) + 
                            df["hla_match_drb1_high"].fillna(0))
    df["hla_low_res_10"] = (df["hla_match_a_low"].fillna(0) + 
                            df["hla_match_b_low"].fillna(0) + 
                            df["hla_match_c_low"].fillna(0) + 
                            df["hla_match_drb1_low"].fillna(0) +
                            df["hla_match_dqb1_low"].fillna(0))
    df["hla_high_res_10"] = (df["hla_match_a_high"].fillna(0) + 
                             df["hla_match_b_high"].fillna(0) + 
                             df["hla_match_c_high"].fillna(0) + 
                             df["hla_match_drb1_high"].fillna(0) +
                             df["hla_match_dqb1_high"].fillna(0))
    return df

def add_features(data, NUMS, CATS):
    df = data.copy()
    df['nan_value_num_cnt'] = df[NUMS].isna().sum(axis=1)
    df['nan_value_cat_cnt'] = df[CATS].isna().sum(axis=1)
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail'])
    df['year_hct']=df['year_hct'].replace(2020,2019)
    df['year_hct'] -= 2000
    df['age_group']=df['age_at_hct']//10
    #karnofsky_score 40 only 10 rows.
    df['karnofsky_score']=df['karnofsky_score'].replace(40,50)
    #hla_high_res_8=2 only 2 rows.
    df['hla_high_res_8']=df['hla_high_res_8'].replace(2,3)
    #hla_high_res_6=0 only 1 row.
    df['hla_high_res_6']=df['hla_high_res_6'].replace(0,2)
    #hla_high_res_10=3 only 1 row.
    df['hla_high_res_10']=df['hla_high_res_10'].replace(3,4)
    #hla_low_res_8=2 only 1 row.
    df['hla_low_res_8']=df['hla_low_res_8'].replace(2,3)
    df['dri_score']=df['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
    for col in ['diabetes','pulm_moderate','cardiac']:
        df.loc[df[col].isna(),col]='Not done'
    df['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
    df['comorbidity_score+karnofsky_score']=df['comorbidity_score']+df['karnofsky_score']
    df['comorbidity_score-karnofsky_score']=df['comorbidity_score']-df['karnofsky_score']
    df['comorbidity_score*karnofsky_score']=df['comorbidity_score']*df['karnofsky_score']
    df['comorbidity_score/karnofsky_score']=df['comorbidity_score']/df['karnofsky_score']
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


