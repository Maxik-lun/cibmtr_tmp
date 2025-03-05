from .get_data import prepare_data, add_features, combine_train_test
from .imputation import knn_impute, catboost_iterimpute
from . import constants as ct
from . import target_gen as tg
import numpy as np
import pandas as pd

def create_all_targets(df):
    train = df.copy()
    train["group_km_y"] = tg.groupwise_transform(tg.transform_survival_probability, train)
    train["km_y"] = tg.transform_survival_probability(train)
    train["efs_time_cox"] = np.where(train.efs==1, train.efs_time, -train.efs_time)
    # train["hazard_y"] = tg.transform_partial_hazard(train)
    train["group_hazard_y"] = tg.groupwise_transform(tg.transform_partial_hazard, train)
    train["quantile_y"] = tg.transform_quantile(train)
    train["rank_log_y"] = tg.transform_rank_log(train)
    train["separate_y"] = tg.transform_separate(train)
    # CatBoost Accelerated failure time model
    train['cb_lb'] = train.efs_time
    train['cb_ub'] = np.where(train.efs == 1, train.efs_time, -1)
    # XGBoost Accelerated failure time model
    train['xgb_lb'] = train.efs_time # Observed survival time
    train['xgb_ub'] = np.where(train.efs == 1, train.efs_time, float("inf")) # Censored data upper bound is infinity
    return train

def baseline_preprocess(train_path = None, test_path = None, data_info = None):
    train, test, CATS = prepare_data(train_path, test_path, data_info)
    for c in CATS:
        train[c] = train[c].fillna("NAN").astype('category')
        test[c] = test[c].fillna("NAN").astype('category')
    FEATURES = [c for c in train.columns if not c in ct.RMV]
    return train, test, CATS, FEATURES

def advanced_preprocess(train_path = None, test_path = None, data_info = None):
    train, test, CATS = prepare_data(train_path, test_path, data_info)
    FEATURES = [c for c in train.columns if not c in ct.RMV]
    NUMS = [c for c in FEATURES if not c in CATS]
    combined_df = combine_train_test(train, test)
    # count nans
    combined_df['nan_value_num_cnt'] = combined_df[NUMS].isna().sum(axis=1)
    combined_df['nan_value_cat_cnt'] = combined_df[CATS].isna().sum(axis=1)
    # data cleaning numerical
    num_df = knn_impute(combined_df, NUMS)
    combined_df[NUMS] = num_df[NUMS]
    # new features
    combined_df = add_features(combined_df, NUMS)
    FEATURES = [c for c in combined_df.columns if not c in ct.RMV and c != 'df_kind']
    NUMS = [c for c in FEATURES if not c in CATS]
    # for xgb
    for c in CATS:
        combined_df[c] = combined_df[c].fillna('NAN').astype('category')
    train_new = combined_df.query("df_kind == 'train'")
    test_new = combined_df.query("df_kind == 'test'")
    train_new = train_new.set_index(train.index)
    test_new = test_new.set_index(test.index)
    del train_new['df_kind']
    del test_new['df_kind']
    del test_new['efs']
    del test_new['efs_time']
    return train_new, test_new, CATS, FEATURES