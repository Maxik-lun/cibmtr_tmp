from .get_data import prepare_data
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