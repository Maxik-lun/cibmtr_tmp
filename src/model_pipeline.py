import pandas as pd
import numpy as np
import scipy.stats as ss
from . import constants as ct
import xgboost as xgb
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_predict
from .metric import score

def kfold_validation(model, data, FEATURES, TARGET):
    kf = KFold(n_splits=ct.FOLDS, shuffle=True, random_state=42)
    model_oof = np.zeros(len(data))
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        print("#"*25)
        print(f"### Fold {i+1}")
        print("#"*25)
        x_train = data.loc[train_index,FEATURES].copy()
        x_valid = data.loc[test_index,FEATURES].copy()
        if TARGET == 'aft':
            y_train = data.loc[train_index, ['cb_lb', 'cb_ub']]
            y_valid = data.loc[test_index, ['cb_lb', 'cb_ub']]
        else:
            y_train = data.loc[train_index,TARGET]
            y_valid = data.loc[test_index,TARGET]
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],  
            verbose=500 
        )
        # elif aft == 'xgb':
        #     dtrain = xgb.DMatrix(x_train)
        #     dtrain.set_float_info('label_lower_bound', data.loc[train_index, 'xgb_lb'])
        #     dtrain.set_float_info('label_upper_bound', data.loc[train_index, 'xgb_ub'])
        #     dval = xgb.DMatrix(x_valid)
        #     dval.set_float_info('label_lower_bound', data.loc[test_index, 'xgb_lb'])
        #     dval.set_float_info('label_upper_bound', data.loc[test_index, 'xgb_ub'])
        #     bst = xgb.train(xgb_params, dtrain,
        #         evals=[(dtrain, 'train')])
        # INFER OOF
        model_oof[test_index] = model.predict(x_valid)
    true_df = data[["ID","efs","efs_time","race_group"]].copy()
    pred_df = data[["ID"]].copy()
    pred_df["prediction"] = model_oof
    metric_score = score(true_df.copy(), pred_df.copy(), "ID")
    return model_oof, metric_score

def simple_validation(model, data, FEATURES, TARGET, test_size = 0.3):
    rs = ShuffleSplit(2, test_size=test_size, random_state=13)
    train_index, test_index = next(rs.split(data))
    x_train = data.loc[train_index,FEATURES].copy()
    x_valid = data.loc[test_index,FEATURES].copy()
    if TARGET == 'aft':
        y_train = data.loc[train_index, ['cb_lb', 'cb_ub']]
        y_valid = data.loc[test_index, ['cb_lb', 'cb_ub']]
    else:
        y_train = data.loc[train_index,TARGET]
        y_valid = data.loc[test_index,TARGET]
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )
    true_df = data.loc[test_index, ["ID","efs","efs_time","race_group"]].copy()
    pred_df = data.loc[test_index, ["ID"]].copy()
    pred_df["prediction"] = model.predict(x_valid)
    metric_score = score(true_df.copy(), pred_df.copy(), "ID")
    return metric_score
