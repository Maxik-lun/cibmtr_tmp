import optuna
import xgboost as xgb
import numpy as np
from catboost import CatBoostRegressor
from catboost.utils import CatBoostError
from sklearn.model_selection import ShuffleSplit
from src.metric import eval_score, CustomMetric, MultiCustomMetric
from optuna.integration import XGBoostPruningCallback, CatBoostPruningCallback

def create_xgb_objective(data, FEATURES, TARGET, test_size = 0.3, 
                         booster = 'gbtree', n_rounds = 1000, early_stopping_rounds = 100):
    def objective(trial: optuna.Trial):
        rs = ShuffleSplit(2, test_size=test_size, random_state=13)
        train_index, test_index = next(rs.split(data))
        xgb_params = {
            'n_estimators': n_rounds,
            'max_depth': trial.suggest_int("max_depth", 4, 12, step=2),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 3, 20),
            'gamma': trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            'eta': trial.suggest_float("eta", 1e-4, 1.0, log=True),
            'tree_method': 'hist',
            'booster': booster, 'device': 'cpu',
            'enable_categorical': True
        }
        if booster == 'dart':
            xgb_params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            xgb_params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            xgb_params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 0.5, log=True)
            xgb_params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 0.5, log=True)
        else:
            xgb_params['early_stopping_rounds'] = early_stopping_rounds
        if TARGET == 'efs_time_cox':
                xgb_params['objective'] = 'survival:cox'
        else:
            xgb_params['objective'] = trial.suggest_categorical('objective', [
                'reg:squarederror', 'reg:absoluteerror'])
        # model validation
        meta_df = data.loc[test_index, ["ID","efs","efs_time","race_group"]].copy()
        def custom_metric(y_true, y_pred):
            return -eval_score(y_true, y_pred, meta_df = meta_df)
        x_train = data.loc[train_index,FEATURES].copy()
        x_valid = data.loc[test_index,FEATURES].copy()
        y_train = data.loc[train_index,TARGET]
        y_valid = data.loc[test_index,TARGET]
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-" + custom_metric.__name__)
        model = xgb.XGBRegressor(**xgb_params, random_state=11,
                                 eval_metric = custom_metric,
                                 callbacks = [pruning_callback])
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],  
            verbose=500 
        )
        y_pred = model.predict(x_valid)
        metric_score = custom_metric(y_valid, y_pred)
        return metric_score
    return objective

def create_cb_objective(data, FEATURES, TARGET, CATS, test_size = 0.3,
                        n_rounds = 1000, early_stopping_rounds = 100):
    def objective(trial: optuna.Trial):
        rs = ShuffleSplit(2, test_size=test_size, random_state=13)
        train_index, test_index = next(rs.split(data))
        cb_params = {
            'n_estimators': n_rounds, 'early_stopping_rounds': early_stopping_rounds,
            'max_depth': trial.suggest_int("max_depth", 4, 16,  step=2),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),#rsm
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
            # "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 31, step = 2),
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 22, step = 5),
            # 'fold_permutation_block': trial.suggest_int("fold_permutation_block", 1, 500, 10),
            'eta': trial.suggest_float("eta", 0.005, 1.0, log=True),
            # 'leaf_estimation_method': trial.suggest_categorical(
            #      'leaf_estimation_method', ['Newton', 'Gradient']),
            'grow_policy': trial.suggest_categorical(
                 'grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical(
                 'bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'task_type': 'CPU',
            # categories
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 2, 8)
        }
        if cb_params['bootstrap_type'] == 'Bayesian':
             cb_params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        elif cb_params["bootstrap_type"] == "Bernoulli":
            cb_params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)
        if TARGET == 'efs_time_cox':
            cb_params['objective'] = 'Cox'
            y_train = data.loc[train_index,TARGET]
            y_valid = data.loc[test_index,TARGET]
        elif TARGET == 'aft':
            dist_param = trial.suggest_categorical('aft_dist', ['Normal', 'Logistic', 'Extreme'])
            scale_param = trial.suggest_float("aft_scale", 0.01, 10.0, log=True)
            cb_params['objective'] = f'SurvivalAft:dist={dist_param};scale={scale_param}'
            y_train = data.loc[train_index, ['cb_lb', 'cb_ub']]
            y_valid = data.loc[test_index, ['cb_lb', 'cb_ub']]
        else:
            cb_params['objective'] = trial.suggest_categorical('objective', ['MAE', 'RMSE', 'LogCosh'])
            y_train = data.loc[train_index,TARGET]
            y_valid = data.loc[test_index,TARGET]
        # model validation
        meta_df = data[["ID","efs","efs_time","race_group"]].copy()
        x_train = data.loc[train_index,FEATURES].copy()
        x_valid = data.loc[test_index,FEATURES].copy()
        if TARGET == 'aft':
            pruning_callback = CatBoostPruningCallback(trial, 'MultiCustomMetric')
            eval_metric = MultiCustomMetric(meta_df, test_index, train_index)
        else:
            pruning_callback = CatBoostPruningCallback(trial, 'CustomMetric')
            eval_metric = CustomMetric(meta_df, test_index, train_index)
        model = CatBoostRegressor(**cb_params, random_state=42, cat_features=CATS,
                                 eval_metric = eval_metric)
        try:
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks = [pruning_callback],
                verbose=500 
            )
        except CatBoostError:
            return np.nan
        # evoke pruning manually.
        pruning_callback.check_pruned()
        y_pred = model.predict(x_valid)
        metric_score = eval_score(y_valid, y_pred, meta_df.loc[test_index])
        return metric_score
    return objective
