import pandas as pd
import numpy as np
from src.data_pipeline import baseline_preprocess, create_all_targets
import src.constants as ct
from optuna_objective import create_xgb_objective
import optuna

def xgb_main(TARGET, study_name = "base", min_resource = 30, reduction_factor = 3):
    # target_choice = ['km_y', 'hazard_y', 'efs_time_cox', 'quantile_y', 'separate_y']
    train, test, CATS, FEATURES = baseline_preprocess(ct.LOCAL_TRAIN, ct.LOCAL_TEST, ct.LOCAL_DESCRIPTION)
    train = create_all_targets(train)
    storage_name = f"sqlite:///{study_name}.db"
    sampler = optuna.samplers.TPESampler(seed=101) # create a seed for the sampler for reproducibility
    pruner = optuna.pruners.HyperbandPruner(min_resource=min_resource, reduction_factor=reduction_factor)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                            sampler=sampler, pruner=pruner, direction="maximize")
    objective = create_xgb_objective(train, FEATURES, TARGET)
    study.optimize(objective, n_trials=500)    