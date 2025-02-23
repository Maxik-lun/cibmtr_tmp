#!/usr/bin/env python
# coding: utf-8

"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index
from catboost import MultiTargetCustomMetric
class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.merge(solution, submission, on = row_id_column_name)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group'], observed = True).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

def eval_score(y_true, y_pred, meta_df = None):
    event_label = 'efs'
    interval_label = 'efs_time'
    if meta_df is None:
        return np.mean(np.abs(y_true - y_pred))
    merged_df = meta_df[['race_group', event_label, interval_label]].copy()
    merged_df['pred'] = y_pred
    # merged_df['true'] = y_true
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group'], observed = True).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(merged_df_race[interval_label],
                                         -merged_df_race['pred'],
                                         merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

class MultiCustomMetric(MultiTargetCustomMetric):
    def __init__(self, meta_df, test_idx, train_idx) -> None:
            self.meta_df = meta_df.copy()
            self.test_idx = test_idx
            self.train_idx = train_idx

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.
        approx = approxes[0]
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        w = 1.0
        if len(approx)/len(self.meta_df) < 0.5:
            df_ = self.meta_df.loc[self.test_idx]
        else:
            df_ = self.meta_df.loc[self.train_idx]
        metric_val = eval_score(target, approx, meta_df = df_)
        return metric_val, w

    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return error

class CustomMetric(object):
    def __init__(self, meta_df, test_idx, train_idx) -> None:
            self.meta_df = meta_df.copy()
            self.test_idx = test_idx
            self.train_idx = train_idx

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        w = 1.0
        if len(approx)/len(self.meta_df) < 0.5:
            df_ = self.meta_df.loc[self.test_idx]
        else:
            df_ = self.meta_df.loc[self.train_idx]
        metric_val = eval_score(target, approx, meta_df = df_)
        return metric_val, w

    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return error