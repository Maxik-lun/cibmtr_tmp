import pandas as pd
import numpy as np
import scipy.stats as ss
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import quantile_transform
import warnings

def groupwise_transform(func, df, group = 'race_group'):
    groups = df[group].unique()
    y = pd.Series(np.nan, index = df.index)
    for grp in groups:
        ix = df[group] == grp
        y_ = func(df[ix])
        y[y_.index] = y_.values
    return y

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return pd.Series(y, index = df.index)

def transform_partial_hazard(df, time_col='efs_time', event_col='efs'):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/andreasbis/cibmtr-eda-ensemble-model
    """
    time, event = df[time_col], df[event_col]
    data = pd.DataFrame({'efs_time': time, 'efs': event, 'time': time, 'event': event})
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data, duration_col='time', event_col='event')
    return pd.Series(cph.predict_partial_hazard(data), index = df.index)

def transform_separate(df, time_col='efs_time', event_col='efs'):
    """Transform the target by separating events from non-events
    From https://www.kaggle.com/code/mtinti/cibmtr-lofo-feature-importance-gpu-accelerated"""
    time, event = df[time_col], df[event_col]
    transformed = time.values.copy()
    mx = transformed[event == 1].max() # last patient who dies
    mn = transformed[event == 0].min() # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = ss.rankdata(transformed)
    transformed[event == 0] += len(transformed) // 2
    transformed = transformed / transformed.max()
    return - pd.Series(transformed, index = df.index)

def transform_rank_log(df, time_col='efs_time', event_col='efs'):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    From https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676"""
    time, event = df[time_col], df[event_col]
    transformed = time.values.copy()
    mx = transformed[event == 1].max() # last patient who dies
    mn = transformed[event == 0].min() # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = ss.rankdata(transformed)
    transformed[event == 0] += len(transformed) * 2
    transformed = transformed / transformed.max()
    transformed = np.log(transformed)
    return - pd.Series(transformed, index = df.index)

def transform_quantile(df, time_col='efs_time', event_col='efs'):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    From https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense"""
    time, event = df[time_col], df[event_col]
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    return pd.Series(transformed, index = df.index)
