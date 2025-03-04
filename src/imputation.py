from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

def knn_impute(data, NUMS):
    X = data.copy()
    scaler = MinMaxScaler()
    model = KNNImputer(n_neighbors=5)
    X_num = scaler.fit_transform(X[NUMS])
    X_num = model.fit_transform(X_num)
    X_num = scaler.inverse_transform(X_num)
    X[NUMS] = X_num
    return X

def catboost_iterimpute(df, CATS, FEATURES, MISS_COLS, cat_params, threshold = 0.7, max_iterations=5, na_val = 'NAN'):
    # Step 2: Initially fill all missing values with "Missing"
    NA_VAL = na_val
    df_ = df[FEATURES].copy()
    for c in CATS:
        df_[c] = df_[c].fillna(NA_VAL)
    mask_dict = {}
    for c in MISS_COLS:
        mask_dict[c] = df_[c] == NA_VAL
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in MISS_COLS:
            other_features = [x for x in CATS if x != feature]
            # Step 3: Use the remaining features to predict missing values
            X_train = df_[~mask_dict[feature]].drop(columns=[feature])
            y_train = df_[~mask_dict[feature]][feature]
            X_test = df_[mask_dict[feature]].drop(columns=[feature])
            catboost_classifier = CatBoostClassifier(**cat_params, random_state = 13)
            catboost_classifier.fit(X_train, y_train, cat_features=other_features, verbose=False)
            # Step 4: Predict missing values for the feature and update all N features
            y_pred = pd.DataFrame(catboost_classifier.predict_proba(X_test),
                                  columns = catboost_classifier.classes_,
                                  index = X_test.index)
            y_pred = np.where(y_pred.max(axis = 1) >= threshold, y_pred.idxmax(axis=1), NA_VAL)
            df_.loc[X_test.index, feature] = y_pred
    return df_