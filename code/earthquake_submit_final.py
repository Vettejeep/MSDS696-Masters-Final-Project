#!/usr/bin/env python

__author__ = 'Kevin Maher'
__license__ = "Apache 2.0"
__email__ = 'vettejeep365@gmail.com'

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# derived from:
# Preda, G (2019). LANL Earthquake EDA and Prediction. Kaggle.
# Retrieved from: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

DATA_DIR = r'd:\#earthquake\data'  # set for local environment

# lightgbm parameters
params = {'num_leaves': 21,  # reducing to this helped with over fit
         'min_data_in_leaf': 20,
         'objective':'regression',
         'max_depth': 108,  # bogus, check lgb hard limit max depth, depth controlled by min_data_in_leaf
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def main():
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    scaled_train_X = pd.read_csv(r'pk8/scaled_train_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_train_X_8_slope.csv')  # adds in wavelet features
    scaled_train_X = scaled_train_X.join(df)

    scaled_test_X = pd.read_csv(r'pk8/scaled_test_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_test_X_8_slope.csv')  # adds in wavelet features
    scaled_test_X = scaled_test_X.join(df)

    pcol = []
    pcor = []
    pval = []
    y = pd.read_csv(r'pk8/train_y_8.csv')['time_to_failure'].values

    # use pearson's to eliminate features with suspect correlation - helped kaggle scores
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= 0.05]

    drop_cols = []

    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    train_y = pd.read_csv(r'pk8/train_y_8.csv')
    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    # as an alternate, select 6 fold and shuffle=False for non-overlap cross validation
    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        t0 = time.time()
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        # use commenting to select lgb or xgb model
        # model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
        # model.fit(X_tr, y_tr,
        #           eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
        #           verbose=1000, early_stopping_rounds=200)

        model = xgb.XGBRegressor(n_estimators=1000,
                                       learning_rate=0.1,
                                       max_depth=6,
                                       subsample=0.9,
                                       colsample_bytree=0.67,
                                       reg_lambda=1.0, # seems best within 0.5 of 2.0
                                       # gamma=1,
                                       random_state=777+fold_,
                                       n_jobs=12,
                                       verbosity=2)
        model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)  # uncomment for lgb, and below
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits

        # validation error
        # mean absolute error
        preds = model.predict(X_val)  # , num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # training error, monitor for over fit - still an area for improvement despite great kaggle scores
        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)

        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        tr_maes.append(mae)

        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)
        print('loop time:', time.time() - t0)

    # print out statistics for validation and training sets (training monitors overfitting)
    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    # reset file names as needed
    submission.time_to_failure = predictions
    submission.to_csv('$submission_xgb_slope_pearson_6fold.csv')  # index needed, it is seg id

    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'preds_tr_xgb_slope_pearson.csv', index=False)
    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))



if __name__ == '__main__':
    main()
    print('DONE!')
