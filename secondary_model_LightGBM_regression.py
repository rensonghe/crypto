import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import datetime
import joblib
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
import pickle
def read_pickle(path):
    with open(path, "rb") as f:
        obj_again = pickle.load(f)
    return obj_again
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))




def secondary_lightgbm_model(X_train, X_train_target, X_test, X_test_target, secondary_LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    _y_pred = np.zeros(len(X_test_target))
    __y_pred = np.zeros(len(X_test_target))
    _y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []

    weight = [0.03, 0.05, 0.07, 0.2, 0.65]
    # weight = [0.01, 0.02, 0.03, 0.04, 0.9]

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        w = weight[fold]

        params = {
            'boosting_type': 'gbdt',
            # 'metric': 'multi_logloss',
            # 'objective': 'multiclass',
            'metric': {'cross_entropy','auc','average_precision',},
            'objective': 'binary',  # regression,binary,multiclass
            # 'num_class': 3,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'num_leaves': int(secondary_LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(secondary_LGB_BO.max['params']['learning_rate']),
            'max_depth': int(secondary_LGB_BO.max['params']['max_depth']),
            'n_estimators': int(secondary_LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(secondary_LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(secondary_LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(secondary_LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(secondary_LGB_BO.max['params']['subsample']),
            'min_child_samples': int(secondary_LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(secondary_LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(secondary_LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(secondary_LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(secondary_LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(secondary_LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 40
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        # y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred += model.predict(X_test, num_iteration=model.best_iteration) * w
        __y_pred += model.predict(X_test, num_iteration=model.best_iteration)

        y_pred = _y_pred / kf.n_splits
        # y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) * w

        y_pred_train = _y_pred_train / kf.n_splits
        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list

