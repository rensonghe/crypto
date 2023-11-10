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

def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', pearsonr(preds, train_data.get_label())[0], is_higher_better



def lightgbm_model(train_, X_train, X_train_target,X_train_, X_train_target_, X_test, X_test_target, X_test_, X_test_target_, LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    _y_pred = np.zeros(len(X_test_target))
    y_pred_ = np.zeros(len(X_test_target_))
    _y_pred_ = np.zeros(len(X_test_target_))
    y_pred_train = np.zeros(len(X_train_target))
    y_pred_train_ = np.zeros(len(X_train_target_))
    _y_pred_train = np.zeros(len(X_train_target))
    _y_pred_train_ = np.zeros(len(X_train_target_))
    importances = []
    model_list = []
    LGB_BO.max['params'] = LGB_BO.max['params']
    features = train_.iloc[:, 5:105].columns
    features = list(features)
    weight = [0.03,0.05,0.07,0.2,0.65]
    # weight = [0.01, 0.02, 0.03, 0.04, 0.9]

    def plot_importance(importances, features, PLOT_TOP_N=20, figsize=(20, 20)):
        importance_df = pd.DataFrame(data=importances, columns=features)
        sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
        sorted_importance_df = importance_df.loc[:, sorted_indices]
        plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
        _, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_xscale('log')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        sns.boxplot(data=sorted_importance_df[plot_cols],
                    orient='h',
                    ax=ax)
        plt.show()

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_, X_train_target_)):
        print('Model:',fold)
        x_train, x_val = X_train_[train_index], X_train_[val_index]
        y_train, y_val = X_train_target_[train_index], X_train_target_[val_index]
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
            'num_leaves': int(LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(LGB_BO.max['params']['learning_rate']),
            'max_depth': int(LGB_BO.max['params']['max_depth']),
            'n_estimators': int(LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(LGB_BO.max['params']['subsample']),
            'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 40
        }

        # weight = [i for i in weight]

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        # y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred += model.predict(X_test, num_iteration=model.best_iteration) * w
        # print(_y_pred[:10])
        y_pred = _y_pred/ kf.n_splits
        # y_pred_ += model.predict(X_test_, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_ += model.predict(X_test_, num_iteration=model.best_iteration) * w
        y_pred_ = _y_pred_ / kf.n_splits
        # y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) * w
        # print(_y_pred_train[:10])
        y_pred_train = _y_pred_train / kf.n_splits
        # y_pred_train_ += model.predict(X_train_, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train_ += model.predict(X_train_, num_iteration=model.best_iteration) * w
        y_pred_train_ = _y_pred_train_/ kf.n_splits

        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list, y_pred_, y_pred_train_


def first_model_train_test(X_train_target_, first_y_pred_train, X_test_target_, first_y_pred):

    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    from numpy import sqrt, argmax
    fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target_, first_y_pred_train)
    gmeans_train = sqrt(tpr_train * (1-fpr_train))
    ix_train = argmax(gmeans_train)
    # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
    thresholds_point_train = thresholds_train[ix_train]
    first_yhat_train = [1 if y > thresholds_point_train else 0 for y in first_y_pred_train]
    # print("secondary_model训练集表现：")
    # print(classification_report(yhat_train,X_train_target))

    fpr, tpr, thresholds = roc_curve(X_test_target_, first_y_pred)
    # fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # thresholds_point = thresholds_train[ix_train]
    first_yhat = [1 if y > thresholds[ix] else 0 for y in first_y_pred]
    # yhat = [1 if y > 0.55 else 0 for y in y_pred]
    # print("secondary_model测试集表现：")
    # print(classification_report(secondary_yhat,X_test_target))
    # print(metrics.confusion_matrix(yhat, X_test_target))
    # print('AUC:', metrics.roc_auc_score(secondary_yhat, X_test_target))
    return first_yhat_train, first_yhat