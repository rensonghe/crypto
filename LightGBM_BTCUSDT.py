#%%
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
#%%
symbol = 'xrpusdt'
amount = '120000'
min = 20
data_2022 = read_pickle('/home/data_crypto/binance/%s/tick_factor/%s_tick_factor_%s_%smin_2022.pkl'%(symbol,symbol, amount, min))
data_2023 = read_pickle('/home/data_crypto/binance/%s/tick_factor/%s_tick_factor_%s_%smin_2023.pkl'%(symbol,symbol, amount, min))
data = pd.concat([data_2022, data_2023], axis=0)
del data_2022, data_2023
data = data.sort_values(by='closetime', ascending=True)

#%%
bar = 5
x = 0.001
vwap = 'vwap_5s'
data['target'] = np.log(data[vwap]/data[vwap].shift(bar))
data['target'] = data['target'].shift(-bar)
# data['return'] = np.log(data['price']).diff()
# data['return'] = data['return'].shift(-1)
# data['volaility'] = data['return'].rolling(bar).apply(realized_volatility)
# data['volaility'] = data['volaility'].shift(-bar)
def classify(y):

    if y < -x:
        return 0
    if y > x:
        return 1
    else:
        return -1
print(data['target'].apply(lambda x:classify(x)).value_counts())
print(len(data[data['target'].apply(lambda x:classify(x))==-1])/len(data['target'].apply(lambda x:classify(x))))
#%%
def calcpearsonr(data,rolling):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[7:88]):

        ic = data[column].rolling(rolling).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)
        IC = pd.DataFrame(ic_list)
        columns = pd.DataFrame(data.columns[7:88])
        IC_columns = pd.concat([IC, columns], axis=1)
        col = ['value', 'factor']
        IC_columns.columns = col
    return IC_columns
IC_columns = calcpearsonr(data,rolling=bar)
#%%
time_1 = '2023-09-29 00:00:00'
time_2 = '2023-09-30 23:59:59'

cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[data.index < time_1]
test = data[(data.index >= time_1)&(data.index <= time_2)]
train['target'] = train['target'].apply(lambda x:classify(x))
train_ = train[~train['target'].isin([-1])]

train_set_ = train_[train_col]
train_set_ = train_set_.iloc[:,5:90] #90
train_target_ = train_["target"]

train_set = train[train_col]
train_set = train_set.iloc[:,5:90] #90
train_target = train["target"]

test_set = test[train_col]
test_set = test_set.iloc[:,5:90]
test_target = test["target"]
print(test['target'].apply(lambda x:classify(x)).value_counts())
print(len(test[test['target'].apply(lambda x:classify(x))==-1])/len(test['target'].apply(lambda x:classify(x))))
# #
test_ = test.copy()
test_['target'] = test_['target'].apply(lambda x:classify(x))
test_ = test_[~test_['target'].isin([-1])]
test_set_ = test_[train_col]
test_set_ = test_set_.iloc[:,5:90]
test_target_ = test_["target"]
#
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_train_ = np.array(train_set_)
X_train_target_ = np.array(train_target_)

X_test = np.array(test_set)
X_test_target = np.array(test_target)
X_test_ = np.array(test_set_)
X_test_target_ = np.array(test_target_)
#
del train_set, test_set, train_target, test_target, test_set_, test_target_, test_
df = test.copy()
df = df.reset_index()
df['min'] = ((df['closetime']-df['closetime'].shift(bar))/1000)/60
print(df['min'].describe())
print(abs(df['target']).describe())
del df

#%% first model
def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', pearsonr(preds, train_data.get_label())[0], is_higher_better
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample):
    # LightGBM expects next three parameters need to be integer.
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    learning_rate = float(learning_rate)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    n_estimators = int(n_estimators)
    min_child_samples = float(min_child_samples)
    min_split_gain = float(min_split_gain)
    # scale_pos_weight = float(scale_pos_weight)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    kf = TimeSeriesSplit(n_splits=5)
    X_train_pred = np.zeros(len(X_train_target_))
    _X_train_pred = np.zeros(len(X_train_target_))
    # weight = [1, 1, 1, 1, 1]

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_, X_train_target_)):
        x_train, x_val = X_train_[train_index], X_train_[val_index]
        y_train, y_val = X_train_target_[train_index], X_train_target_[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 2 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)

        # w = weight[fold]
        params = {
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'min_split_gain': min_split_gain,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'subsample': subsample,
            'n_estimators': n_estimators,
            # 'learning_rate' : learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'cross_entropy',
            # 'objective': 'multiclass',
            # 'num_class': '3',
            'save_binary': True,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'boosting_type': 'gbdt',
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            'metric': {'cross_entropy','auc'},
            # 'metric': {'multi_logloss','auc'},
            'num_threads': 40}


        model = lgb.train(params, train_set=train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100) #fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)
        # X_train_pred += model.predict(X_train_, num_iteration=model.best_iteration) / kf.n_splits
        _X_train_pred += model.predict(X_train_, num_iteration=model.best_iteration)
        X_train_pred = _X_train_pred /kf.n_splits
        # fpr_train, tpr_train, thresholds_train = roc_auc_score(x_val, y_val)
        # gmeans_train = sqrt(tpr_train * (1 - fpr_train))
        # ix_train = argmax(gmeans_train)
        # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
        #
        # thresholds_point_train = thresholds_train[ix_train]
        # x_val_thresholds = [1 if y > thresholds_point_train else 0 for y in x_val]
        score = roc_auc_score(X_train_target_, X_train_pred)

        # score = bayesian_ic_lgbm(X_train_pred, X_train_target)

        return score

bounds_LGB = {
    'colsample_bytree': (0.7, 1),
    'n_estimators': (500, 10000),
    'num_leaves': (31, 500),
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.001, 0.3),
    'min_child_weight': (0.00001, 0.01),
    'min_child_samples': (2, 100),
    'min_split_gain': (0.1, 1),
    'subsample': (0.7, 1),
    'reg_alpha': (1, 2),
    'reg_lambda': (1, 2),
    'max_depth': (-1, 50),
    # 'scale_pos_weight':(0.5, 10)
}
# bounds_LGB = {k: tuple(v) for k, v in bounds_LGB.items()}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

# LGB_BO.max['target']
# LGB_BO.max['params']

# first model
def lightgbm_model(X_train, X_train_target,X_train_, X_train_target_, X_test, X_test_target, X_test_, X_test_target_, LGB_BO):

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
    features = train_.iloc[:, 5:90].columns
    features = list(features)



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
        _y_pred += model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = _y_pred/ kf.n_splits
        # y_pred_ += model.predict(X_test_, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_ += model.predict(X_test_, num_iteration=model.best_iteration)
        y_pred_ = _y_pred_ / kf.n_splits
        # y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train += model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_train = _y_pred_train / kf.n_splits
        # y_pred_train_ += model.predict(X_train_, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train_ += model.predict(X_train_, num_iteration=model.best_iteration)
        y_pred_train_ = _y_pred_train_/ kf.n_splits

        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list, y_pred_, y_pred_train_

y_pred, y_pred_train, model_list,y_pred_,y_pred_train_= lightgbm_model(X_train=X_train, X_train_target=X_train_target, X_train_=X_train_, X_train_target_=X_train_target_,X_test=X_test, X_test_target=X_test_target,
                                                  X_test_=X_test_, X_test_target_=X_test_target_,LGB_BO=LGB_BO)

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
first_yhat_train, first_yhat = first_model_train_test(X_train_target_, y_pred_train_, X_test_target_, y_pred_)
print("first_model训练集表现：")
print(classification_report(first_yhat_train,X_train_target_))
print("first_model测试集表现：")
print(classification_report(first_yhat,X_test_target_))
print('AUC:', metrics.roc_auc_score(first_yhat,X_test_target_))
#
test_data = test.copy()
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred,columns=['predict'])
predict['closetime'] = test_data['closetime']
predict['target'] = test_data['target']
# train_data = train.copy()
# train_data = train_data.reset_index(drop=True)
# train_predict = pd.DataFrame(y_pred_train,columns=['predict'])
# train_predict['closetime'] = train_data['closetime']
# train_predict['target'] = train_data['target']
# train_predict = train_predict.iloc[-9000:,:]
# final_predict = pd.concat([train_predict, predict],axis=0)
# final_predict = final_predict.sort_values(by='closetime', ascending=True)
# #
# from numba import njit
# predict['pctrank'] = predict.index.map(lambda x : predict.loc[:x].predict.rank(pct=True)[x])
# def pctrank(x):
#     n = len(x)
#     temp = x.argsort()
#     ranks = np.empty(n)
#     ranks[temp] = (np.arange(n) + 1) / n
#     return ranks[-1]
#
# final_predict['pctrank'] = final_predict['predict'].rolling(8000).apply(pctrank, engine='numba', raw=True)
# final_predict['datetime'] = pd.to_datetime(final_predict['closetime']+28800000, unit='ms')
# final_predict = final_predict[(final_predict.datetime >= time_1)&(final_predict.datetime <= time_2)]
# #
# df_1 = final_predict.loc[final_predict['pctrank']>0.9]
# df_0 = final_predict.loc[final_predict['pctrank']<0.1]
# print(len(df_1))
# print(len(df_0))

# df_1 = predict.loc[predict['predict']>np.percentile(y_pred_train_[-23000:], 95)]
# df_0 = predict.loc[predict['predict']<np.percentile(y_pred_train_[-23000:], 5)]
df_1 = predict.loc[predict['predict']>=np.percentile(y_pred_train, 90)]
df_0 = predict.loc[predict['predict']<=np.percentile(y_pred_train, 10)]
print(len(df_1))
print(len(df_0))
df_1['side'] = 'buy'
df_0['side'] = 'sell'
final_df = pd.concat([df_1, df_0], axis=0)
final_df = final_df.sort_values(by='closetime', ascending=True)
final_df = final_df.reset_index(drop=True)
print(final_df.loc[:,['target','predict']].corr())
# print(final_df.loc[:,['return','predict']].corr())
del final_df
#
signal = test.reset_index()
signal['predict'] = predict['predict']
# signal_1 = signal[signal['predict']>=np.percentile(y_pred_train_[-56000:], 95)]
# signal_0 = signal[signal['predict']<=np.percentile(y_pred_train_[-56000:], 5)]
signal_1 = signal[signal['predict']>=np.percentile(y_pred_train, 90)]
signal_0 = signal[signal['predict']<=np.percentile(y_pred_train, 10)]
# signal['pctrank'] = predict['pctrank']
# signal_1 = signal[signal['pctrank']>0.9]
# signal_0 = signal[signal['pctrank']<0.1]
signal_1['side'] = 'buy'
signal_0['side'] = 'sell'
signal_df = pd.concat([signal_1, signal_0],axis=0)
signal_df = signal_df.sort_values(by='closetime', ascending=True)
signal_df = signal_df.set_index('datetime')
signal_df_only = signal_df.loc[:,[vwap, 'price','predict','target','side']]
print(abs(signal_df_only['target']).describe())
# vol = 0.003
def abs_classify(y):

    if y > x:
        return 1
    else:
        return 0

train = data[data.index < time_1]
train_set = train[train_col]
train_set = train_set.iloc[:,5:90]
train_target = abs(train['target']).apply(lambda x:abs_classify(x))
# train_target = abs(train['volaility']).apply(lambda x:abs_classify(x)) #realized volaility
test_set = signal_df[train_col]
test_set = test_set.iloc[:,5:90]#65
test_target = abs(signal_df["target"]).apply(lambda x:abs_classify(x))
# test_target = abs(signal_df["volaility"]).apply(lambda x:abs_classify(x)) #realized volaility
print(signal_df['target'].apply(lambda x:abs_classify(x)).value_counts())
# print(signal_df['volaility'].apply(lambda x:abs_classify(x)).value_counts())
print(len(signal_df[signal_df['target'].apply(lambda x:abs_classify(x))==1])/len(signal_df['target'].apply(lambda x:abs_classify(x))))
# print(len(signal_df[signal_df['volaility'].apply(lambda x:abs_classify(x))==1])/len(signal_df['volaility'].apply(lambda x:abs_classify(x))))
#
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)

#
secondary_LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    secondary_LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

def secondary_lightgbm_model(X_train, X_train_target, X_test, X_test_target, LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    _y_pred = np.zeros(len(X_test_target))
    _y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []
    LGB_BO.max['params'] = secondary_LGB_BO.max['params']

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        # w = weight[fold]

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

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        # y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred += model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = _y_pred / kf.n_splits
        # y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        _y_pred_train += model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_train = _y_pred_train / kf.n_splits
        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list

secondary_y_pred, secondary_y_pred_train, secondary_model_list = secondary_lightgbm_model(X_train=X_train, X_train_target=X_train_target,
                                                                                          X_test=X_test, X_test_target=X_test_target, LGB_BO=secondary_LGB_BO)
def seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred):

    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    from numpy import sqrt, argmax
    fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, secondary_y_pred_train)
    gmeans_train = sqrt(tpr_train * (1-fpr_train))
    ix_train = argmax(gmeans_train)
    # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
    thresholds_point_train = thresholds_train[ix_train]
    secondary_yhat_train = [1 if y > thresholds_point_train else 0 for y in secondary_y_pred_train]
    # print("secondary_model训练集表现：")
    # print(classification_report(yhat_train,X_train_target))

    fpr, tpr, thresholds = roc_curve(X_test_target, secondary_y_pred)
    # fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # thresholds_point = thresholds_train[ix_train]
    secondary_yhat = [1 if y > thresholds[ix] else 0 for y in secondary_y_pred]
    # yhat = [1 if y > 0.55 else 0 for y in y_pred]
    # print("secondary_model测试集表现：")
    # print(classification_report(secondary_yhat,X_test_target))
    # print(metrics.confusion_matrix(yhat, X_test_target))
    # print('AUC:', metrics.roc_auc_score(secondary_yhat, X_test_target))
    return secondary_yhat_train, secondary_yhat
secondary_yhat_train, secondary_yhat = seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred)
print("secondary_model训练集表现：")
print(classification_report(secondary_yhat_train,X_train_target))
print("secondary_model测试集表现：")
print(classification_report(secondary_yhat,X_test_target))
print('AUC:', metrics.roc_auc_score(secondary_yhat,X_test_target))
#%% two model saving
def model_saveing(model_list,secondary_model_list,base_path, future):

    joblib.dump(model_list[0],'{}/{}/{}_lightGBM_side_0.pkl'.format(base_path,future, future))
    joblib.dump(model_list[1],'{}/{}/{}_lightGBM_side_1.pkl'.format(base_path,future,future))
    joblib.dump(model_list[2],'{}/{}/{}_lightGBM_side_2.pkl'.format(base_path,future,future))
    joblib.dump(model_list[3],'{}/{}/{}_lightGBM_side_3.pkl'.format(base_path,future,future))
    joblib.dump(model_list[4],'{}/{}/{}_lightGBM_side_4.pkl'.format(base_path,future,future))
    joblib.dump(secondary_model_list[0],'{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,future,future))
    joblib.dump(secondary_model_list[1],'{}/{}/{}_lightGBM_out_1.pkl'.format(base_path,future,future))
    joblib.dump(secondary_model_list[2],'{}/{}/{}_lightGBM_out_2.pkl'.format(base_path,future,future))
    joblib.dump(secondary_model_list[3],'{}/{}/{}_lightGBM_out_3.pkl'.format(base_path,future,future))
    joblib.dump(secondary_model_list[4],'{}/{}/{}_lightGBM_out_4.pkl'.format(base_path,future,future))
    return
base_path = '/home/xianglake/songhe/crypto_saved_model/'
model_saveing(model_list, secondary_model_list, base_path, symbol)
#%%
print('long side threshold:',np.percentile(y_pred_train, 90))
print('short side threshold:',np.percentile(y_pred_train, 10))
print('out threshold:',np.percentile(secondary_y_pred_train, 70))
#%%
out_threshold_set = [50,60,70,80,90]
# out_threshold_set = [60]
for out_threshold in out_threshold_set:
    secondary_predict = pd.DataFrame(secondary_y_pred, columns=['out'])
    # secondary_predict['out_pctrank'] = secondary_predict.index.map(lambda x : secondary_predict.loc[:x].out.rank(pct=True)[x])
    signal_df_ = signal_df.reset_index()
    signal_df_['out'] = secondary_predict['out']
    # signal_df_['out_pctrank'] = secondary_predict['out_pctrank']
    # final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train[-50000:], out_threshold)]
    final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train, out_threshold)]
    # final_df = signal_df_[signal_df_['out'] >= np.percentile(secondary_y_pred, out_threshold)]
    final_df['side_'] = np.where(final_df['side'] == 'buy', 1, -1)
    final_df['acc'] = np.where((abs(final_df['target']) >= x) & (final_df['target'] * final_df['side_'] > 0), 1, 0)
    # final_df['acc'] = np.where((abs(final_df['volaility']) >= x) & (final_df['target'] * final_df['side_'] > 0), 1, 0)
    print('--------------------'
          'out_threshold:', out_threshold)
    print((len(final_df[final_df['acc']==1])/len(final_df))*100)
    print('信号长度:',len(final_df))
    final_df = final_df.sort_values(by='closetime', ascending=True)
    final_df = final_df.loc[:,['datetime','closetime',vwap,'price','predict','target','side']]
    # final_df = final_df.loc[:, ['datetime', 'closetime', vwap, 'price', 'pctrank', 'target', 'side']]
    print(final_df.loc[:,['target','predict']].corr())
    # print(final_df.loc[:, ['target', 'pctrank']].corr())
    # print(pearsonr(final_df['target'],final_df['predict'])[0])

    final_df = final_df.dropna(axis=0)
    # final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
    # final_df_min = final_df.copy()
    # final_df_min['time'] = None
    # from datetime import datetime
    # for i in range(len(final_df_min['datetime'])):
    #     final_df_min['time'].iloc[i] = ((final_df_min['datetime'].iloc[i].replace(second=59, microsecond=999999).timestamp()-28800)*1000)
    #     if final_df_min['time'].iloc[i] % 100 == 0:
    #         final_df_min['time'].iloc[i] = (final_df_min['time'].iloc[i])-1
    # final_df_min['closetime'] = final_df_min['time'].astype('int')
    # del final_df_min['time']
    final_df = final_df.set_index('datetime')
    # final_df_min = final_df_min.set_index('datetime')
    final_df['closetime'] = final_df['closetime'].astype('int')
    # final_df_min.to_csv(
    #     '/home/xianglake/songhe/crypto_backtest/{}/binance_{}_20230101_0130_{}bar_{}_ST1.0_20230912_filter_{}_{}_99sec.csv'.format(
    #         symbol, symbol, bar, vwap, out_threshold, amount))
    final_df.to_csv(
        '/home/xianglake/songhe/crypto_backtest/{}/binance_{}_20230101_0130_{}bar_{}_ST1.0_20230926_filter_{}_{}.csv'.format(
            symbol, symbol, bar, vwap, out_threshold, amount))
 #%%
out_threshold_set = [30,]
for out_threshold in out_threshold_set:
    # out_threshold = 70
    df1 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20221212_0203_{}bar_vwap30s_ST1.0_20230717_filter_{}.csv'.format(future, future, bar, out_threshold))
    df2 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230206_0331_{}bar_vwap30s_ST1.0_20230717_filter_{}.csv'.format(future, future, bar, out_threshold))
    df3 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230403_0603_{}bar_vwap30s_ST1.0_20230717_filter_{}.csv'.format(future, future, bar, out_threshold))
    df4 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230605_0630_{}bar_vwap30s_ST1.0_20230717_filter_{}.csv'.format(future, future, bar, out_threshold))

    final_df = pd.concat([df1,df2,df3],axis=0)
    del df1,df2,df3
    final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
    final_df = final_df.set_index('datetime')
    final_df.to_csv('/root/songhe/future_backtest/{}/SHFE_{}_20221212_0630_{}bar_vwap30s_ST1.0_20230717_filter_{}.csv'.format(future, future, bar, out_threshold))
#%%
df1 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230104_0203_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df2 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230206_0304_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df3 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230306_0331_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df4 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230403_0428_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df5 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230501_0603_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
final_df = pd.concat([df1,df2,df3,df4,df5],axis=0)
del df1,df2,df3,df4,df5
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
final_df.to_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230104_0603_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
#%%
final_df = pd.read_csv('/songhe/future/AI/SHFE_{}_20220502_0527_{}bar_vwap_ST2.0_20230530_filter_{}.csv'.format(future, bar, out_threshold))
final_df = final_df.iloc[:,1:]
final_df = final_df.rename({'closetime':'timestamp'},axis='columns')
final_df['symbol'] = future

from pyarrow import Table

minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
dir_name = 'datafile/eval/songhe/{}'.format("cta_future_SHFE_{}_{}bar_vwap_20220502_0527_20230530_{}".format(future, bar, out_threshold))
pq.write_to_dataset(Table.from_pandas(df=final_df),
                    root_path=dir_name,
                    filesystem=minio, basename_template="part-{i}.parquet",
                    existing_data_behavior="overwrite_or_ignore")