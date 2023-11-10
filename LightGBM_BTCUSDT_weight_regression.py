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
min = 40
# data = read_pickle('/home/xianglake/songhe/binance/%s/tick_factor/%s_tick_factor_%s_%smin.pkl'%(symbol,symbol, amount, min))
data_2022 = read_pickle('/home/data_crypto/binance/%s/tick_factor/%s_tick_factor3_%s_%smin_2022.pkl'%(symbol,symbol, amount, min))
data_2023 = read_pickle('/home/data_crypto/binance/%s/tick_factor/%s_tick_factor3_%s_%smin_2023.pkl'%(symbol,symbol, amount, min))
data = pd.concat([data_2022, data_2023], axis=0)
del data_2022, data_2023
data = data.sort_values(by='closetime', ascending=True)

#%%
bar = 5
x = 0.001
vwap = 'vwap_2s'
data[vwap] = data[vwap].fillna(method='ffill')
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
time_1 = '2023-09-01 00:00:00'
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
train_set_ = train_set_.iloc[:,5:105] #105
train_target_ = train_["target"]

train_set = train[train_col]
train_set = train_set.iloc[:,5:105] #105
train_target = train["target"]
#
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_train_ = np.array(train_set_)
X_train_target_ = np.array(train_target_)

test_set = test[train_col]
test_set = test_set.iloc[:,5:105]
test_target = test["target"]
print(test['target'].apply(lambda x:classify(x)).value_counts())
print(len(test[test['target'].apply(lambda x:classify(x))==-1])/len(test['target'].apply(lambda x:classify(x))))
# #
test_ = test.copy()
test_['target'] = test_['target'].apply(lambda x:classify(x))
test_ = test_[~test_['target'].isin([-1])]
test_set_ = test_[train_col]
test_set_ = test_set_.iloc[:,5:105]
test_target_ = test_["target"]

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
from first_model_LighGBM import ic_lgbm, lightgbm_model, first_model_train_test
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
    weight = [0.03, 0.05, 0.07, 0.2, 0.65]
    # weight = [0.01, 0.02, 0.03, 0.04, 0.9]

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_, X_train_target_)):
        x_train, x_val = X_train_[train_index], X_train_[val_index]
        y_train, y_val = X_train_target_[train_index], X_train_target_[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 2 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)

        w = weight[fold]
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
        _X_train_pred += model.predict(X_train_, num_iteration=model.best_iteration) * w
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
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


y_pred, y_pred_train, model_list,y_pred_,y_pred_train_= lightgbm_model(train_=train_, X_train=X_train, X_train_target=X_train_target, X_train_=X_train_, X_train_target_=X_train_target_,X_test=X_test, X_test_target=X_test_target,
                                                  X_test_=X_test_, X_test_target_=X_test_target_,LGB_BO=LGB_BO)


first_yhat_train, first_yhat = first_model_train_test(X_train_target_, y_pred_train_, X_test_target_, y_pred_)
print("first_model训练集表现：")
print(classification_report(first_yhat_train,X_train_target_))
print("first_model测试集表现：")
print(classification_report(first_yhat,X_test_target_))
print('AUC:', metrics.roc_auc_score(first_yhat,X_test_target_))
#%%
signal = test.reset_index()
predict = pd.DataFrame(y_pred,columns=['predict'])
signal['predict'] = predict['predict']
# signal_1 = signal[signal['predict']>=np.percentile(y_pred_train_[-40000:], 90)]
# signal_0 = signal[signal['predict']<=np.percentile(y_pred_train_[-40000:], 10)]
signal_1 = signal[signal['predict']>=np.percentile(y_pred_train, 95)]
signal_0 = signal[signal['predict']<=np.percentile(y_pred_train, 5)]
# signal['pctrank'] = predict['pctrank']
# signal_1 = signal[signal['pctrank']>0.9]
# signal_0 = signal[signal['pctrank']<0.1]
print(len(signal_1))
print(len(signal_0))
signal_1['side'] = 'buy'
signal_0['side'] = 'sell'
signal_df = pd.concat([signal_1, signal_0],axis=0)
signal_df = signal_df.sort_values(by='closetime', ascending=True)
signal_df = signal_df.set_index('datetime')
print(signal_df.loc[:,['target','predict']].corr())
signal_df_only = signal_df.loc[:,['closetime',vwap, 'price','predict','target','side']]
print(abs(signal_df_only['target']).describe())
#%%
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False
train = data[data.index < time_1]
train_set = train[train_col]
train_set = train_set.iloc[:,5:105] #105
train_target = abs(train['target'])
test_set = signal_df[train_col]
test_set = test_set.iloc[:,5:105]#65
test_target = abs(signal_df["target"])
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)
#%%
from sklearn.metrics import mean_squared_error,r2_score
# from seondary_model_LightGBM_regression import secondary_lightgbm_model
def secondary_LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample, lambda_l1):
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
    X_train_pred = np.zeros(len(X_train_target))
    # weight = [0.03, 0.05, 0.07, 0.2, 0.65]
    # weight = [1, 1, 1, 1, 1]

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 2 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)


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
            'objective': 'regression',

            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'boosting_type': 'gbdt',


            'metric': 'rmse',
            "lambda_l1": lambda_l1,
            'num_threads': 40}


        model = lgb.train(params, train_set=train_set, num_boost_round=5000, early_stopping_rounds=200,
                          valid_sets=[train_set, val_set], verbose_eval=100) #fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)
        # X_train_pred += model.predict(X_train_, num_iteration=model.best_iteration) / kf.n_splits
        X_train_pred += model.predict(X_train, num_iteration=model.best_iteration)
        print(X_train_pred)
        # fpr_train, tpr_train, thresholds_train = roc_auc_score(x_val, y_val)
        # gmeans_train = sqrt(tpr_train * (1 - fpr_train))
        # ix_train = argmax(gmeans_train)
        # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
        #
        # thresholds_point_train = thresholds_train[ix_train]
        # x_val_thresholds = [1 if y > thresholds_point_train else 0 for y in x_val]
        score = r2_score(X_train_target, X_train_pred)

        # score = bayesian_ic_lgbm(X_train_pred, X_train_target)

        return score

secondary_bounds_LGB = {
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
    'lambda_l1': (0, 6)
}

secondary_LGB_BO = BayesianOptimization(secondary_LGB_bayesian, secondary_bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    secondary_LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
#%%
secondary_y_pred, secondary_y_pred_train, secondary_model_list = secondary_lightgbm_model(X_train=X_train, X_train_target=X_train_target,
                                                                                          X_test=X_test, X_test_target=X_test_target, secondary_LGB_BO=secondary_LGB_BO)

# two model saving
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

# model_saveing(model_list, secondary_model_list, base_path, symbol)
#
# print('95 long side threshold:',np.percentile(y_pred_train, 95))
# print('5 short side threshold:',np.percentile(y_pred_train, 5))
# print('80 out threshold:',np.percentile(secondary_y_pred_train, 80))
# print('50 out threshold:',np.percentile(secondary_y_pred_train, 50))

secondary_predict = pd.DataFrame(secondary_y_pred, columns=['out'])
# secondary_predict['out_pctrank'] = secondary_predict.index.map(lambda x : secondary_predict.loc[:x].out.rank(pct=True)[x])
signal_df_ = signal_df.reset_index()
signal_df_['out'] = secondary_predict['out']
sample_min = '10s'
out_threshold_set = [50,60,70,80]
# out_threshold_set = [50]
for out_threshold in out_threshold_set:

    # signal_df_['out_pctrank'] = secondary_predict['out_pctrank']
    # final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train[-50000:], out_threshold)]
    final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train, out_threshold)]
    # final_df = signal_df_[signal_df_['out'] >= np.percentile(secondary_y_pred, out_threshold)]
    final_df['side_'] = np.where(final_df['side'] == 'buy', 1, -1)
    final_df['acc'] = np.where((abs(final_df['target']) >= x) & (final_df['target'] * final_df['side_'] > 0), 1, 0)
    # final_df = final_df.set_index('datetime').groupby(pd.Grouper(freq=sample_min)).apply('first')
    # final_df = final_df.dropna()
    # final_df = final_df.reset_index(drop=True)
    # final_df['datetime'] = pd.to_datetime(final_df['closetime'] + 28800000, unit='ms')
    # final_df['acc'] = np.where((abs(final_df['volaility']) >= x) & (final_df['target'] * final_df['side_'] > 0), 1, 0)
    print('--------------------'
          'out_threshold:', out_threshold)
    print((len(final_df[final_df['acc']==1])/len(final_df))*100)
    print('信号长度:',len(final_df))
    final_df = final_df.sort_values(by='closetime', ascending=True)
    final_df = final_df.loc[:,['datetime','closetime',vwap,'price','predict','target','side','out']]
    # final_df = final_df.loc[:, ['datetime', 'closetime', vwap, 'price', 'pctrank', 'target', 'side']]
    print(final_df.loc[:,['target','predict']].corr())
    # print(final_df.loc[:, ['target', 'pctrank']].corr())
    # print(pearsonr(final_df['target'],final_df['predict'])[0])
    final_df = final_df.dropna(axis=0)
    final_df = final_df.set_index('datetime')
    final_df['closetime'] = final_df['closetime'].astype('int')

    final_df.to_csv(
        '/home/xianglake/songhe/crypto_backtest/{}/binance_{}_20230901_0930_{}bar_{}_ST1.0_20231108_filter_70_{}_{}.csv'.format(
            symbol, symbol, bar, vwap, out_threshold, amount))
    # final_df.to_csv(
    #     '/home/xianglake/songhe/crypto_backtest/{}/binance_{}_20230101_0130_{}bar_{}_ST1.0_20231108_filter_95_{}_{}_{}.csv'.format(
    #         symbol, symbol, bar, vwap, out_threshold, amount, sample_min))


