#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import pandas as pd

columns = ['name', 'current_year', 'select_year', 'male', 'female', 'pub_num', 'pub_alpha', 'cite_num', 'cite_alpha',
           'h_index', 'i10_index', 'sim_rank', 'pub_per_year', 'cite_per_year', 'doctoral_university_score',
           'years_away']
feature_cols = ['male', 'female', 'pub_num', 'pub_alpha', 'cite_num', 'cite_alpha', 'h_index', 'i10_index', 'sim_rank',
                'pub_per_year', 'cite_per_year', 'doctoral_university_score', 'years_away']

features = np.load('features.npy')
targets = np.load('targets.npy')

X = features.reshape([features.shape[0], -1])
trainX = X[:-100]
trainT = targets[:-100]
testX = X[-100:]
testT = targets[-100:]


def prepare_data(df, cut_year=2003,  # predicting getting fellow after this year
                 time_window=10, org='ieee', type='fellow'):
    cond = (df['type1'] == org) & (df['type2'] == type) & (df['current_year'] <= cut_year) & (
    df['select_year'] >= cut_year)  #
    years = np.arange(cut_year - time_window, cut_year + 1)
    subdf = df[columns][cond]
    names = np.unique(subdf['name'])
    #  数据需要补全: 'years_away' 重新计算
    # 补全策略：male,female， sim_rank， doctoral_university_score 直接copy
    # 其余字段，第一年之前，补0    # 第一年之后的空年，补为前一年的值

    copy_cols = ['name', 'select_year', 'male', 'female', 'sim_rank', 'doctoral_university_score']
    zero_cols = ['pub_num', 'pub_alpha', 'cite_num', 'cite_alpha', 'h_index', 'i10_index', 'pub_per_year',
                 'cite_per_year']
    features, targets = [], []
    for aname in names:
        curdf = subdf[subdf['name'] == aname].reset_index(drop=True)
        if min(curdf['current_year']) > min(years):
            nrowdict = {acpcol: curdf.iloc[0][acpcol] for acpcol in copy_cols}
            nrowdict.update({azrcol: 0 for azrcol in zero_cols})
            nrowdict['current_year'] = min(years)
            nrowdict['years_away'] = nrowdict['select_year'] - nrowdict['current_year']
            curdf = pd.concat([curdf, pd.DataFrame.from_dict([nrowdict])], ignore_index=True)
            # 前序补0，只需补第一年，后续会复制
        fy = []
        for ayear in years:
            try:
                a = np.where(curdf['current_year'] == ayear)[0][0]
            except:
                # 如有空，则复制前一年
                a = np.where(curdf['current_year'] < ayear)[0][-1]
            cyear = curdf.iloc[a][feature_cols].values
            fy.append(cyear[:-1])
        targets.append(curdf['select_year'].values[0] - cut_year)
        features.append(fy)

    features = np.array(features, dtype=np.float64)
    features[np.isnan(features)] = 0
    targets = np.array(targets)
    return features, targets


def baseline(model, trainX, trainT, testX, testT, parameters={}):
    if len(parameters) == 0:
        amodel = GridSearchCV(model)
    else:
        amodel = model(**parameters)
    amodel.fit(trainX, trainT)
    predy = amodel.predict(testX)
    error = np.mean(np.sqrt(np.square(predy - testT)))
    # print(amodel.best_params_)
    return error, predy, amodel


# er, py, m = baseline(SVR(), trainX, trainT, testX, testT, {'C': 0.001, 'kernel': 'linear'})
# print("error in average", er)
# for predicting, real in zip(py[:10], testT[:10]):
#     print("%.2f %d" % (predicting, real))


from sklearn.preprocessing import StandardScaler

trainX = np.array(trainX)
testX = np.array(testX)
trainT = np.array(trainT)
testT = np.array(testT)
sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.transform(testX)

# params = dict(hidden_layer_sizes=(6, 3), learning_rate=('constant', 'invscaling', 'adaptive'), max_iter=(1000, 2000),
#               learning_rate_init=(0.001, 0.005), activation=('tanh', 'relu', 'logistic'), alpha=(10e-4, 10e-5),
#               solver=('lbfgs', 'sgd', 'adam'))

params = dict(hidden_layer_sizes=(6, 4), learning_rate='adaptive', max_iter=2000, learning_rate_init=0.005,
              activation='logistic', alpha=10e-5, solver='adam')
er, py, m = baseline(MLPRegressor, trainX, trainT, testX, testT, params)
print("error in average", er)
for predicting, real in zip(py[:10], testT[:10]):
    print("%.2f %d" % (predicting, real))
