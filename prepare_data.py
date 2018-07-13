#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

columns = ['name', 'current_year', 'select_year', 'male', 'female', 'pub_num', 'pub_alpha', 'cite_num', 'cite_alpha',
           'h_index', 'i10_index', 'sim_rank', 'pub_per_year', 'cite_per_year', 'doctoral_university_score',
           'years_away']
feature_cols = ['male', 'female', 'pub_num', 'pub_alpha', 'cite_num', 'cite_alpha', 'h_index', 'i10_index', 'sim_rank',
                'pub_per_year', 'cite_per_year', 'doctoral_university_score', 'years_away']


def prepare_data(df, cut_year=2015,  # predicting getting fellow after this year
                 time_window=82, org='ieee', type='fellow'):
    cond = (df['type1'] == org) & (df['type2'] == type)
    subdf = df[columns][cond]
    copy_cols = ['name', 'select_year', 'male', 'female', 'sim_rank', 'doctoral_university_score']
    zero_cols = ['pub_num', 'pub_alpha', 'cite_num', 'cite_alpha', 'h_index', 'i10_index', 'pub_per_year',
                 'cite_per_year']
    features, targets = [], []

    # 训练集
    traindf = subdf[subdf['select_year'] <= cut_year]
    names = np.unique(traindf['name'])
    years = np.arange(cut_year - time_window, cut_year + 1)
    #  数据需要补全: 'years_away' 重新计算
    # 补全策略：male,female， sim_rank， doctoral_university_score 直接copy
    # 其余字段，第一年之前，补0    # 第一年之后的空年，补为前一年的值

    for aname in names:
        curdf = subdf[subdf['name'] == aname].reset_index(drop=True)
        first_pub_year = min(curdf['current_year'])
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
                a = np.where(curdf['current_year'] < ayear)[0][-1]
            cyear = curdf.iloc[a][feature_cols].values
            fy.append(cyear[:-1])
        targets.append(curdf['select_year'].values[0] - first_pub_year)  # wait_year
        features.append(fy)
    print(len(targets))

    # 测试集

    testdf = subdf[subdf['select_year'] > cut_year]
    names = np.unique(testdf['name'])
    for aname in names:
        curdf = subdf[subdf['name'] == aname].reset_index(drop=True)
        first_pub_year = min(curdf['current_year'])
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
                a = np.where(curdf['current_year'] < ayear)[0][-1]
            cyear = curdf.iloc[a][feature_cols].values
            fy.append(cyear[:-1])
        targets.append(curdf['select_year'].values[0] - first_pub_year)
        features.append(fy)

    features = np.array(features, dtype=np.float64)
    features[np.isnan(features)] = 0
    targets = np.array(targets)
    return features, targets


df = pd.read_csv("g:\\dynamic_full_member_sufficient_7.csv")
features, targets = prepare_data(df)

np.save('features_3.npy', features)
np.save('targets_3.npy', targets)
