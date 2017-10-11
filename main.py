# coding= utf-8
"""
=====================================================================
Main model part for SAIC data
=====================================================================

> regression model: GBRT and BayesRidge
>
"""
__author__ = 'WM'

from processing import process, get_file_names
from feature_extraction import feature_extraction
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import logging
import pandas as pd
import numpy as np
import time



class WorkFlow(object):
    """
    =================================
    workflow for SAIC data mining
    =================================
    step 1: pre_processing consist of  data clean and data merge;
    step 2: feature extraction for all the experiment, take the influence of data drop into consideration
    step 3: build a model, GBRT and BayesianRidge can be choose

    more information have be described in https://gitlab.gridsum.com/research/SAIC/issues
    """

    method_mapping = {'BayesianRidge': BayesianRidge, 'GBRT': GradientBoostingRegressor}

    def __init__(self, target='fuel_cost', scaler=False,
                 feature_selection=False, reg='BayesianRidge',
                 save_predict=True, save_train_test=False, save_feature_importance=False):

        self.target = target
        self.scaler = scaler
        self.feature_selection = feature_selection
        self.method = reg
        self.save_predict = save_predict
        self.save_train_test = save_train_test
        self.save_feature_importance = save_feature_importance

    @staticmethod
    def pre_processing():
        process()

    @staticmethod
    def feature_extraction():
        feature_extraction()

    @staticmethod
    def cal_loss(predict, test):
        result = []
        for index, value in enumerate(predict):
            result.append(abs((predict[index]-test[index])/(test[index]+1)))
        _result = np.array(result)
        return _result.mean()

    def model(self):
        """

        :return:
        """
        df = pd.read_csv('../feature/{}'.format(get_file_names('feature')[-1])).dropna(axis=0).drop_duplicates()

        target_y = df[self.target]
        df_train = df.drop(['id', 'fuel_cost', 'electric_cost', 'energy_cost'], axis=1)
        if self.scaler:
            df_train = MinMaxScaler().fit_transform(df_train)

        x_train, x_test, y_train, y_test = train_test_split(df_train, target_y, test_size=0.4, random_state=0)
        res = self.method_mapping[self.method]()
        res.fit(x_train, y_train)
        y_predict = res.predict(x_test)

        print('pearson: ', pearsonr(y_test,  y_predict)[0],
              'loss: ', self.cal_loss(y_predict, y_test.tolist()))

        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        if self.save_predict:
            _df = pd.DataFrame({'y_test': y_test, 'y_predict': y_predict})
            _df.to_csv('../result/model_{}_{}.csv'.format(self.method, time_now), index=False)

        if self.save_train_test:
            x_train[self.target] = target_y
            x_test[self.target] = y_test
            x_train.to_csv('../result/{}_train_{}.csv'.format(self.method, time_now), index=False)
            x_test.to_csv('../result/{}_test_{}.csv'.format(self.method, time_now), index=False)

        if self.save_feature_importance:
            if self.method == 'BayesianRidge':
                logging.error('Only GBRT have the property of feature importance, not BayesianRidge!! ')
            else:
                df_train[self.target] = target_y
                _df = pd.DataFrame({'feature': df_train.columns, 'importance': res.feature_importances_})
                _df.to_csv('../result/{}_feature_importance_{}.csv'.format(self.method, time_now),
                           index=False)



def main(_pre_processing, _feature_extraction,
         target='fuel_cost', scaler=False,
         reg='BayesianRidge', save_predict=True,
         save_train_test=False, save_feature_importance=True):
    """
    Main function for the workflow,
    including  pre_processing, feature extraction and model

    """
    workflow = WorkFlow(target=target, scaler=scaler,reg=reg,
                        save_predict=save_predict, save_train_test=save_train_test,
                        save_feature_importance=save_feature_importance)
    if _pre_processing:
        workflow.pre_processing()
    if _feature_extraction:
        workflow.feature_extraction()
    workflow.model()

if __name__ == '__main__':
    main(_pre_processing=False, _feature_extraction=False, target='energy_cost', reg='GBRT')
    pass
    pass
    pass









