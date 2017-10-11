# coding= utf-8
"""
===============================================
Feature extraction for SAIC data
===============================================
1. vector feature: VSELongtAcc_h1HSC1
                   StrgWhlAngHSC1

2. scalar feature: EPTAccelActuPosHSC1
                   BrkPdlDrvrAppdPrs_h1HSC1
                   VehSpdAvgNonDrvn_h1HSC1
                   ACComprActuPwrHSC1
                   FuelCsumpHSC1
                   LSBMSPackSOC_h6HSC6
                   EPTDrvngMdSwStsHSC1=1


> mean and  std for above feature, and additional time for ACComprActuPwrHSC1
> pay more attention for fuel cost and electric cost  as data has been cut by EPTDrvngMdSwStsHSC1,
  which has no differences with  features by integration

> 50 experiments for 47 seconds running time
"""

__author__ = 'WM'

from processing import get_file_names,Parameter
from multiprocessing import Pool
import pandas as pd
import numpy as np
import logging
import time
import re

input_folder, output_folder, id_name = 'data_clean', 'feature', u'序号'.encode('gbk')
no_this_feature = 'missing feature'

# todo 2: self + logging + return -------done
def get_vector_feature(_df, feature):
    """
    get the mean and std of the  vector feature U input, which has direction
    ---------------------------------------------------------------------

    :param _df: DataFrame, which getting vector feature  depends on
    :param feature: the name of vector features, u want to calculate
    :return: mean and std, distinguished between global and local, both positive and negative
    """

    try:
        df = _df[feature]
        result_mean = df.mean()
        result_mean_pos = df[df >= 0].mean()
        result_mean_neg = df[df < 0].mean()
        result_std = df.std()
        result_std_pos = df[df >= 0].std()
        result_std_neg = df[df < 0].std()
        logging.info(df.mean(), df.std(), df[df >= 0].mean())
    except Exception as e:
        logging.error(e)
        result_mean = no_this_feature
        result_mean_pos = no_this_feature
        result_mean_neg = no_this_feature
        result_std = no_this_feature
        result_std_pos = no_this_feature
        result_std_neg = no_this_feature
    return result_mean, result_mean_pos, result_mean_neg,\
        result_std, result_std_pos, result_std_neg


def get_scalar_feature(_df, feature):
    """
    get the mean and std of the  scalar feature U input, which has no direction
    ---------------------------------------------------------------------------

    :param _df: DataFrame, which getting scalar features  depends on
    :param feature: the name of scalar features, u want to calculate
    :return: mean and std
    """
    try:
        df = _df[feature]
        result_mean = df.mean()
        result_std = df.std()
        logging.info(df.mean(), df.std(), df[df >= 0].mean())
    except Exception as e:
        logging.error(e)
        result_mean = no_this_feature
        result_std = no_this_feature
    return result_mean, result_std


def speed_integration(_df):
    """
    calculate the KM by integrating the speed, which frequency is 10hz

    :param _df:
    :return:
    """
    try:
        df = _df['VehSpdAvgNonDrvn_h1HSC1']
        result = df.sum()/36000
    except Exception as e:
        logging.error(e)
        result = no_this_feature
    return result


def accompractupwrhsc1_time(_df):
    """
    calculate the time air-conditioner
    """
    try:
        df = _df['ACComprActuPwrHSC1']
        result = (df > 0).sum()*0.1
    except Exception as e:
        logging.error(e)
        result = no_this_feature
    return result


def split_df(df):
    """
     to simplify the calculation by splitting DataFrame,
     for those data that is dropped in the middle truncation and id is not continuous

    :param df: DataFrame
    :return: a few DataFrame that id of each is continuous
    """
    result = []
    id_list = df[id_name].tolist()
    for index in xrange(len(id_list)-1):
        if id_list[index+1]-id_list[index] > 1:
            result.append(df.ix[:index, :])
            df = df.ix[index+1:, :]
            if index == len(id_list)-2:
                result.append(df)

    result.append(df)
    return result


def get_fuel(df):
    """
    calculate the fuel cost, max value is 65536,
    and be careful to this state that fuel value sharply drop

    :param df: DataFrame
    :return: fuel cost
    """
    split_result = []
    try:
        _df = df.ix[:, [id_name, 'FuelCsumpHSC1']]
        _df_set = split_df(_df)
        for df in _df_set:
            if len(df.index) == 1:
                continue
            df = df['FuelCsumpHSC1'].tolist()
            df_result = np.array(df[1:]) - np.array(df[:-1])
            df_result[df_result < -60000] += 65536
            df_result[(df_result >= -60000) & (df_result < 0)] *= -1
            split_result.append(df_result.sum())

        result = sum(split_result)
    except Exception as e:
        logging.error(e)
        result = no_this_feature

    return result


# Todo 1: 加入初始电量 ------done
def get_electric(df):
    """
    get init electric and electric cost,
    init electric is important factor for fuel cost,
    and electric cost is one of target to be predicted

    :param df: DataFrame
    :return:  init electric and electric cost
    """
    split_result = []
    split_result_init = []

    try:

        _df = df.ix[:, [id_name, 'LSBMSPackSOC_h6HSC6']]
        _df_set = split_df(_df)
        for df in _df_set:
            if len(df.index) == 1:
                continue
            df = [i for i in df['LSBMSPackSOC_h6HSC6'].tolist() if i != 0]
            init = df[0]
            final = df[-1]
            split_result.append(final-init)
            split_result_init.append(init)

        result = sum(split_result)
    except Exception as e:
        logging.error(e)
        result = no_this_feature

    return split_result_init[0], result


def get_feature(filename):
    """
    the function to extract feature of a experiment

    :param filename: the file name of a experiment
    :return: the features of this experiment
    """

    result = []
    df = pd.read_csv('../{}/{}'.format(input_folder,filename))
    VSELongtAcc_h1HSC1_result = get_vector_feature(df, 'VSELongtAcc_h1HSC1')
    StrgWhlAngHSC1_result = get_vector_feature(df, 'StrgWhlAngHSC1')
    EPTAccelActuPosHSC1_result = get_scalar_feature(df, 'EPTAccelActuPosHSC1')
    BrkPdlDrvrAppdPrs_h1HSC1_result = get_scalar_feature(df, 'BrkPdlDrvrAppdPrs_h1HSC1')
    VehSpdAvgNonDrvn_h1HSC1_result = get_scalar_feature(df, 'VehSpdAvgNonDrvn_h1HSC1')
    ACComprActuPwrHSC1_result = get_scalar_feature(df, 'ACComprActuPwrHSC1')
    km = speed_integration(df)
    ACComprActuPwrHSC1_time = accompractupwrhsc1_time(df)
    fuel_cost = get_fuel(df)
    electric_cost = get_electric(df)

    [result.extend(i) for i in [VSELongtAcc_h1HSC1_result, StrgWhlAngHSC1_result, EPTAccelActuPosHSC1_result,
                                BrkPdlDrvrAppdPrs_h1HSC1_result, VehSpdAvgNonDrvn_h1HSC1_result,
                                ACComprActuPwrHSC1_result,
                                electric_cost]]

    [result.append(i) for i in [km, ACComprActuPwrHSC1_time, fuel_cost]]
    print('finish one')

    return result





def feature_extraction():
    """
    Main function to extract features for all the experiments

    :param _input: the input folder to be extracted feature
           _output: the output folder to save features

    """

    # global input_folder, output_folder, id_name
    # para = Parameter(_input, _output, _index)
    # input_folder, output_folder, id_name = para.callback()

    column_names = ['VSELongtAcc_h1HSC1_mean', 'VSELongtAcc_h1HSC1_mean_pos', 'VSELongtAcc_h1HSC1_mean_neg',
                    'VSELongtAcc_h1HSC1_std', 'VSELongtAcc_h1HSC1_std_pos', 'VSELongtAcc_h1HSC1_std_neg',
                    'StrgWhlAngHSC1_mean', 'StrgWhlAngHSC1_mean_pos', 'StrgWhlAngHSC1_mean_neg',
                    'StrgWhlAngHSC1_std', 'StrgWhlAngHSC1_std_pos', 'StrgWhlAngHSC1_std_neg',
                    'EPTAccelActuPosHSC1_mean', 'EPTAccelActuPosHSC1_std',
                    'BrkPdlDrvrAppdPrs_h1HSC1_mean', 'BrkPdlDrvrAppdPrs_h1HSC1_std',
                    'VehSpdAvgNonDrvn_h1HSC1_mean', 'VehSpdAvgNonDrvn_h1HSC1_std',
                    'ACComprActuPwrHSC1_mean', 'ACComprActuPwrHSC1_std',
                    'electric_init', 'electric_cost',
                    'KM', 'ACComprActuPwrHSC1_time', 'fuel_cost']

    pool = Pool(3)
    file_names = get_file_names(input_folder)
    _id = map(lambda x: re.findall('(.*?).csv', x)[0], file_names)
    result = pool.map(get_feature, file_names)
    df = pd.DataFrame(result, columns=column_names)
    df['id'] = _id
    df['energy_cost'] = -df['electric_cost']/100*12*0.3073+df['fuel_cost']*(10**(-6))
    time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    df.to_csv('../{}/feature_{}.csv'.format(output_folder, time_now), columns=['id']+column_names+['energy_cost'], index=False)



if __name__ == '__main__':
    # import timeit
    # print timeit.timeit('feature_extraction()', 'from __main__ import feature_extraction', number=1)
    feature_extraction()
