# coding=utf-8
"""
===================================
Processing of raw data from SAIC
===================================
1. delete front useless lines
2. assemble files that belong to same experiment
3. delete disturbed data,which feature named 'EPTDrvngMdSwStsHSC1' equal to 2 or 0
   drop experiment which feature named 'ElecVehSysMdHSC1' exist 8 or
   numbers of 'EPTDrvngMdSwStsHSC1' as 0  exceed 10

"""
__author__ = 'WM'
import os
import re
import logging
import pandas as pd
from multiprocessing import Pool


input_folder, output_folder, id_name = 'data', 'data_clean', u'序号'.encode('gbk')


def get_file_names(directory):
    """
    return a list that contains all the file names in the directory U input

    :param directory: folder name, where U want to get file names
    """
    root_dir = u'..\{}'.format(directory)
    file_names = [filename for _, __, filename in os.walk(root_dir)][0]
    return file_names


def data_clean(df):
    """
    return a DataFrame cleaned  by the condition ,tip 3 above

    :param df: DataFrame, raw data
    """
    lsbm_0_num = (df['LSBMSPackSOC_h6HSC6'] == 0).sum()
    elec_8_num = (df['ElecVehSysMdHSC1'] == 8).sum()
    if lsbm_0_num > 10 or elec_8_num > 0:
        pass
    else:
        df = df[df['EPTDrvngMdSwStsHSC1'] == 1]
        return df


def assemble_clean_files(files):
    """
    one experiment can have more than one files, so need to merge files from the same experiment
    :param files: the files which belong to the same experiment

    """
    skip = range(14)+[15, 16]
    write_name = re.findall('(.*?)_\d+\.txt', files[0])[0]
    abs_path = '../{}/{}.csv'.format(output_folder, write_name)
    if os.path.exists(abs_path):
        print('have done')
        return
    try:
        _df = pd.DataFrame()
        for filename in files:
            df = pd.read_csv('../{}/{}'.format(input_folder, filename), sep='\t', skiprows=skip)
            _df = _df.append(df, ignore_index=True)

        _df = _df.sort_values(by=id_name)
        _df = data_clean(_df)

        if len(_df):
            write_name = re.findall('(.*?)_\d+\.txt', files[0])[0]
            abs_path = '../{}/{}.csv'.format(output_folder, write_name)
            if not os.path.exists(abs_path):
                _df.to_csv('../{}/{}.csv'.format(output_folder, write_name), index=False)
            print('finish one')
    except Exception as e:
        logging.error('file named {} come out error: {}'.format(filename, e))


class Parameter(object):

    """
    multiprocessing will have a unpickable error when all function in one class,
     so just to solve that problem by doing this temporarily
    """

    def __init__(self, _input_folder, _output_folder, _id_name):
        self.input_folder = _input_folder
        self.out_folder = _output_folder
        self.id_name = _id_name

    def callback(self):
        return self.input_folder,\
            self.out_folder,\
            self.id_name


def process():
    """
    Main function of preprocessing, which mainly contains two steps,
    one is cleaning disturb samples, the other is merge files from the same experiment
    """
    # global input_folder, output_folder, id_name
    #
    # para = Parameter(_input, output, index)
    # input_folder, output_folder, id_name = para.callback()
    file_names = get_file_names(input_folder)

    # the key to mark the experiment that files belong to
    key_file_names = map(lambda x: re.findall('(.*?)_\d+\.txt', x)[0], file_names)

    key_value = {}
    for _index, value in enumerate(file_names):
        key_file_name = key_file_names[_index]
        if key_file_name not in key_value.keys():
                key_value[key_file_name] = []
        key_value[key_file_name].append(value)
    print(key_value.values())
    pool = Pool(3)
    pool.map(assemble_clean_files, key_value.values())




if __name__ == '__main__':
    # global input_folder, output_folder, id_name
    # input_folder, output_folder, id_name = 'data', 'data_clean', u'序号'.encode('gbk')
    process()



