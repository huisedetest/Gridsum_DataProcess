# coding= utf-8
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.externals import joblib
import time

def temp():
    df = pd.read_excel('../doc/model/train_data.xlsx')
    df = df[df['KM'] >= 3]
    reg = BayesianRidge()
    reg.fit(df.ix[:, 1:], df['fuel_split'])
    joblib.dump(reg, 'FuelPredict.model')

def test():
    reg = joblib.load('FuelPredict.model')
    df = pd.read_excel('../doc/model/train_data.xlsx')
    df = df[df['KM'] >= 3]
    pre = reg.predict(df.ix[0,1:])
    print(type(pre)),pre
    assert 1==3
    df['predict'] = pre
    df.to_csv('compare.csv', index=False)

if __name__ == '__main__':
    # temp()
    test()
