import numpy as np
import math
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import os
from src.DataIO import loadDataFromTsv,loadTrainTranDataFromTsv,loadTestTranDataFromTsv
from src.Representation import transform
from src.Segment import getSeriesFeatures
from src.FCN import FCN

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

"""
dataset: 测试的数据集
segAvgLen：期望的平均Segment长度，默认为10
p：距离计算Lp中p值，默认为2
nb_epochs: FCN迭代次数，默认值为2000
importancesSumThreshold:选取特征的重要性和的阈值
"""

def featureNumAccExperiemnt(dataset='Beef',segAvgLen=10,nb_epochs=2000):


    x_train_origin, y_train_origin, x_test_origin, y_test_origin = loadDataFromTsv(dataset)

    features = []
    segNumber = max(math.ceil(len(x_train_origin[0]) / segAvgLen), 3)
    maxLength = max(math.ceil(len(x_train_origin[0]) / 5), 10)
    minLength = 5
    for seriesId in range(len(x_train_origin)):
        values = x_train_origin[seriesId]
        seriesFeatures = getSeriesFeatures(seriesId, values, segNumber, maxLength, minLength)
        for feature in seriesFeatures:
            features.append(feature)


    if os.path.exists('..\\trans\\' + dataset + '_TRAIN.tsv'):
        x_train_trans= loadTrainTranDataFromTsv(dataset)
    else:
        x_train_trans = transform(features, x_train_origin)

    classifier = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    classifier.fit(x_train_trans, y_train_origin)

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]


    nb_classes = len(np.unique(y_test_origin))

    y_train = (y_train_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (nb_classes - 1)
    y_test = (y_test_origin - y_test_origin.min()) / (y_test_origin.max() - y_test_origin.min()) * (nb_classes - 1)
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)


    df=pd.DataFrame(columns=['dataset','featureNum','precision','accuracy','recall'])
    for i in range(8):
        j=i+4

        featureNum=np.power(2, j)

        threshold = 0
        if (len(importances) > featureNum):
            threshold = importances[indices[featureNum-1]]

        x_train_selected = x_train_trans[:, importances >= threshold]
        if os.path.exists('..\\trans\\' + dataset + '_TEST.tsv'):
            x_test_trans= loadTestTranDataFromTsv(dataset)
            x_test_selected = x_test_trans[:, importances >= threshold]
        else:
            selectedIndex=np.argwhere(importances >= threshold)
            #selectedIndex=list(selectedIndex)
            #selectedIndex=np.array(selectedIndex)
            featuresSelected=[]
            for index in selectedIndex:
                feature=features[index[0]]
                featuresSelected.append(feature)

            x_test_selected = transform(featuresSelected, x_test_origin)

        x_train_mean = x_train_selected.mean()
        x_train_std = x_train_selected.std()
        x_train = (x_train_selected - x_train_mean) / (x_train_std)
        x_test = (x_test_selected - x_train_mean) / (x_train_std)

        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        fcn = FCN(x_train.shape[1:], nb_classes)
        accuracy=fcn.fit(x_train, x_test, Y_train, Y_test,nb_epochs=nb_epochs)
        '''
        y_pred = fcn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        '''
        print('dataset',dataset)
        print('time',datetime.datetime.now())
        print('featureNum',featureNum)
        print('accuracy', accuracy)

        df=df.append({'dataset':dataset,'featureNum':featureNum,'accuracy':accuracy},ignore_index=True)

    return df


if __name__ == '__main__':

    datasets=['MedicalImages','Mallat','TwoPatterns']
    for dataset in datasets:
        for itr in range(1):

            df=pd.DataFrame(columns=['dataset','featureNum','accuracy'])
            df_dataset=featureNumAccExperiemnt(dataset=dataset)
            df=df.append(df_dataset)
            resultFileName="..\\result\\FeatureNumAccTest_"+dataset+"_"+str(itr)+".csv"
            df.to_csv(resultFileName)