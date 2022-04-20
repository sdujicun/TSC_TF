import numpy as np
import math
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import os
from src.DataIO import loadDataFromTsv, loadTrainTranDataFromTsv, loadTestTranDataFromTsv
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


def accuracyExperiemnt(dataset='Beef', feature_number=256, nb_epochs=600, segAvgLen=10):
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
        x_train_trans = loadTrainTranDataFromTsv(dataset)
    else:
        x_train_trans = transform(features, x_train_origin)

    classifier = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    classifier.fit(x_train_trans, y_train_origin)

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    nb_classes = len(np.unique(y_test_origin))

    y_train = (y_train_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (nb_classes - 1)
    y_test = (y_test_origin - y_test_origin.min()) / (y_test_origin.max() - y_test_origin.min()) * (nb_classes - 1)
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    threshold = 0
    if (len(importances) > feature_number):
        threshold = importances[indices[feature_number - 1]]

    x_train_selected = x_train_trans[:, importances >= threshold]
    if os.path.exists('..\\trans\\' + dataset + '_TEST.tsv'):
        x_test_trans = loadTestTranDataFromTsv(dataset)
        x_test_selected = x_test_trans[:, importances >= threshold]
    else:
        selectedIndex = np.argwhere(importances >= threshold)
        featuresSelected = []
        for index in selectedIndex:
            feature = features[index[0]]
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
    accuracy = fcn.fit(x_train, x_test, Y_train, Y_test, nb_epochs=nb_epochs)
    '''
    y_pred = fcn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    '''

    return accuracy


if __name__ == '__main__':
    dataset_epoch = {'Adiac': 900, 'Beef': 900, 'CBF': 700, 'ChlorineConcentration': 1000, 'CinCECGTorso': 700,
                     'Coffee': 500, 'CricketX': 900, 'CricketY': 600, 'CricketZ': 900, 'DiatomSizeReduction': 1000,
                     'ECG200': 700, 'ECGFiveDays': 600, 'FaceAll': 1000, 'FaceFour': 500, 'FacesUCR': 600,
                     'FiftyWords': 1000, 'Fish': 500, 'GunPoint': 900, 'Haptics': 700, 'InlineSkate': 700,
                     'ItalyPowerDemand': 900, 'Lightning2': 600, 'Lightning7': 900, 'Mallat': 600, 'MedicalImages': 800,
                     'MoteStrain': 1000, 'OliveOil': 900, 'OSULeaf': 600, 'SonyAIBORobotSurface1': 900,
                     'SonyAIBORobotSurface2': 1000, 'StarLightCurves':1000, 'SwedishLeaf': 800, 'Symbols': 1000,
                     'SyntheticControl': 1000, 'Trace': 500, 'TwoLeadECG': 900, 'TwoPatterns': 800,
                     'UWaveGestureLibraryX': 700, 'UWaveGestureLibraryY': 800, 'UWaveGestureLibraryZ': 900,
                     'Wafer': 600, 'WordsSynonyms': 800, 'Yoga': 700}
    dataset_feature = {'Adiac': 64, 'Beef': 64, 'CBF': 256, 'ChlorineConcentration': 1024, 'CinCECGTorso': 512,
                       'Coffee': 64, 'CricketX': 512, 'CricketY': 512, 'CricketZ': 512, 'DiatomSizeReduction': 32,
                       'ECG200': 256, 'ECGFiveDays': 256, 'FaceAll': 256, 'FaceFour': 64, 'FacesUCR': 256,
                       'FiftyWords': 256, 'Fish': 1024, 'GunPoint': 128, 'Haptics': 512, 'InlineSkate': 1024,
                       'ItalyPowerDemand': 64, 'Lightning2': 128, 'Lightning7': 512, 'Mallat': 256,
                       'MedicalImages': 512, 'MoteStrain': 256, 'OliveOil': 256, 'OSULeaf': 1024,
                       'SonyAIBORobotSurface1': 512, 'SonyAIBORobotSurface2': 32, 'StarLightCurves': 512,
                       'SwedishLeaf': 512, 'Symbols': 1024, 'SyntheticControl': 256, 'Trace': 256, 'TwoLeadECG': 64,
                       'TwoPatterns': 256, 'UWaveGestureLibraryX': 1024, 'UWaveGestureLibraryY': 512,
                       'UWaveGestureLibraryZ': 1024, 'Wafer': 256, 'WordsSynonyms': 256, 'Yoga': 512}

    UCR_43 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY',
              'CricketZ', 'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR',
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2',
              'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace',
              'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
              'Wafer', 'WordSynonyms', 'Yoga']

    for dataset in UCR_43:
        df = pd.DataFrame(columns=['dataset', 'itr', 'accuracy'],dtype=object)
        nb_epochs = dataset_epoch[dataset]
        print(nb_epochs)
        featureNum = dataset_feature[dataset]
        print(featureNum)
        for itr in range(3):
            print('dataset', dataset)
            print('epoch', nb_epochs)
            print('itr', itr)
            print('time', datetime.datetime.now())
            accuracy = accuracyExperiemnt(dataset=dataset, feature_number=featureNum, nb_epochs=nb_epochs)
            print('accuracy', accuracy)
            print('========================================================')
            df = df.append({'dataset': dataset, 'itr': itr, 'accuracy': accuracy}, ignore_index=True)
        resultFileName = "..\\result\\accuracy_" + dataset + ".csv"
        df.to_csv(resultFileName)