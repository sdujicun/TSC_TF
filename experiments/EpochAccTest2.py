import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import os
from src.DataIO import loadDataFromTsv, loadTrainTranDataFromTsv, loadTestTranDataFromTsv
from src.Representation import transform
from src.Segment import getSeriesFeatures
from src.FCN import FCN
import datetime

"""
dataset: 测试的数据集
segAvgLen：期望的平均Segment长度，默认为10
p：距离计算Lp中p值，默认为2
nb_epochs: FCN迭代次数，默认值为2000
importancesSumThreshold:选取特征的重要性和的阈值
"""


def epochAccExperiemnt(dataset='Beef', segAvgLen=10, nb_epochs=1000, featureNum=2048):
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
    if (len(importances) > featureNum):
        threshold = importances[indices[featureNum - 1]]

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
    # df=fcn.fitLog(x_train, x_test, Y_train, Y_test,nb_epochs=nb_epochs)
    df = fcn.fitAccLog(x_train, x_test, Y_train, Y_test, nb_epochs=nb_epochs)
    return df


if __name__ == '__main__':
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

    UCR_43 = ['UWaveGestureLibraryZ',
              'WordSynonyms', 'Yoga']
    '''
    UCR_43 = ['Adiac', 'Beef',  'ChlorineConcentration',  'CricketX',
              'CricketZ', 'DiatomSizeReduction', 'ECG200',  
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate',  'Lightning2',
              'Lightning7', 'MedicalImages',  'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2',  'SwedishLeaf', 'Symbols', 'SyntheticControl',
              'TwoLeadECG', 'UWaveGestureLibraryX',  'UWaveGestureLibraryZ',
              'WordSynonyms', 'Yoga']
    '''

    for dataset in UCR_43:
        featureNum = dataset_feature[dataset]
        print(featureNum)
        print('dataset', dataset)
        print('time', datetime.datetime.now())
        df = epochAccExperiemnt(dataset=dataset,featureNum=featureNum)
        resultFileName = "..\\result\\EpochAccTest_" + dataset + "_" + ".csv"
        df.to_csv(resultFileName)
        print('========================================================')


