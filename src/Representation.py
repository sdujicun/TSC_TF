import numpy as np
import datetime
from src.Distance import getMinLpDistance as distance

def transform(features,dataset):
    instanceNum = len(dataset)
    featureNum = len(features)
    representation = np.zeros((instanceNum, featureNum))
    for j in range(featureNum):
        values=features[j].values
        for i in range(instanceNum):
            representation[i,j]=distance(values,dataset[i])

    return representation


