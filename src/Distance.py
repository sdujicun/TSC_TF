import sys
import numpy as np
def getMinLpDistance(subsequence, timeSeries):
    min = sys.float_info.max
    l1 = len(subsequence)
    l2 = len(timeSeries)
    for i in range(0, l2 - l1 + 1):
        dist = 0.0
        for j in range(0, l1):
            dist = dist + pow(subsequence[j] - timeSeries[i + j],2)
            if dist >= min:
                break
        if dist < min:
            min = dist
    return np.sqrt(min / l1)

