import numpy as np
from src.TemporalFeature import Feature

class Line():
    def __init__(self,startPos,values,errors):
        self.startPos=startPos
        self.length = len(values)
        self.values=values
        self.errors=errors
        self.weight=max(self.errors)


    def __eq__(self, other):
        return (self.weight == other.weight)

    def __gt__(self, other):
        return (self.weight> other.weight)

    def __lt__(self, other):
        return (self.length < other.length)


    def split(self):
        maxIndex=np.argmax(self.errors)
        return self.startPos+maxIndex


def getSeriesFeatures(seriesId,values,segNum,maxLength,minLength=5):
    seriesFeatures = []

    tempSegNum=1
    lines = []
    errors=calErrors(values)
    line=Line(0, values,errors)
    lines.append(line)
    if(line.length<=maxLength):
        feture= Feature(seriesId,line.startPos,line.length,np.copy(line.values))
        seriesFeatures.append(feture)

    while(len(lines) > 0 and tempSegNum < segNum):
        lines = sorted(lines)
        tempLine = lines.pop()
        pipIndex = tempLine.split()

        tempSegNum = tempSegNum + 1

        if pipIndex-tempLine.startPos+1>=minLength:
            lineValue=values[tempLine.startPos:pipIndex+1]
            lineError=calErrors(lineValue)
            line=Line(tempLine.startPos,lineValue,lineError)
            lines.append(line)
            if(line.length<=maxLength):
                feture= Feature(seriesId,line.startPos,line.length,np.copy(line.values))
                seriesFeatures.append(feture)

        if(tempLine.startPos+tempLine.length-pipIndex>=5):
            lineValue=values[pipIndex:tempLine.startPos+tempLine.length]
            lineError = calErrors(lineValue)
            line = Line(pipIndex, lineValue, lineError)
            lines.append(line)
            if(line.length<=maxLength):
                feture= Feature(seriesId,line.startPos,line.length,np.copy(line.values))
                seriesFeatures.append(feture)

    return seriesFeatures


'''   
#Identification of Perceptually Important Points
def identifyPIPs(values,num):

    selected=np.zeros(len(values),dtype=int)

    selected[0] = 1
    selected[len(values)-1] = 1
    selectedNum = 2

    lines = []

    errors=calErrors(values)
    line=Line(0, values,errors)
    if (line.length >= 5):
        lines.append(line)
    while(len(lines) > 0 and selectedNum < num):
        lines = sorted(lines)
        tempLine = lines.pop()
        pipIndex = tempLine.split()

        selected[pipIndex] = 1
        selectedNum = selectedNum + 1

        if pipIndex-tempLine.startPos+1>=5:
            lineValue=values[tempLine.startPos:pipIndex+1]
            lineError=calErrors(lineValue)
            line=Line(tempLine.startPos,lineValue,lineError)
            lines.append(line)
        if(tempLine.startPos+tempLine.length-pipIndex>=5):
            lineValue=values[pipIndex:tempLine.startPos+tempLine.length]
            lineError = calErrors(lineValue)
            line = Line(pipIndex, lineValue, lineError)
            lines.append(line)
        
    PIPs=[]
    for i in range(len(selected)):
        if(selected[i]==1):
            PIPs.append(i)

    return PIPs
'''

def calErrors(values):

    length=len(values)
    xb = 0
    yb = values[xb]
    xe = length-1
    ye = values[xe]

    errors = []
    for i in range(length):
        error=calPointError(xb,yb,xe,ye,i,values[i])
        errors.append(error)

    return errors

def calPointError(xb,yb,xe,ye,x,y):
    error=abs((ye-yb)/(xe-xb)*(x-xb)+yb-y)
    return error

if __name__ == '__main__':
    values = np.random.rand(30)
    #values=[1,2,6,8,9,5,5,3,1]
    PIPs=identifyPIPs(values,6)
    print(PIPs)