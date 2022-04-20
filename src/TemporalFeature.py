class Feature:
    def __init__(self,seriesId,startPos,length,values):
        self.seriesId = seriesId     # 序列id
        self.startPos = startPos
        self.length = length     # shapelet长度
        self.values = values    # shapelet值