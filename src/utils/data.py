"""
Data structs
"""
import numpy as np

class BB(object):

    def __init__(self,bb,yolo_data=None):
        assert len(bb) == 5
        [self.xmin, self.ymin, self.width, self.height, self.color] = bb
        if yolo_data is not None:
            [self.label , self.confidence] = yolo_data

    def str(self):
        return "[%0d,%0d,%0d,%0d,%0d]" % (self.xmin, self.ymin, self.width, self.height, self.color)

class ImageBB(object):
    def __init__(self,name,width=3648,height=2736,bbList = None):
        self.name = name
        self.height = height
        self.width = width

        if bbList is None :
            self.bbList = []
        else:
            self.bbList = bbList

    def str(self):
        str = self.name + ':'
        bb_num = 0
        for bb in self.bbList :
            str += bb.str() + ','
            bb_num += 1
        if bb_num > 0 :
            str = str[:len(str)-1] + '\n'
        else:
            str += '[0,0,10,10,1]\n' #FIXME
        return str

    def features(self):
        features = np.zeros([2,0],dtype=np.float32)
        for bb in self.bbList :
            bbFeatures = np.expand_dims(np.array([bb.width * 1. / self.width , bb.height * 1. / self.height]),axis=1)
            features = np.concatenate((features,bbFeatures),axis=1)
        return features


class DatasetBB(object):

    def __init__(self,filePath,imageBBList=None):
        self.filePath = filePath

        if imageBBList is None :
            self.imageBBList = []
        else:
            self.imageBBList = imageBBList

    def str(self):
        str = ''
        for imageBB in self.imageBBList :
            str += imageBB.str()
        return str

    def features(self):
        features = np.zeros([2,0],dtype=np.float32)
        for image in self.imageBBList :
            features = np.concatenate((features,image.features()),axis=1)
        return features

    def save(self):
        with open(self.filePath, 'w') as file:
            file.write(self.str())

    def load(self):

        with open(self.filePath, 'r') as file:
            lines = file.readlines()


        for line in lines:
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            name ,BBs = line.split(':')
            BBs = BBs.split('],[')

            imageBB = ImageBB(name)

            for bb in BBs :
                bb = bb.replace('[', '')
                bb = bb.replace(']', '')
                bb = [int(a) for a in bb.split(',')]
                bb = BB(bb)
                imageBB.bbList.extend([bb])

            self.imageBBList.extend([imageBB])

        return





