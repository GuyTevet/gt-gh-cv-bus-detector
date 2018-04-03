"""
Data structs
"""


class BB(object):

    def __init__(self,bb,yolo_data=None):
        assert len(bb) == 5
        [self.xmin, self.ymin, self.width, self.height, self.color] = bb
        if yolo_data is not None:
            [self.label , self.confidence] = yolo_data

    def str(self):
        return "[%0d,%0d,%0d,%0d,%0d]" % (self.xmin, self.ymin, self.width, self.height, self.color)

class ImageBB(object):
    def __init__(self,name,bbList = None):
        self.name = name

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





