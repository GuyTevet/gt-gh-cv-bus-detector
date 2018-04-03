import os
import sys
import git
import cv2
import urllib.request
sys.path.append('utils')

#local packages
from utils.data import *

darkFlowDir = os.path.abspath(os.path.join(os.getcwd(),'..' ,'..' , 'darkflow'))
if os.path.isdir(darkFlowDir):
    sys.path.append(darkFlowDir)
    from darkflow.net.build import TFNet

class Detector(object):

    def __init__(self,datasetBB):
        self.datasetBB = datasetBB

        #hard coded
        self.currDir = os.path.abspath(os.getcwd())
        self.darkflowDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'darkflow'))
        self.darkflowURL = 'https://github.com/thtrieu/darkflow.git'
        self.weightsURL = 'https://pjreddie.com/media/files/yolov2.weights'
        self.cfg_file = os.path.join("cfg" , "yolo.cfg")
        self.weights_file = os.path.join("yolov2.weights")
        self.positiveLabelsList = ['car','truck','bus']
        self.invalidLabel = 1 #green #FIXME - should be 0 - 1 is valid for debug perpusses
        self.networkResolution = 608 #544 #288


    def detect(self):

        #initialize
        self.gitImport()
        self.gitConfigure()

        os.chdir(self.darkflowDir)
        options = {"model": self.cfg_file , "load": self.weights_file , "threshold": 0.1}
        dataset = sorted(os.listdir(os.path.join(self.currDir , '..' , 'dataset')),key=str.lower) #FIXME - pre-read

        #remove non jpeg files
        for file in dataset :
            if file.find('.jpg') == -1 and file.find('.jpeg') == -1 and file.find('.JPG') == -1 and file.find('.JPEG') == -1 :
                dataset.remove(file)

        #load model
        tfnet = TFNet(options)

        for img in dataset:

            imageBB = ImageBB(img)

            imgcv = cv2.imread(os.path.join(self.currDir , '..' , 'dataset', img)) #FIXME - pre-read
            prediction = tfnet.return_predict(imgcv) #TODO - consider batch inferance instead of single inferance

            for darkflowBB in prediction :
                if darkflowBB['label'] in self.positiveLabelsList :

                    bb = BB([darkflowBB['topleft']['x'] ,
                             darkflowBB['topleft']['y'],
                             darkflowBB['bottomright']['x'] - darkflowBB['topleft']['x'],
                             darkflowBB['bottomright']['y'] - darkflowBB['topleft']['y'],
                             self.invalidLabel],
                            yolo_data = [darkflowBB['label'],
                             darkflowBB['confidence']])

                    if bb.confidence < 0.2 : #TODO - exit from debug mode - mabe we want to eliminate *reletively* small BBs
                        bb.color = 2 #white

                    imageBB.bbList.append(bb)

            #self.bbElimination(imageBB,debugMode=True) #TODO - exit from debug mode
            self.datasetBB.imageBBList.append(imageBB)


            print(prediction)

        os.chdir(self.currDir)

    def gitImport(self):

        if not os.path.isdir(self.darkflowDir):

            #clone
            print('cloning darkflow...')
            git.Repo.clone_from(self.darkflowURL,self.darkflowDir)
            os.chdir(self.darkflowDir)

            #setup
            print('building repository...')
            ver = sys.version_info
            assert sys.version_info >= (3,0) #assert python 3 #TODO - support python2
            sys.argv = ['setup.py','build_ext','--inplace']
            exec(open('setup.py').read())

            #download weights
            print('downloading weights...')
            if not os.path.isfile(self.weights_file):
                urllib.request.urlretrieve(self.weightsURL, self.weights_file)

            #import
            print('importing module...')
            path1 = sys.path
            sys.path.append(self.darkflowDir)
            path2 = sys.path
            from darkflow.net.build import TFNet #FIXME - not working

            os.chdir(self.currDir)

    def gitConfigure(self):

        os.chdir(self.darkflowDir)

        with open(self.cfg_file , 'r') as file :
            cfgTxt = file.read()

        #set input resolution
        cfgTxtSplit = cfgTxt.split('width=',maxsplit=1)
        cfgTxtSplit[1] = str(self.networkResolution) + cfgTxtSplit[1][3:] #assuming 3 digit resolution
        cfgTxt = 'width='.join(cfgTxtSplit)

        cfgTxtSplit = cfgTxt.split('height=',maxsplit=1)
        cfgTxtSplit[1] = str(self.networkResolution) + cfgTxtSplit[1][3:] #assuming 3 digit resolution
        cfgTxt = 'height='.join(cfgTxtSplit)

        with open(self.cfg_file , 'w') as file :
            file.write(cfgTxt)

        os.chdir(self.currDir)

    def bbElimination(self,imageBB,debugMode=False):
        """
        eliminating unnecessary BBs
        :param imageBB: ImageBB instance containing image bb prediction
        :param debugMode: @ debug mode the eliminated BBs will be colored and not eliminated
        :return: void - elimination is done on the given instance
        """

        if debugMode:
            eliminated_mark = 6 #red
        else:
            eliminated_mark = 'remove'

        #TODO - document
        #FIXME - BUG here
        imageBB.bbList = sorted(imageBB.bbList,  key=lambda bb: bb.confidence) #sort by confidence
        for bb_i in range(len(imageBB.bbList)):
            if imageBB.bbList[bb_i].color == eliminated_mark :
                continue
            else:
                for bb_j in range(bb_i+1,len(imageBB.bbList)):
                    if self.calcIOU(imageBB.bbList[bb_i] , imageBB.bbList[bb_j]) > 0.8 :
                        imageBB.bbList[bb_j].color = eliminated_mark





    def calcIOU(self,bbA, bbB):

        #convert our BB to regular representation
        boxA = [bbA.xmin,bbA.ymin,bbA.width,bbA.height]
        boxB = [bbB.xmin, bbB.ymin, bbB.width, bbB.height]

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou



