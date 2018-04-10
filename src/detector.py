import os
import sys
import git
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
sys.path.append('utils')

#local packages
from utils.data import *
import utils.visualisations as visualisations

class Detector(object):

    def __init__(self,datasetBB):
        self.datasetBB = datasetBB

        #hard coded
        self.currDir = os.path.abspath(os.getcwd())
        self.darkflowDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'darkflow'))
        self.featureVectorsFile = os.path.join(self.currDir,'detectorFeatureVectors.npy')
        self.trainFeatureVectors = np.load(self.featureVectorsFile)
        self.darkflowURL = 'https://github.com/thtrieu/darkflow.git'
        self.weightsURL = 'https://pjreddie.com/media/files/yolov2.weights'
        self.cfg_file = os.path.join("cfg" , "yolo.cfg")
        self.weights_file = os.path.join("yolov2.weights")
        self.positiveLabelsList = ['car','truck','bus']
        self.invalidLabel = 1 #invalid #FIXME

    def detect(self,image,imageName,debugMode=False):

        os.chdir(self.darkflowDir)

        if debugMode:
            self.invalidLabel = 1 #green

        prediction = tfnet.return_predict(image) #TODO - consider batch inferance instead of single inferance

        self.imheight = image.shape[0]
        self.imwidth = image.shape[1]

        imageBB = ImageBB(imageName,width=self.imwidth,height=self.imheight)

        for darkflowBB in prediction :
            for label in darkflowBB['label']:
                if label in self.positiveLabelsList :

                    bb = BB([darkflowBB['topleft']['x'] ,
                             darkflowBB['topleft']['y'],
                             darkflowBB['bottomright']['x'] - darkflowBB['topleft']['x'],
                             darkflowBB['bottomright']['y'] - darkflowBB['topleft']['y'],
                             self.invalidLabel],
                            yolo_data = [darkflowBB['label'],
                             darkflowBB['confidence']])

                    #empiricaly testing we conclude that the confidence level
                    #is not enough corelated with miss detection
                    # if bb.confidence < 0.2 :
                    #     bb.color = 2 #white

                    imageBB.bbList.append(bb)

                    break #we dont want to add the same BB twice


                # elif debugMode is True:
                #
                #     bb = BB([darkflowBB['topleft']['x'] ,
                #              darkflowBB['topleft']['y'],
                #              darkflowBB['bottomright']['x'] - darkflowBB['topleft']['x'],
                #              darkflowBB['bottomright']['y'] - darkflowBB['topleft']['y'],
                #              5], #blue
                #             yolo_data = [darkflowBB['label'],
                #              darkflowBB['confidence']])
                #
                #     #imageBB.bbList.append(bb)



        self.bbElimination(imageBB,debugMode=debugMode)
        self.datasetBB.imageBBList.append(imageBB)


        print(prediction)
        if debugMode:
            #visualisations.darkflowDebug(imgcv,img,prediction,debugPath=os.path.join(self.currDir,'..','debug'))
            pass


        os.chdir(self.currDir)


    def bbElimination(self,imageBB,debugMode=False):
        """
        eliminating unnecessary BBs
        :param imageBB: ImageBB instance containing image bb prediction
        :param debugMode: @ debug mode the eliminated BBs will be colored and not eliminated
        :return: void - elimination is done on the given instance
        """

        if debugMode:
            eliminatedMark = 6 #red
        else:
            eliminatedMark = 'remove'

        ### 1. eliminating intersecting BBs with big IOU
        ### (the more cofident BB will remain and the others will be eliminated)
        imageBB.bbList = sorted(imageBB.bbList,  key=lambda bb: 1 - bb.confidence) #sort by confidence (highest to lowest)
        for bb_i in range(len(imageBB.bbList)):
            if imageBB.bbList[bb_i].color == eliminatedMark :
                continue
            else:
                for bb_j in range(bb_i+1,len(imageBB.bbList)):
                    if self.calcIOU(imageBB.bbList[bb_i] , imageBB.bbList[bb_j]) > 0.7 :
                        imageBB.bbList[bb_j].color = eliminatedMark

        ### 2. eliminate BBs with too big and too small aspect ratio
        maxAspectRatio = 3.5 #determined empirically according the trainset
        minAspectRatio = 1. / maxAspectRatio
        for bb in imageBB.bbList:
            aspectRatio = bb.width * 1. / bb.height
            if aspectRatio > maxAspectRatio or aspectRatio < minAspectRatio :
                bb.color = eliminatedMark

        ### 3. eliminate containment pairs:
        ### detect one BB that contains or almost contains another BB - eliminate one of them using ***
        # for bb_i in range(len(imageBB.bbList)):
        #     for bb_j in range(bb_i + 1, len(imageBB.bbList)):
        #         if imageBB.bbList[bb_i].color == eliminatedMark or imageBB.bbList[bb_j].color == eliminatedMark :
        #             continue
        #         elif self.checkContainment(imageBB.bbList[bb_i],imageBB.bbList[bb_j]) == True:
        #             #imageBB.bbList[bb_i].color = 5 #blue
        #             #imageBB.bbList[bb_j].color = 5  # blue
        #
        #             #eliminate the less tipical BB
        #             BBFeatureVector_i = self.calcFeatureVector(imageBB.bbList[bb_i])
        #             BBFeatureVector_j = self.calcFeatureVector(imageBB.bbList[bb_j])
        #             typicality_i = self.calcBBTypicality(BBFeatureVector_i, self.trainFeatureVectors)
        #             typicality_j = self.calcBBTypicality(BBFeatureVector_j, self.trainFeatureVectors)
        #
        #             plt.figure()
        #             for i in range(self.trainFeatureVectors.shape[1]):
        #                 plt.scatter(self.trainFeatureVectors[0,i], self.trainFeatureVectors[1,i], s=10 ,c='red', marker='o')
        #             plt.scatter(BBFeatureVector_i[0], BBFeatureVector_i[1], s=10, c='green', marker='o')
        #             plt.scatter(BBFeatureVector_j[0], BBFeatureVector_j[1], s=10, c='blue', marker='o')
        #             plt.savefig('debug.png')
        #             plt.close()
        #
        #             #eliminate the less typical BB
        #             if typicality_i < typicality_j :
        #                 imageBB.bbList[bb_i].color = 5  # blue
        #             else:
        #                 imageBB.bbList[bb_j].color = 5  # blue


        ### 3. eliminate too big BBs
        maxWidth = 0.55
        maxHeight = 0.4
        #both determined empirically according the trainset
        for bb in imageBB.bbList:
            if  bb.height * 1. / imageBB.height > maxHeight:
                bb.color = eliminatedMark
            if  bb.width * 1. / imageBB.width > maxWidth:
                bb.color = eliminatedMark

        # REMOVE BBs
        imageBB.bbList = [bb for bb in imageBB.bbList if bb.color != 'remove']

    def calcIOU(self,bbA, bbB):

        #convert our BB to regular representation
        boxA = [bbA.xmin,bbA.ymin,bbA.xmin+bbA.width-1,bbA.ymin+bbA.height-1]
        boxB = [bbB.xmin, bbB.ymin, bbB.xmin+bbB.width-1, bbB.ymin+bbB.height-1]

        if not ((boxB[0] <= boxA[0] <= boxB[2]) or (boxA[0] <= boxB[0] <= boxA[2])): #no intersection @ x
            return 0.

        if not ((boxB[1] <= boxA[1] <= boxB[3]) or (boxA[1] <= boxB[1] <= boxA[3])): #no intersection @ y
            return 0.

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

    def checkContainment(self,bbA,bbB,reletiveMargin=0.1):
        """
        :param bbA:
        :param bbB:
        :param reletiveMargin: [0,1] reletive to the minimal height & width of the BBs
        :return: True if one of the BBs contains / almost contains
        (considering the reletive margin) the other. false o.w.
        """

        heightMargin = 1. * reletiveMargin * min(bbA.height,bbB.height)
        widthMargin = 1. * reletiveMargin * min(bbA.width, bbB.width)

        #convert our BB to regular representation
        boxA = [bbA.xmin,bbA.ymin,bbA.xmin+bbA.width-1,bbA.ymin+bbA.height-1]
        boxB = [bbB.xmin, bbB.ymin, bbB.xmin+bbB.width-1, bbB.ymin+bbB.height-1]

        if  boxB[0] <= boxA[0] + widthMargin and boxA[2] <= boxB[2] + widthMargin and boxB[1] <= boxA[1] + heightMargin and boxA[3] <= boxB[3] + heightMargin:
            return True # B contain A
        elif boxA[0] <= boxB[0] + widthMargin and boxB[2] <= boxA[2] + widthMargin and boxA[1] <= boxB[1] + heightMargin and boxB[3] <= boxA[3] + heightMargin:
            return True # A contain B
        else:
            return False

    def calcFeatureVector(self,BB):
        return np.array([BB.width * 1. / self.imwidth ,BB.height * 1. / self.imheight],dtype=np.float32)

    def calcBBTypicality(self,BBFeatureVector , trainFeatureVectors , k=5):
        """
        typicality of a bounding box is defined [1 / radius of k nearest neighbours env]
        :param BBFeatureVector: dims [n_features]
        :param trainFeatureVectors: dims [n_features x n_vectors] should be derived from trainset GT BBs
        :param k: num of neighbours to be in the env
        :return: [1 / radius of k nearest neighbours env]
        """

        #calc distanses
        nVectors = trainFeatureVectors.shape[1]
        diffVectors = trainFeatureVectors - np.tile(np.expand_dims(BBFeatureVector,axis=1),[1,nVectors]) #dims [n_features x n_vectors]
        distVector = np.linalg.norm(diffVectors,axis=0) #dims [n_vectors]

        #eliminate (k-1) closest neighbours
        for _ in range(k-1):
            i = np.argmin(distVector)
            distVector[i] = np.inf

        #choose the distance to the k'th neighbour
        radius = np.min(distVector)

        return 1. / radius


"""
Load Module:
"""

class DetectorGitLoader(object):

    def __init__(self):

        #hard coded
        self.currDir = os.path.abspath(os.getcwd())
        self.darkflowDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'darkflow'))
        self.featureVectorsFile = os.path.join(self.currDir,'detectorFeatureVectors.npy')
        self.trainFeatureVectors = np.load(self.featureVectorsFile)
        self.darkflowURL = 'https://github.com/thtrieu/darkflow.git'
        self.weightsURL = 'https://pjreddie.com/media/files/yolov2.weights'
        self.cfg_file = os.path.join("cfg" , "yolo.cfg")
        self.weights_file = os.path.join("yolov2.weights")
        self.networkResolution = 608 #864 #608 #544 #288

    def gitImport(self):
        if not os.path.isdir(self.darkflowDir):

            # clone
            print('cloning darkflow...')
            git.Repo.clone_from(self.darkflowURL, self.darkflowDir)
            os.chdir(self.darkflowDir)

            # setup
            print('building repository...')
            ver = sys.version_info
            assert sys.version_info >= (3, 0)  # assert python 3 #TODO - support python2
            origArgv = sys.argv
            sys.argv = ['setup.py', 'build_ext', '--inplace']
            exec(open('setup.py').read())
            sys.argv = origArgv

            # download weights
            print('downloading weights...')
            if not os.path.isfile(self.weights_file):
                urllib.request.urlretrieve(self.weightsURL, self.weights_file)

            os.chdir(self.currDir)


    def gitConfigure(self):
        os.chdir(self.darkflowDir)

        with open(self.cfg_file, 'r') as file:
            cfgTxt = file.read()

        # set input resolution
        cfgTxtSplit = cfgTxt.split('width=', maxsplit=1)
        cfgTxtSplit[1] = str(self.networkResolution) + cfgTxtSplit[1][3:]  # assuming 3 digit resolution
        cfgTxt = 'width='.join(cfgTxtSplit)

        cfgTxtSplit = cfgTxt.split('height=', maxsplit=1)
        cfgTxtSplit[1] = str(self.networkResolution) + cfgTxtSplit[1][3:]  # assuming 3 digit resolution
        cfgTxt = 'height='.join(cfgTxtSplit)

        with open(self.cfg_file, 'w') as file:
            file.write(cfgTxt)

        os.chdir(self.currDir)

# define consts
cfg_file = os.path.join("cfg", "yolo.cfg")
weights_file = os.path.join("yolov2.weights")

#define dirs
darkFlowDir = os.path.abspath(os.path.join(os.getcwd(),'..' ,'..' , 'darkflow'))
currDir = os.path.abspath(os.getcwd())

# initialize
loader = DetectorGitLoader()
loader.gitImport()
loader.gitConfigure()

#import darkflow
if os.path.isdir(darkFlowDir):
    from darkflowExtensions import darkflowForBusDetection

# define YOLO model:
options = {"model": cfg_file , "load": weights_file , "threshold": 0.05}

#load model
os.chdir(darkFlowDir)
tfnet = darkflowForBusDetection(options)
#tfnet = TFNet(options)
os.chdir(currDir)






