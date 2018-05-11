import os
import sys
import git
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from shutil import copyfile
sys.path.append('utils')
import time

#local packages
from utils.data import *
import utils.visualisations as visualisations


#########################
##CHOOSE IMPLEMENTATION##
#########################
implementation='yad2k' #supporting {yad2k,darkflow} #IF YOU WANT TO CHOOSE ANOTHER IMP - DO IT HERE!
#implementation='darkflow'

#####################
##CHOOSE RESOLUTION##
#####################
global_input_width = 608  # 864 #608 #544 #288
global_input_height = 608  # 864 #608 #544 #288




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
        self.invalidLabel = 1

    def detect(self,image,imageName,debugMode=False):

        t0  = time.time()

        os.chdir(self.darkflowDir)

        if debugMode:
            self.invalidLabel = 1 #green

        prediction = tfnet.return_predict(image)

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

        self.bbElimination(imageBB,debugMode=debugMode)
        self.datasetBB.imageBBList.append(imageBB)

        if debugMode:
            print(prediction)
        t1 = time.time()
        print("detected %s in %0.4f[sec]"%(imageName,(t1 - t0)))
        os.chdir(self.currDir)
        return imageBB

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

    def __init__(self,implementation = 'yad2k'): #supporting yad2k & darkflow

        #hard coded params

        self.implementation = implementation
        self.currDir = os.path.abspath(os.getcwd())

        if implementation == 'darkflow':
            self.repDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'darkflow'))
            self.repURL = 'https://github.com/thtrieu/darkflow.git'
            self.cfg_file = os.path.join("cfg", "yolo.cfg")
        elif implementation == 'yad2k':
            self.repDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'yad2k_rep'))
            self.repURL = 'https://github.com/allanzelener/YAD2K.git'
            self.cfg_file = os.path.join("yolov2.cfg")
        else:
            raise ValueError('supporting {yad2k,darkflow} implementations at the moment')

        self.weightsURL = 'https://pjreddie.com/media/files/yolov2.weights'
        self.weights_file = os.path.join("yolov2.weights")

    def gitImport(self):

        if not os.path.isdir(self.repDir):

            # clone
            print("cloning repository [%0s]..." % self.implementation)
            git.Repo.clone_from(self.repURL, self.repDir)
            os.chdir(self.repDir)

            # setup
            if self.implementation == 'darkflow':
                print('building repository (only for darkflow implementation)...')
                ver = sys.version_info
                assert sys.version_info >= (3, 0)  # assert python 3 #TODO - support python2
                origArgv = sys.argv
                sys.argv = ['setup.py', 'build_ext', '--inplace']
                exec(open('setup.py').read())
                sys.argv = origArgv

            #some modifications
            if self.implementation == 'yad2k':
                os.rename('yad2k.py','yad2k_main.py')

            # download weights
            print('downloading weights...')
            if not os.path.isfile(self.weights_file):
                urllib.request.urlretrieve(self.weightsURL, self.weights_file)

            os.chdir(self.currDir)


    def gitConfigure(self):

        #copy cfg file
        local_cfg = 'yolov2.cfg'
        copyfile(local_cfg,os.path.join(self.repDir,self.cfg_file))

        #change configurations
        os.chdir(self.repDir)

        with open(self.cfg_file, 'r') as file:
            cfgTxt = file.read()

        # set input resolution
        cfgTxtSplit = cfgTxt.split('width=', maxsplit=1)
        cfgTxtSplit[1] = str(global_input_width) + cfgTxtSplit[1][3:]  # assuming 3 digit resolution
        cfgTxt = 'width='.join(cfgTxtSplit)

        cfgTxtSplit = cfgTxt.split('height=', maxsplit=1)
        cfgTxtSplit[1] = str(global_input_height) + cfgTxtSplit[1][3:]  # assuming 3 digit resolution
        cfgTxt = 'height='.join(cfgTxtSplit)

        with open(self.cfg_file, 'w') as file:
            file.write(cfgTxt)

        os.chdir(self.currDir)

# define consts
if implementation == 'darkflow':
    repDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'darkflow'))
    currDir = os.path.abspath(os.getcwd())
    cfg_file = os.path.join("cfg", "yolo.cfg")
    weights_file = os.path.join("yolov2.weights")
elif implementation == 'yad2k':
    repDir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'yad2k_rep'))
    currDir = os.path.abspath(os.getcwd())
    cfg_file = os.path.join("yolo.cfg")
    weights_file = os.path.join("yolov2.weights")
else:
    raise ValueError('supporting {yad2k,darkflow} implementations at the moment')

# initialize
loader = DetectorGitLoader(implementation=implementation)
loader.gitImport()
loader.gitConfigure()

#import implementation

if implementation == 'darkflow':
    from darkflowExtensions import darkflowForBusDetection
elif implementation == 'yad2k':
    from yad2kExtensions import yad2kForBusDetection

# define YOLO model:
options = {"model": cfg_file , "load": weights_file , "threshold": 0.05 , "iou_threshold": 0.5, "input_height" : global_input_height , "input_width": global_input_width}

#load model
os.chdir(repDir)
if implementation == 'darkflow':
    tfnet = darkflowForBusDetection(options)
elif implementation == 'yad2k':
    tfnet = yad2kForBusDetection(options)

os.chdir(currDir)







