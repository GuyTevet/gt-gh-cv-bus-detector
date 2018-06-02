import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage, misc
import cv2
import sys
import math
import operator
import networkx as nx
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize
from numpy import unravel_index

path = '../dataset/'

class listNode():
    def __init__(self, featureVec):
        self.featureVec = featureVec

def returnTestFv(image,bboxes,fvDim):
    testFV = []
    numOfBBox = len(bboxes.bbList)
    for bboxNum in range(0, numOfBBox):
        color = bboxes.bbList[bboxNum].color
        height = bboxes.bbList[bboxNum].height
        width = bboxes.bbList[bboxNum].width
        xmin = bboxes.bbList[bboxNum].xmin
        ymin = bboxes.bbList[bboxNum].ymin
        subIm = image[ymin:ymin+height, xmin:xmin+width, :]
        subIm = Image.fromarray(subIm)
        featureVec = createFeatureVectors(subIm, fvDim)
        testFV.append(listNode(featureVec))

    return testFV

def createFeatureVectors(imRGB, fvDim):
    debugFlag = 0
    cropx = int(imRGB.size[0]/2)
    cropy = int(imRGB.size[1]/2)
    imRGB_cut = crop_center(imRGB, cropx, cropy)

    featureVector = np.zeros(fvDim)
    blackValTh = 0.3#0.2
    numOfBins = 13#30
    imHSV_cut = color.rgb2hsv(imRGB_cut)

    fv = calcFvFromHist_3D(imRGB_cut, imHSV_cut, debugFlag, fvDim, blackValTh, numOfBins, 0)
    featureVector[0:fvDim] = fv

    return featureVector

def crop_center(img, cropx, cropy):
    img = np.asarray(img)
    x = img.shape[1]
    y = img.shape[0]
    startx = int(x // 2 - (cropx // 2))
    starty = int(y // 2 - (cropy // 2))
    return img[starty:starty + cropy, startx:startx + cropx, 0:3]

def calcFvFromHist_3D(imRGB, imHSV, debugFlag, fvDim, blackValTh, numOfBins, featureNum):
    imVAL = imHSV.copy()
    imVAL[:, :, 0] = 0
    imVAL[:, :, 1] = 0
    VAL = imVAL[:, :, 2]
    VAL = np.reshape(VAL, VAL.shape[0] * VAL.shape[1])
    validIdx = (VAL > blackValTh)

    imRGBValid = imRGB.copy()
    imRGBValid = np.reshape(imRGBValid, [imRGBValid.shape[0] * imRGBValid.shape[1], imRGBValid.shape[2]])
    imRGBValid = imRGBValid[validIdx, :]

    img = np.zeros([imRGBValid.shape[0], 1, 3])
    img[:, 0, 0] = imRGBValid[:, 0]
    img[:, 0, 1] = imRGBValid[:, 1]
    img[:, 0, 2] = imRGBValid[:, 2]
    n_channels = img.shape[2]
    channels = list(range(n_channels))
    sizes = [numOfBins, numOfBins, numOfBins]
    ranges = [-1, 256] * n_channels
    img = img.astype(np.uint8)

    img = img.transpose(2, 0, 1)
    hist = cv2.calcHist(img, channels, None, sizes, ranges) #uniform hist
    sumHist = np.sum(hist)
    maxVal = np.amax(hist)
    maxIdx = unravel_index(hist.argmax(), hist.shape)
    bins = np.linspace(0, 255, numOfBins+1)
    centers = (bins[:-1] + bins[1:]) / 2
    fv = [centers[maxIdx[0]], centers[maxIdx[1]], centers[maxIdx[2]]]
    return fv

def kNNClassifier(testFV, allTrainFVs, numOfClasses, k, bboxes):
    # predictions = np.zeros(len(testFV))
    for testBboxNum in range(0, len(testFV)):
        currFv = testFV[testBboxNum].featureVec
        numOfNeighbors = np.zeros(numOfClasses)
        dist = np.zeros(len(allTrainFVs))
        for i in range(0, len(allTrainFVs)):
            dist[i] = np.linalg.norm(currFv-np.asarray(allTrainFVs[i].featureVec))

        sortIdxOfDist = np.argsort(dist)  # ascending order
        sortIdxOfDist = sortIdxOfDist[0:k] # [1:k+1] - when we are checking the train set

        for i in range(0, len(sortIdxOfDist)):
            color = allTrainFVs[sortIdxOfDist[i]].color
            numOfNeighbors[color-1] += 1

        maxIdx = np.argwhere(numOfNeighbors == np.amax(numOfNeighbors))
        if len(maxIdx) == 1:
            prediction = maxIdx[0] + 1
        else:# if there is a draw between some classes - choose the one that holds the closest feature
            foundMatchFlag = 0
            for i in range(0, len(sortIdxOfDist)):
                if foundMatchFlag:
                    break
                for j in range(0, len(maxIdx)):
                    if allTrainFVs[sortIdxOfDist[i]].color == maxIdx[j] + 1:
                        prediction = maxIdx[j] + 1
                        foundMatchFlag = 1
                        break

        # predictions[testBboxNum] = int(prediction[0])
        bboxes.bbList[testBboxNum].color = int(prediction[0])

    return bboxes

def calcAccuracy(allTrainFVs, numOfClasses, bboxes, k):
    numOfTrue = 0
    numOfFalse = 0

    for i in range(0, len(allTrainFVs)):
        currFv = np.asarray(allTrainFVs[i].featureVec)
        groundTruth = allTrainFVs[i].color
        prediction = kNNClassifier(currFv, allTrainFVs, numOfClasses, k)
        if prediction == groundTruth:
            numOfTrue += 1
            print("class =", groundTruth)
        else:
            numOfFalse += 1
            print("gT =", groundTruth, ', prediction =', prediction)

    print('true =', numOfTrue, ', false =', numOfFalse)

# only for training
def cropBBoxesFromImages(dataSetBB, fvDim):
    cropImList = []
    imList = dataSetBB.imageBBList
    imListLen = len(imList)

    for i in range(0, imListLen):
        print(i)
        image = mpimg.imread(path + imList[i].name)
        numOfBBox = len(imList[i].bbList)
        for j in range(0, numOfBBox):
            color = imList[i].bbList[j].color
            height = imList[i].bbList[j].height
            width = imList[i].bbList[j].width
            xmin = imList[i].bbList[j].xmin
            ymin = imList[i].bbList[j].ymin
            subIm = image[ymin:ymin+height, xmin:xmin+width, :]
            subIm = Image.fromarray(subIm)
            featureVec = createFeatureVectors(subIm, fvDim)
            cropImList.append(listNode(subIm, color, featureVec))

    return cropImList