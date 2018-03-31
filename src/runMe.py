import numpy as np
import ast
import os
import sys
sys.path.append('utils')

#local packages
from utils.data import *

def run(myAnnFileName, buses):
    annFileNameGT = os.path.join(os.getcwd(), '..', 'ground_truth', 'annotationsTrain.txt')

    #test
    fff = DatasetBB(annFileNameGT)
    fff.load()
    fff.filePath = myAnnFileName
    fff.save()





def runDummy(myAnnFileName, buses):
    annFileNameGT = os.path.join(os.getcwd(),'..','ground_truth','annotationsTrain.txt')
    writtenAnnsLines = {}
    annFileEstimations = open(myAnnFileName, 'w+')
    annFileGT = open(annFileNameGT, 'r')
    writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())

    for line_ in writtenAnnsLines['Ground_Truth']:

        line = line_.replace(' ','')
        imName = line.split(':')[0]
        anns_ = line[line.index(':') + 1:].replace('\n', '')
        anns = ast.literal_eval(anns_)
        if (not isinstance(anns, tuple)):
            anns = [anns]
        corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
        corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
        strToWrite = imName + ':'
        for i, ann in enumerate(corruptAnn):
            posStr = [str(x) for x in ann]
            posStr = ','.join(posStr)
            strToWrite += '[' + posStr + ']'
            if (i == int(len(anns)) - 1):
                strToWrite += '\n'
            else:
                strToWrite += ','
        annFileEstimations.write(strToWrite)