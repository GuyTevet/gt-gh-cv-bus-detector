import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

#local
from data import *

def darkflowDebug(image,name,darkflowPrediction,debugPath='../debug'):
    positiveLabelsList = ['car', 'truck', 'bus']

    if not os.path.isdir(debugPath):
        os.mkdir(debugPath)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bb in darkflowPrediction:
        if bb['label'] in positiveLabelsList:
            color = 'green'
        elif bb['label'] == 'cell phone' :
            color = 'red'
        else:
            color = 'blue'

        rect = patches.Rectangle((bb['topleft']['x'],bb['topleft']['y']), bb['bottomright']['x'] - bb['topleft']['x'],
                             bb['bottomright']['y'] - bb['topleft']['y'] , linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(bb['topleft']['x'], bb['topleft']['y'], "%0s - %0.3f" %(bb['label'],bb['confidence']),
                color='black', fontsize=6 , bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})

    plt.savefig(os.path.join(debugPath, name), dpi=500)
    plt.close()

def detectorFeaturesDistribution():
    annFileNameGT = os.path.join(os.getcwd(), '..','..', 'ground_truth', 'annotationsTrain.txt')

    GT = DatasetBB(annFileNameGT)
    GT.load()
    features = GT.features()
    np.save('detectorFeatureVectors',features)
    print(features)
    print(features.shape)
    plt.figure()
    for i in range(features.shape[1]):
        plt.scatter(features[0,i], features[1,i], c='red', marker='o')
    plt.savefig('features.png')
    plt.close()

if __name__ == "__main__":
    detectorFeaturesDistribution()
