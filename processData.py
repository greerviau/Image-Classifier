import numpy as np
import os, cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def rotateImage(img,deg):
    rows,cols,depth = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    return cv2.warpAffine(img,M,(cols,rows))

def scaleImage(img):
    return cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

def formatImage(img):
    size = 32
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    #img = img.flatten()
    return img

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getCifar10Classes(dataDir):
    metaDataDict = unpickle(dataDir + "/batches.meta")
    cifarClasses = metaDataDict[b'label_names']
    cifarClasses = np.array(cifarClasses)
    return cifarClasses

def getCifar10Data(dataDir, negatives=False):

    trainingData = []

    cifarClasses = getCifar10Classes(dataDir)

    # training data
    cifarTrainData = None
    cifarTrainLabels = []

    for i in range(1, 6):
        cifarTrainDataDict = unpickle(dataDir + "/data_batch_{}".format(i))
        if i == 1:
            cifarTrainData = cifarTrainDataDict[b'data']
        else:
            cifarTrainData = np.vstack((cifarTrainData, cifarTrainDataDict[b'data']))
        cifarTrainLabels += cifarTrainDataDict[b'labels']

    cifarTrainData = cifarTrainData.reshape([-1, 3, 32, 32])
    cifarTrainData = cifarTrainData / 255
    cifarTrainData = cifarTrainData.transpose([0, 2, 3, 1]).astype(np.float32)

    for raw, ind in zip(cifarTrainData,cifarTrainLabels):
        label = np.zeros(len(cifarClasses))
        label[ind] = 1.
        trainingData.append([raw, label])

    trainingData = np.array(trainingData)
    np.random.shuffle(trainingData)

    testData = []

    cifarTestDataDict = unpickle(dataDir + "/test_batch")
    cifarTestData = cifarTestDataDict[b'data']
    cifarTestLabels = cifarTestDataDict[b'labels']

    cifarTestData = cifarTestData.reshape([-1, 3, 32, 32])
    cifarTestData = cifarTestData / 255
    cifarTestData = cifarTestData.transpose([0, 2, 3, 1]).astype(np.float32)

    '''
    cv2.imshow('frame',cifarTestData[0])
    if cv2.waitKey(1) and 0xFF == ord('q'):
        cv2.destroyAllWindows()
    '''

    for raw, ind in zip(cifarTestData,cifarTestLabels):
        label = np.zeros(len(cifarClasses))
        label[ind] = 1.
        testData.append([raw, label])

    testData = np.array(testData)
    np.random.shuffle(testData)

    xTrain = list(trainingData[:,0])
    yTrain = list(trainingData[:,1])
    xTest = list(testData[:,0])
    yTest = list(testData[:,1])

    #print(xTrain[0].shape)
    #print(yTrain[0].shape)
    #print(xTest[0].shape)
    #print(yTest[0].shape)

    return xTrain, yTrain, xTest, yTest

def getData(datasetPath='D:/Data/CatsAndDogs/train'):
    try:
        #LOAD DATA IF IT EXISTS
        xTrain = list(np.load('train_data/xTrain.npy'))
        yTrain = list(np.load('train_data/yTrain.npy'))
        xTest = list(np.load('train_data/xTest.npy'))
        yTest = list(np.load('train_data/yTest.npy'))
    except:
        #LOAD CATEGORIES
        CATEGORIES = os.listdir(datasetPath)
        print('Categories: {}'.format(CATEGORIES))
        trainingData = []
        #FOR EVERY CATEGORY
        for category in CATEGORIES:
            categoryPath = os.path.join(datasetPath,category)
            classNum = CATEGORIES.index(category)
            #FOR EVERY IMAGE IN THE CATEGORY
            for data in tqdm(os.listdir(categoryPath)):
                try:
                    label = np.zeros(len(CATEGORIES))
                    label[classNum] = 1.
                    img = cv2.imread(os.path.join(categoryPath,data))
                    imgRaw = formatImage(img)
                    trainingData.append([imgRaw,label])
                    #USE THESE TO AUGMENT DATA
                    #imgScaled = scaleImage(imgRaw)
                    #trainingData.append([imgScaled,label])
                    img90 = rotateImage(imgRaw,90)
                    trainingData.append([img90,label])
                    #img180 = rotateImage(imgRaw,180)
                    #trainingData.append([img180,label])
                    #img270 = rotateImage(imgRaw,270)
                    #trainingData.append([img270,label])
                except Exception as e:
                    pass

        print(len(trainingData))

        #SHUFFLE THE DATA
        trainingData = np.array(trainingData)
        np.random.shuffle(trainingData)

        #SPLIT THE DATA FOR TRAINING AND TESTING
        testSize = int(len(trainingData) * 0.1)

        xTrain = trainingData[:,0][:-testSize]
        yTrain = trainingData[:,1][:-testSize]
        xTest = trainingData[:,0][-testSize:]
        yTest = trainingData[:,1][-testSize:]

        #SAVE DATA
        if not os.path.exists('train_data/'):
            os.makedirs('train_data/')
        np.save('train_data/xTrain.npy',xTrain)
        np.save('train_data/yTrain.npy',yTrain)
        np.save('train_data/xTest.npy',xTest)
        np.save('train_data/yTest.npy',yTest)

        xTrain = list(xTrain)
        yTrain = list(yTrain)
        xTest = list(xTest)
        yTest = list(yTest)

        print(xTrain[0].shape)

    return xTrain, yTrain, xTest, yTest
