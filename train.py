import tensorflow as tf
import numpy as np
import cv2, os, random
from processData import getData, getCifar10Data
from model import convoNeuralNet, initVariables

if __name__ == "__main__":

    #LOAD CUSTOM DATASET
    #xTrain, yTrain, xTest, yTest = getData(datasetPath='D:/Data/flowers')
    #LOAD CIFAR10 DATASET
    xTrain, yTrain, xTest, yTest = getCifar10Data('data')

    #INIT VARIABLES
    nInput = len(xTrain[0])
    nClasses = len(yTrain[0])

    x, y, keepRate = initVariables(nInput, nClasses)

    batchSize = 32
    learningRate = 0.001
    hmEpochs = 50

    #DEFINE FUNCTIONS
    prediction=convoNeuralNet(x, nClasses, keepRate)
    cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)))
    optimizer=tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()
    tfLog = 'tf.log'

    #START SESSION
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, './checkpoints/checkpoint.ckpt')
        totalAccuracy = []
        totalLoss = []
        for epoch in range(hmEpochs):
            print("\nEpoch: {}/{}".format(epoch+1,hmEpochs))

            #TRAINING
            loss = []
            acc = []
            batches = len(xTrain)//batchSize
            batch = 0
            for i in range(0,len(xTrain),batchSize):
                xBatch = np.array(xTrain[i:i+batchSize])
                yBatch = np.array(yTrain[i:i+batchSize])
                #OPTIMIZIZE FOR BATCH
                sess.run(optimizer, feed_dict={x: xBatch, y: yBatch, keepRate: 5.})

                #GET LOSS AND ACCURACY FOR BATCH
                l, a = sess.run([cost, accuracy], feed_dict={x: xBatch, y: yBatch, keepRate: 1.})
                loss.append(l)
                acc.append(a)
                avgLoss = sum(loss)/len(loss)
                avgAcc = sum(acc)/len(acc)

                print("\rBatch: {:4d}/{}".format(batch,batches)+" - loss: " + "{:7.3f}".format(avgLoss) + " - acc: " + "{:.3f}".format(avgAcc),end="")
                batch+=1

            #VALIDATION (SPLIT INTO BATCHES FOR MEMORY LIMITATIONS)
            valBatchSize = int(len(xTest)*0.1)
            valLoss = []
            valAcc = []
            for t in range(0,len(xTest),valBatchSize):
                xBatch = np.array(xTest[t:t+valBatchSize])
                yBatch = np.array(yTest[t:t+valBatchSize])

                #GET LOSS AND ACCURACY FOR BATCH
                vl, va = sess.run([cost, accuracy], feed_dict={x: xBatch, y: yBatch, keepRate: 1.})
                valLoss.append(vl)
                valAcc.append(va)

            #CALCULATE AVERAGE LOSS AND ACCURACY FOR VALIDATION
            avgValLoss = sum(valLoss)/len(valLoss)
            avgValAcc = sum(valAcc)/len(valAcc)
            totalAccuracy.append(avgValAcc)
            totalLoss.append(avgValLoss)
            print(" - valLoss: " + "{:.3f}".format(avgValLoss) + " - valAcc: " + "{:.3f}".format(avgValAcc))

            #CHECKPOINT
            if epoch != 0 and epoch % 10 == 0:
                checkPath = "{}/checkpoints/checkpoint.ckpt".format(os.getcwd())
                saver.save(sess, checkPath)
                print("Checkpoint Saved - {}".format(checkPath))

            #SHUFFLE TRAINING DATA
            zipped = list(zip(xTrain,yTrain))
            random.shuffle(zipped)
            xTrain[:], yTrain[:] = zip(*zipped)

        #SAVE FINAL MODEL
        highestAcc = max(totalAccuracy)*100
        lowestLoss = min(totalLoss)
        print("\nHighest Accuracy: {:.3f}% - Lowest Loss: {:.3f}".format(highestAcc,lowestLoss))
        savePath = "{}/model/model.ckpt".format(os.getcwd())
        saver.save(sess, savePath)
        print("Model saved - {}".format(savePath))
