import tensorflow as tf
import cv2, os, argparse
from processData import formatImage, getCifar10Classes
from model import convoNeuralNet, initVariables

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image to classify")
args = vars(ap.parse_args())

loadPath = "./model/model.ckpt"
imageData = cv2.imread(args['image'])
inputData = formatImage(imageData)

#LOAD CUSTOM CLASSES
#classes = os.listdir('D:/Data/flowers')
#LOAD CIFAR10 CLASSES
classes = getCifar10Classes('data')

x, y, keepRate = initVariables(len(inputData), len(classes))

prediction = convoNeuralNet(x, len(classes), keepRate)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, loadPath)
    result = sess.run(tf.argmax(prediction,1), feed_dict={x: [inputData], keepRate: 1.})
    acc = sess.run(tf.reduce_max(tf.nn.softmax(prediction)), feed_dict={x: [inputData], keepRate: 1.})
    print("Prediction - {} | Confidence - {:.2f}%".format(classes[int(result)],acc*100))
