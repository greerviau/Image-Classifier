import tensorflow as tf

def initVariables(nInput, nClasses):
    x = tf.placeholder('float',[None,nInput,nInput,3])
    y = tf.placeholder('float',[None,nClasses])
    keepRate = tf.placeholder('float')

    return x, y, keepRate

def convoNeuralNet(x, nClasses, rate):

    #DEFINE CONVOLUTIONAL FILTER WEIGHTS
    conv1Filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2Filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3Filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=0.08))

    #x = tf.reshape(x, shape=[-1,64,64,3])

    #CONV LAYER 1
    conv1 = tf.nn.conv2d(x,conv1Filter,strides=[1,1,1,1],padding="SAME")
    conv1 = tf.nn.relu(conv1)

    #POOL LAYER 1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    pool1 = tf.layers.batch_normalization(pool1)

    #CONV LAYER 2
    conv2 = tf.nn.conv2d(pool1,conv2Filter,strides=[1,1,1,1],padding="SAME")
    conv2 = tf.nn.relu(conv2)

    #POOL LAYER 2
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    pool2 = tf.layers.batch_normalization(pool2)

    #CONV LAYER 3
    conv3 = tf.nn.conv2d(pool2,conv3Filter,strides=[1,1,1,1],padding="SAME")
    conv3 = tf.nn.relu(conv3)

    #POOL LAYER 3
    pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    pool3 = tf.layers.batch_normalization(pool3)

    #FLATTEN
    flat = tf.contrib.layers.flatten(pool3)

    #FULLY CONNECTED LAYER 1
    fc1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    fc1 = tf.layers.batch_normalization(fc1)
    fc1 = tf.nn.dropout(fc1, rate)

    #OUTPUT LAYER
    out = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=nClasses, activation_fn=None)

    return out
