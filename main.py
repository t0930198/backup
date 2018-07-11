# import models
from resnet import softmax_layer, conv_layer, residual_block

import pickle
import numpy as np
import tensorflow as tf
import os
import time

classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
super_classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                 'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                 'vehicles 1', 'vehicles 2']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def one_hot_vec(size):
    def _one_hot(label):
        vec = np.zeros(size)
        vec[label] = 1
        return vec
    return _one_hot

def load_data(isCar = False):
    x_all = []
    y_all = []
    z_all = []
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'train')
    test_path = os.path.join(os.path.dirname(__file__), 'data', 'test')
    d = unpickle(train_path)
    x_ = d['data']
#         labels = file["fine_labels"]
#         super_labels = file['coarse_labels']
    y_ = d['coarse_labels']
    z_ = d["fine_labels"]
    x_all.append(x_)
    y_all.append(y_)
    z_all.append(z_)

    d = unpickle(test_path)
    x_all.append(d['data'])
    y_all.append(d['coarse_labels'])
    z_all.append(d["fine_labels"])
    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y2 = map(one_hot_vec(20), y)
    z2 = map(one_hot_vec(100), z)
    y2 = (list(y2))
    z2 = (list(z2))
    X_train = x[0:50000,:,:,:]
    Y_train = y2[0:50000]
    Z_train = z2[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y2[50000:]
    Z_test = z2[50000:]
    
    ## isCar
    if isCar:
        cars_train = list(filter(lambda x: y[x] > 17, range(0,50000)))
        cars_test = list(filter(lambda x: y[x] > 17, range(50000,60000)))
        cars_test = np.array(cars_test)-50000
        X_train = X_train[cars_train]
        Y_train = np.array(Y_train)
        Y_train = Y_train[cars_train]
        X_test = X_test[cars_test]
        Y_test = np.array(Y_test)
        Y_test = Y_test[cars_test]
    print(len(X_train))
    print(X_train.shape)
    return (X_train, Y_train, Z_train, X_test, Y_test, Z_test)



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('batch_size', 250, 'Batch size')

X_train, Y_train, Z_train, X_test, Y_test, Z_test = load_data()
batch_size = 128
# print(Z_test[0].tolist().index(1))
X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 20])
Z = tf.placeholder("float", [None, 100])
zero_arr = np.reshape(np.zeros((1*8*8*64), dtype=np.float32), [1, 8, 8, 64])
data = tf.Variable(tf.constant(zero_arr, shape=[1, 8, 8, 64]), name='data')
seperate = tf.placeholder(tf.bool, shape=(), name="seperate")
learning_rate = tf.placeholder("float", [])


n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures
# ResNet Models
def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)
#     tf.summary.image("conv1", conv1, max_outputs=6)
    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [32, 32, 16]
#     tf.summary.image("conv2", conv2, max_outputs=6)
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]
#     tf.summary.image("conv3", conv3, max_outputs=6)
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    
    with tf.variable_scope('fc'):
        to_fc = tf.cond(tf.equal(seperate, tf.constant(False)), lambda: layers[-1], lambda: data)
        global_pool = tf.reduce_mean(to_fc, [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]
        sm = softmax_layer(global_pool, [64, 20])
        sm2 = softmax_layer(global_pool, [64, 100])
        out = tf.cond(tf.equal(seperate, tf.constant(False)), lambda: sm, lambda: sm2)
#         out = softmax_layer(global_pool, [64, 20])
        layers.append(out)

    return layers[-1], layers[-2]


depth = 32
tf.summary.image("source", X, max_outputs=6)
net, before_flat = resnet(X, depth)
# net, before_flat = resnet(X, 32)
# net, before_flat = resnet(X, 44)
# net, before_flat = resnet(X, 56)

# cross_entropy = -tf.reduce_sum(Y*tf.log(net))
# cross_entropy2 = -tf.reduce_sum(Z*tf.log(net))

opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
# ## middleware
train_op = tf.cond(tf.equal(seperate, tf.constant(False)), lambda: opt.minimize(-tf.reduce_sum(Y*tf.log(net))), lambda: opt.minimize(-tf.reduce_sum(Z*tf.log(net))))
tf.summary.scalar("cross entropy", -tf.reduce_sum(Y*tf.log(net)))
# train_op = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

score = tf.argmax(net, 1)
with tf.name_scope('accuracy'):
    correct_prediction = tf.cond(tf.equal(seperate, tf.constant(False)), lambda: tf.equal(score, tf.argmax(Y, 1)), lambda: tf.equal(score, tf.argmax(Z, 1)))
# correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/"+str(int(time.time())), graph = sess.graph)
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint("./tmp/model/")
# saver.restore(sess, os.getcwd()+"/tmp/model/model.ckpt")
if checkpoint and False:
    print("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
    flat, ans, acc = sess.run([before_flat, score, accuracy],feed_dict={
        X: X_test[0:10000],
        Y: Y_test[0:10000],
        Z: Z_test[0:10000],
        seperate: False
    })

    print(acc)
else:
    print("Couldn't find checkpoint to restore from. Starting over.")
    epoch = 30
    for j in range (epoch):
        for i in range (0, len(X_train), batch_size):
            rate = 0.1
            if j*len(X_train)+(i/batch_size) < 40000:
                rate = 0.1
            elif j*len(X_train)+(i/batch_size) < 60000:
                rate = 0.01
            elif j*len(X_train)+(i/batch_size) < 80000:
                rate = 0.001
            else:
                rate = 0.0001
            end = i+batch_size
            if end > len(X_train):
                end = len(X_train)
            feed_dict={
                X: X_train[i:end], 
                Y: Y_train[i:end],
                Z: Z_train[i:end],
                seperate: False,
                learning_rate: rate
            }
            sess.run([train_op], feed_dict=feed_dict)
            if i % (len(X_train)/10) == 0:
                print("training on image #%d in epoch %d" % (i, j))
        feed_dict = {
            X: X_test[0:10000],
            Y: Y_test[0:10000],
            Z: Z_test[0:10000],
            seperate: False
        }
        result = sess.run(merged, feed_dict)
        writer.add_summary(result, j+1)
    saver.save(sess, os.getcwd()+"/tmp/model/model"+str(depth)+".ckpt")

sess.close()

