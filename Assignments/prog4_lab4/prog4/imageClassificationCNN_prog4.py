from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import tensorflow as tf

cifar10_dataset_folder_path = 'data/cifar-10-batches-py'

import prog4Helper
import numpy as np

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x
def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded
prog4Helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

import pickle
# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name='x')

def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')

def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob')

tf.reset_default_graph()

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    # Create filter dimensions
    filter_height, filter_width, in_channels, out_channels = \
        conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs
    conv_filter = [filter_height, filter_width, in_channels, out_channels]

    # Create weights and bias
    weights = tf.Variable(tf.truncated_normal(conv_filter, stddev=0.05))
    bias = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.05))

    # Create strides
    strides = (1, conv_strides[0], conv_strides[1], 1)

    # Bind all together to create the layer
    conv = tf.nn.conv2d(x_tensor, weights, strides, padding='SAME')
    conv = tf.nn.bias_add(conv, bias)

    # Create ksize
    ksize = (1, pool_ksize[0], pool_ksize[1], 1)

    # Create strides
    strides = (1, pool_strides[0], pool_strides[1], 1)

    pool = tf.nn.max_pool(conv, ksize, strides, padding='SAME')

    print('Convolutional layer with conv_num_outputs:', conv_num_outputs,
          'conv_ksize:', conv_ksize,
          'conv_strides:', conv_strides,
          'pool_ksize:', pool_ksize,
          'pool_strides', pool_strides)
    print('layer input shape', x_tensor.get_shape().as_list(),
          'layer output shape', pool.get_shape().as_list())

    return pool


def flatten(x_tensor):
    _, height, width, channels = x_tensor.get_shape().as_list()
    net = tf.reshape(x_tensor, shape=[-1, height * width * channels])
    print('flatten shape', net.get_shape().as_list())
    return net


def fully_conn(x_tensor, num_outputs):
    _, size = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))

    fully_connected = tf.add(tf.matmul(x_tensor, weights), bias)

    print('layer input shape', x_tensor.get_shape().as_list(),
          'layer output shape', fully_connected.get_shape().as_list())

    return fully_connected


def flatten(x_tensor):
    _, height, width, channels = x_tensor.get_shape().as_list()
    net = tf.reshape(x_tensor, shape=[-1, height * width * channels])
    print('flatten shape', net.get_shape().as_list())
    return net


def fully_conn(x_tensor, num_outputs):
    _, size = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))

    fully_connected = tf.add(tf.matmul(x_tensor, weights), bias)

    print('layer input shape', x_tensor.get_shape().as_list(),
          'layer output shape', fully_connected.get_shape().as_list())

    return fully_connected


def conv_net(input_x, keep_probability):
    net = conv2d_maxpool(input_x, 32, (7, 7), (2, 2), (2, 2), (2, 2))
    net = conv2d_maxpool(net, 64, (3, 3), (1, 1), (2, 2), (2, 2))
    net = conv2d_maxpool(net, 128, (2, 2), (1, 1), (2, 2), (2, 2))

    net = flatten(net)
    net = tf.nn.dropout(net, keep_probability)
    net = fully_conn(net, 1024)
    net = tf.nn.dropout(net, keep_probability)
    net = fully_conn(net, 128)
    net = fully_conn(net, 10)

    return net
# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    batch_loss = session.run(cost, feed_dict= \
        {x: feature_batch, y: label_batch, keep_prob: 1.0})
    batch_accuracy = session.run(accuracy, feed_dict= \
        {x: valid_features, y: valid_labels, keep_prob: 1.0})

    print('batch loss is : ', batch_loss)
    print('batch_accuracy accuracy is :', batch_accuracy)

epochs = 10
batch_size = 256

# keep probability of 70%
keep_probability = 0.5


# a couple of helper functions for loading a single batch
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)
tf.summary.scalar("loss", cost)
# Saving Accuracy
tf.summary.scalar("accuracy", accuracy)
# Merging the Summary
merged_summary = tf.summary.merge_all()
# Summary Writer
summary_writer = tf.summary.FileWriter('./graphs/scalars/', graph=tf.get_default_graph())


print('Checking the Training on a Single Batch...')
with tf.Session() as session:
    # Initializing the variables
    session.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            _, acc, loss, summary = session.run([optimizer, accuracy, cost, merged_summary],
                                                feed_dict={x: batch_features, y: batch_labels,
                                                           keep_prob: keep_probability})
            summary_writer.add_summary(summary, epoch)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(session, batch_features, batch_labels, cost, accuracy)

save_model_path = './image_classification'



print('Training...')
with tf.Session() as session:
    # Initializing the variables
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/', session.graph)

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                session.run(optimizer, feed_dict= {x: batch_features, y: batch_labels, keep_prob: keep_probability})


    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(session, save_model_path)
