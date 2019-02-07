import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

iris = pd.read_csv('dataset.csv')
iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])

X = iris.drop(labels=['Id', 'Species'], axis=1).values
y = iris.Species.values

# set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)
# dataset segmentation
# splitting the dataset as train data and test data
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]
# Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)
# Normalized processing, must be placed after the data set segmentation,
# otherwise the test set will be affected by the training set
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

# Declare the variables that need to be learned and initialization
# Define placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# There are 4 features here, weights dimension is (4, 1)
weights= tf.Variable(tf.random_normal(shape=[4, 1]))
base = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.global_variables_initializer()

# Declare the model you need to learn
Y_prdection = tf.add(tf.matmul(X, weights), base)
# Declare loss function
# Use the sigmoid cross-entropy loss function,
# first doing a sigmoid on the model result and then using the cross-entropy loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_prdection, labels=y))
# Define the learning rate， batch_size etc.
learning_rate = 0.3
batch_size = 50
iter_num = 250
# Define the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)
# Define the goal
goal = opt.minimize(loss)
# Define the accuracy
# The default threshold is 0.5, rounded off directly
prediction = tf.round(tf.sigmoid(tf.add(tf.matmul(X, weights) , base)))
# Bool into float32 type
correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
# Average
accuracy = tf.reduce_mean(correct)
# End of the definition of the model framework
# Start training model
# Define the variable that stores the result
loss_trace = []
train_acc = []
test_acc = []
# training model
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)
    for epoch in range(iter_num):
    # Generate random batch index
        batch_index = np.random.choice(len(train_X), size=batch_size)
        batch_train_X = train_X[batch_index]
        batch_train_y = np.matrix(train_y[batch_index]).T
        sess.run(goal, feed_dict={X: batch_train_X, y: batch_train_y})
        temp_loss = sess.run(loss, feed_dict={X: batch_train_X, y: batch_train_y})
    # convert into a matrix, and the shape of the placeholder to correspond
        temp_train_acc = sess.run(accuracy, feed_dict={X: train_X, y: np.matrix(train_y).T})
        temp_test_acc = sess.run(accuracy, feed_dict={X: test_X, y: np.matrix(test_y).T})
    # recode the result
        loss_trace.append(temp_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
    # output
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                         temp_train_acc, temp_test_acc))

# Visualization of the results
# accuracy
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()