import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

# Artificial Data (Some Made Up Regression Data)
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, '*')
plt.show()

# Variables
m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Cost Function
error = tf.reduce_mean(y_label - (m * x_data + b))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error)

# Initialize Variables
init = tf.global_variables_initializer()

# Saving The Model
saver = tf.train.Saver()

# Create Session and Run!
with tf.Session() as sess:
    sess.run(init)

    epochs = 100

    for i in range(epochs):
        sess.run(optimizer)

    # Fetch Back Results
    final_slope, final_intercept = sess.run([m, b])

    # GO AHEAD AND SAVE IT!
    saver.save(sess,
               'F:\Python & Deep_Learning\ICPs\Deeplearning\Lesson3\my_model.ckpt')

# Evaluate Results
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()

# Loading a Model
with tf.Session() as sess:
    # Restore the model
    saver.restore(sess,
                  'my_model.ckpt')

    # Fetch Back Results
    restored_slope, restored_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)
y_pred_plot = restored_slope * x_test + restored_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()
