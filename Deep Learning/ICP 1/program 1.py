# import TensorFlow library
import tensorflow as tf

# constant matrix inputs a, b, c
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.constant([7, 8, 9, 10, 11, 12], shape=[2, 3])

# calculating (a^2+b)*c
d = tf.pow(a, 2, name='d')
e = tf.add(d,b,name='e')
f = tf.matmul(e,c,name='output')

# printing the input matrices
with tf.Session() as session:
    print("Matrix a: ")
    print(session.run(a))
    print("Matrix b: ")
    print(session.run(b))
    print("Matrix c: ")
    print(session.run(c))

# printing the output
with tf.Session() as session:
    print("Output Matrix: ")
    print(session.run(f))

