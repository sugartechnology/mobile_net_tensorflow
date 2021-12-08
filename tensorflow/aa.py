import tensorflow as tf
import numpy as np


condition = tf.Variable(np.array([[True, False, False],
                                  [False, True, False],
                                  [True, True, True]]),
                        dtype=tf.bool, name='condition')


x = tf.Variable(np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0, 1, 0, 0, 0]]),
                dtype=tf.float32, name='x')

y = tf.Variable(np.array([[-10, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0, 1, 0, 0, 0]]),
                dtype=tf.float32, name='x')


print(tf.reduce_sum(x-y))
r = x[:, 0]
r = tf.where(tf.equal(r, tf.constant(1.0)),
             tf.reduce_mean(x - y), tf.constant(0.0))


# y = tf.Variable(np.array([[0, 1, 0, 4, 5, 6, 7, 8, 9]]),
#                dtype=tf.float32, name='y')


#r = tf.where(condition, x, y)


print(r)
