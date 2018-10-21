import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

a=tf.constant(3.0, dtype=tf.float32)
b=tf.constant(4.0)
total=a+b

sess=tf.Session()
print(sess.run(total))

w=tf.Variable(tf.random_normal([3, 2]), name='w')
b=tf.Variable(tf.random_normal([1, 2]), name='b')
x=tf.placeholder("float", [None, 3], name='x')
y=tf.sigmoid(tf.matmul(x, w)+b, 'y')

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    x_array=np.array(
        [
            [0.4, 0.2, 0.4],
            [0.3, 0.4, 0.5],
            [0.3, -0.4, 0.5]
        ]
    )
    (_b, _w, _x, _y)=sess.run((b, w, x, y), feed_dict={x:x_array})
    #print(_b)
    print(y)