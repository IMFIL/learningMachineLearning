import tensorflow as tf
import numpy as np

data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
previousAverage = tf.Variable(0.)
currentValue = tf.placeholder(tf.float32)
updateAverage = (1-alpha)*previousAverage + alpha * currentValue

init = tf.global_variables_initializer() #initializes all the global variables

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(data)):
        currentAverage = sess.run(updateAverage, feed_dict={currentValue: data[i]}) #the placeholder value needs to be injected into the value through the session
        sess.run(tf.assign(previousAverage, currentAverage))
