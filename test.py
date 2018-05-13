#feature vectors are represented as
#tensors in tensorflow
#each tensor has a rank, the rank is
#simply the amount of nested vectors present
#in the tensor. 2x3x2 has a rank of 3.
import tensorflow as tf
import numpy as np
from math import pi

array = [[1, 2, 3], [4, 5, 6]]
npObject = np.array([[1, 2, 3], [4, 5, 6]])
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

#other objects can be made into tensors,
#as tf can only operate on tensorsself.
tf.convert_to_tensor(array, dtype=tf.float32)
tf.convert_to_tensor(npObject, dtype=tf.float32)

#tf comes with handy methods to generate
#large arrays of all 1s or 0s
#tf.ones(size,size, ...) or tf.zeros(size,size, ...)
allOnes = tf.ones([500,500]) #<- will generate an array of 500 rows
                           #with 500 1s in each row

#we can negate every element in an array with .negate
negativeTensor = tf.negative(tensor)

#Gaussian distribution in tf notation
# mean = 0
# sigma = 1
# tf.multiply(
#     tf.exp(tf.negative(tf.divide(tf.pow(tf.substract(x - mean), 2))), 2*sigma**2),
#     1/((2*sigma**2*pi))**0.5)

#tensor flow sessions are used to allocate gpu/cpu power to
#the needed computations, all calcs need to be done in a sessions

with tf.Session() as sess:
    resultOfNegTensor = sess.run(negativeTensor)


#TensorFlow utilizes:
#placeholders: an unassigned value which will be initialized by the session,
#mainly inputs and outputs of the model
#variables: a value which can change as the model goes through the
#data
#constants: a value that does not change throughout the model run

#variable declaration
variable = tf.Variable(False)
#variable needs to be initialized and ran for it to be used
varible.initializer.run()
#changes the value of the variable to True
tf.assign(variable, True)
