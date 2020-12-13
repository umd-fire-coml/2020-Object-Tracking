#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Flatten


# In[ ]:


def cross_correlation_fn(inputs):
    x = inputs[0]
    x = tf.reshape(x, [1] + x.shape.as_list())
    z = inputs[1]
    z = tf.reshape(z, z.shape.as_list() + [1])
    return tf.nn.convolution(x, z, padding='VALID', strides=(1,1))

def cross_correlation(inputs):
    # Note that dtype MUST be specified, otherwise TF will assert that the input and output structures are the same,
    # which they most certainly are NOT.
    return tf.map_fn(cross_correlation_fn, inputs, dtype=tf.float32, infer_shape=False)

z = tf.range(1, 16, 1.0)
z = tf.reshape(z, [1, 5, 3, 1])
print(z)
x = tf.range(1, 26, 1.0)
x = tf.reshape(x, [1, 5, 5, 1])
x_corr = cross_correlation([x, z])
print(x_corr)

