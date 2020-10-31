import tensorflow as tf
    
# takes in two models as parameters, as well as a label parameter
# the label parameter is either 0 if the inputs are from the same class, or 1 otherwise

def constrastive_loss(model1, model2, label, margin=2.0):
    distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
    similarity = y * tf.square(distance)
    dissimilarity = (1 - y) * tf.square(tf.maximum((margin-distance), 0))
    return tf.reduce_mean(dissimilarity + similarity) / 2