import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPool2D, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def aucMetric(true, pred):

        #We want strictly 1D arrays - cannot have (batch, 1), for instance
    true = (true - K.min(true))/(K.max(true) - K.min(true))
    pred = (pred - K.min(pred))/(K.max(pred) - K.min(pred))
    true= K.flatten(true)
    pred = K.flatten(pred)

        #total number of elements in this batch
    totalCount = K.shape(true)[0]

        #sorting the prediction values in descending order
    values, indices = tf.nn.top_k(pred, k = totalCount)   
        #sorting the ground truth values based on the predictions above         
    sortedTrue = K.gather(true, indices)

        #getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

        #the true positive count per threshold
    TPCurve = K.cumsum(sortedTrue)

        #area under the curve
    auc = K.sum(TPCurve * negatives)

       #normalizing the result between 0 and 1
    totalCount = K.cast(totalCount, K.floatx())
    positiveCount = K.sum(true)
    negativeCount = totalCount - positiveCount
    totalArea = positiveCount * negativeCount
    return  auc / totalArea

def loss_fn(y_true, y_pred):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)#K.mean(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred))
    n_pos = tf.reduce_sum(tf.cast(tf.equal(y_true[0], 1), tf.float32))
    n_neg = tf.reduce_sum(tf.cast(tf.equal(y_true[0], 0), tf.float32))
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    class_weights = tf.where(tf.equal(y_true, 1),
                                w_pos * tf.ones_like(y_true),
                                tf.ones_like(y_true))
    class_weights = tf.where(tf.equal(y_true, 0),
                                w_neg * tf.ones_like(y_true),
                                class_weights)
    loss = loss * class_weights
    loss = tf.reduce_sum(loss, [1, 2])
    loss = tf.reduce_mean(loss)
    return loss

def loss_exp_fn(inputs):
    y_true, y_pred = inputs
    product = -y_true * y_pred
    probs = 1 + K.clip(K.exp(product), 0, 1e6)
    loss = K.log(probs)
    mean_loss = K.mean(K.flatten(loss), axis=-1)
    return product
    
def loss_exp():
    return Lambda(loss_exp_fn)

Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

def conv_layer(filters, kernel_dim, stride_len):
    return [Conv2D(filters, kernel_dim, strides=stride_len,
                  padding='valid', activation='relu', kernel_initializer='glorot_normal')]

def conv_block(filters, kernel_dim, stride_len):
    batch_norm = [BatchNormalization(axis=3)]
    return conv_layer(filters, kernel_dim, stride_len) + batch_norm

def max_pool():
    return [MaxPool2D(pool_size=3, strides=2, padding='valid')]

def alex_net_layers():
    layers = []
    layers += conv_block(48, 11, 2)
    layers += max_pool()
    layers += conv_block(128, 5, 1)
    layers += max_pool()
    layers += conv_block(48, 3, 1)
    layers += conv_block(48, 3, 1)
    layers += [Conv2D(32, 3, strides=1, padding='valid', kernel_initializer='glorot_normal')]
    return layers

def apply_layers(x, layers):
    out = x
    for layer in layers:
        out = layer(out)
    return out

def add_dimension(t):
    return tf.reshape(t, (1,) + t.shape)

def cross_correlation(inputs):
    x = inputs[0]
    x = tf.reshape(x, [1] + x.shape.as_list())
    z = inputs[1]
    z = tf.reshape(z, z.shape.as_list() + [1])
    return tf.nn.convolution(x, z, padding='VALID', strides=(1,1))

def x_corr_map(inputs):
    # Note that dtype MUST be specified, otherwise TF will assert that the input and output structures are the same,
    # which they most certainly are NOT.
    return K.reshape(tf.map_fn(cross_correlation, inputs, dtype=tf.float32, infer_shape=False), shape=(-1,17,17))
    
def x_corr_layer():
    return Lambda(x_corr_map, output_shape=(17, 17))

def make_model(x_shape, z_shape, w_loss=False):
    exemplar = Input(shape=z_shape)
    search = Input(shape=x_shape)
    label_input = Input(shape=(17,17))

    alex_net = alex_net_layers()

    exemplar_features = apply_layers(exemplar, alex_net)
    search_features = apply_layers(search, alex_net)
    score_map = x_corr_layer()([search_features, exemplar_features])
    # bias = tf.get_variable('biases', [1],
    #                          dtype=tf.float32,
    #                          initializer=tf.constant_initializer(0.0, dtype=tf.float32),
    #                          trainable=True)#config['train_bias'])
    bias = tf.Variable(
        initial_value=[0.0], trainable=True,
        name="biases", dtype=tf.float32, shape=[1]
    )
    score_map = 1e-3 * score_map + bias
    # score_map = tf.keras.activations.sigmoid(score_map)
    
    outputs = [score_map]
    inputs = [search, exemplar]
    
    if w_loss:
        loss_layer = loss_exp()([label_input,score_map])
        outputs = outputs + [loss_layer]
        inputs = inputs + [label_input]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss_fn, metrics=[aucMetric])
    return model