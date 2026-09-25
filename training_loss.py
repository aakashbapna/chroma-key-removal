"""Confidence-weighted alpha loss shared by current training runs."""
import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
def alpha_loss(y,p):
    alpha=y[...,:1];confidence=y[...,1:2]
    # Spatial boundary emphasis covers hard JPEG-contaminated outlines too.
    hi=tf.nn.max_pool2d(alpha,5,1,'SAME');lo=-tf.nn.max_pool2d(-alpha,5,1,'SAME')
    weight=confidence*(1+4*(hi-lo)+2*tf.cast((alpha>.02)&(alpha<.98),tf.float32))
    return tf.reduce_sum(weight*(tf.square(alpha-p)+.2*tf.abs(alpha-p)))/tf.maximum(tf.reduce_sum(weight),1.)

@tf.keras.utils.register_keras_serializable()
def alpha_mae(y,p):return tf.reduce_mean(tf.abs(y[...,:1]-p))
