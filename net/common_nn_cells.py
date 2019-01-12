import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

SAVE_VARIABLES = 'save_variables'

# Allocate a variable with the specified parameters 
# specific device field

def _get_variable(name, shape, initializer, dtype= tf.float32, trainable= True):

# emmm ?
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]

# place on cpu
        with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape = shape, initializer = initializer, dtype = dtype, collections = collections, trainable = trainable)

        return var

# BatchNorm layer

# beta and gamma are trainable parameter
# when training, mean and variance are stats according input
# when infer, mean and variance are stats according all inputs
def batch_normalization(x, is_training, scope, weights_param_list, decay= 0.9, epsilon= 0.001):

        is_training = tf.constant(is_training, dtype=tf.bool)

        with tf.variable_scope(scope):

                x_shape = x.get_shape()
                params_shape = x_shape[-1:]

                axis = list(range(len(x_shape) - 1))

                beta = _get_variable('beta', params_shape, initializer= tf.zeros_initializer)
                gamma = _get_variable('gamma', params_shape, initializer= tf.ones_initializer)

                moving_mean = _get_variable('moving_mean', params_shape, initializer= tf.zeros_initializer, trainable= False)
                moving_variance = _get_variable('moving_variance', params_shape, initializer= tf.ones_initializer, trainable= False)

                weights_param_list.append(beta)
                weights_param_list.append(gamma)
                weights_param_list.append(moving_mean)
                weights_param_list.append(moving_variance)

                # These ops will only be preformed when training.

                mean, variance = tf.nn.moments(x, axis)
#                mean = tf.cast(mean, tf.float16)
#                variance = tf.cast(variance, tf.float16)
                
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
#                update_moving_mean = tf.cast(update_moving_mean, tf.float16)
#                update_moving_variance = tf.cast(update_moving_variance, tf.float16)


                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_variance)

                batch_normalization = tf.cond(is_training, 
                                                                  lambda: tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon), 
                                                                  lambda: tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon))

                # mean, variance = tf.nn.moments(x, axis)
                # mean = tf.cast(mean, tf.float16)
                # variance = tf.cast(variance, tf.float16)

                # batch_normalization = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


        return batch_normalization

# convolution layer

# from keras
# layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
# keras.layers.Conv2D(filters, 
#                                  kernel_size, 
#                                  strides=(1, 1), 
#                                  padding='valid', 
#                                  data_format=None, 
#                                  dilation_rate=(1, 1), 
#                                  activation=None, 
#                                  use_bias=True, 
#                                  kernel_initializer='glorot_uniform', 
#                                  bias_initializer='zeros', 
#                                  kernel_regularizer=None, 
#                                  bias_regularizer=None, 
#                                  activity_regularizer=None, 
#                                  kernel_constraint=None, 
#                                  bias_constraint=None)

# from tf
# tf.nn.conv2d(
#     input,
#     filter,
#     strides,
#     padding,
#     use_cudnn_on_gpu=True,
#     data_format='NHWC',
#     dilations=[1, 1, 1, 1],
#     name=None
# )


def conv_layer(x, out_channels, ksizes, strides = (1, 1), padding = "VALID", scope = None, weights_param_list = None):

        with tf.variable_scope(scope):

                x_shape = x.get_shape()
                in_channels = x_shape[3]

#       according paper and keras api value
#                weight_initializer = tf.contrib.layers.xavier_initializer()
                weight_initializer = tf.glorot_uniform_initializer()

                shape = [ksizes[0], ksizes[1], in_channels, out_channels]
                weights = _get_variable('weights', shape, weight_initializer)

                weights_param_list.append(weights)

                conv = tf.nn.conv2d(x, weights, [1, strides[0], strides[1], 1], padding = padding)

        return conv

# separable convolution layer

# from keras
# layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
# keras.layers.SeparableConv2D(filters, 
#                                                  kernel_size, 
#                                                  strides=(1, 1),      
#                                                  padding='valid', 
#                                                  data_format=None, 
#                                                  dilation_rate=(1, 1), 
#                                                  depth_multiplier=1, 
#                                                  activation=None, 
#                                                  use_bias=True, 
#                                                  depthwise_initializer='glorot_uniform', 
#                                                  pointwise_initializer='glorot_uniform', 
#                                                  bias_initializer='zeros', 
#                                                  depthwise_regularizer=None, 
#                                                  pointwise_regularizer=None, 
#                                                  bias_regularizer=None, 
#                                                  activity_regularizer=None, 
#                                                  depthwise_constraint=None, 
#                                                  pointwise_constraint=None, 
#                                                  bias_constraint=None)

# from tf
# tf.nn.separable_conv2d(
#     input,
#     depthwise_filter,
#     pointwise_filter,
#     strides,
#     padding,
#     rate=None,
#     name=None,
#     data_format=None
# )

# we don't use bias and i don't know why yet
# not use l2 reg
# use glorot_uniform initializer
def separable_conv_layer(x, out_channels, ksizes, strides = (1, 1), padding = "VALID", scope = None, weights_param_list = None):

        with tf.variable_scope(scope):
                x_shape = x.get_shape()
                in_channels = x_shape[3]
                
                channel_multiplier = 1

#       according paper and keras api value
#                weight_initializer = tf.contrib.layers.xavier_initializer()
                weight_initializer = tf.glorot_uniform_initializer()

                depthwise_filter_shape = [ksizes[0], ksizes[1], in_channels, channel_multiplier]
                depthwise_filter = _get_variable('depthwise_weights', depthwise_filter_shape, weight_initializer)

                pointwise_filter_shape = [1, 1, channel_multiplier * in_channels, out_channels]
                pointwise_filter = _get_variable('pointwise_weights', pointwise_filter_shape, weight_initializer)

                weights_param_list.append(depthwise_filter)
                weights_param_list.append(pointwise_filter)

                separable_conv = tf.nn.separable_conv2d(x, depthwise_filter, pointwise_filter, strides = [1, strides[0], strides[1], 1], padding = padding, rate = (1, 1))

        return separable_conv

# Max Pooling Layer
def max_pool(x, ksizes, strides = (1, 1), padding = "VALID", scope = None):
        
        with tf.variable_scope(scope):
                max_pool = tf.nn.max_pool(x, ksize=[1, ksizes[0], ksizes[1], 1], strides=[1, strides[0], strides[1], 1], padding = padding)

        return max_pool

# Average Pooling Layer
def avg_pool(x, ksizes, strides = (1, 1), padding = "VALID", scope = None):
        
        with tf.variable_scope(scope):
                avg_pool = tf.nn.avg_pool(x, ksize=[1, ksizes[0], ksizes[1], 1], strides=[1, strides[0], strides[1], 1], padding = padding)
        
        return avg_pool










