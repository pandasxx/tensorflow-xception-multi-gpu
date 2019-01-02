import tensorflow as tf
from net.common_nn_cells import batch_normalization, conv_layer, separable_conv_layer, max_pool, avg_pool

def xception(inputs, num_classes, is_training = True, is_finetune = True):

        weights_param_list = []

        #===========ENTRY FLOW==============
        #Block 1
        with tf.variable_scope("block_1"):
                net = conv_layer(inputs, 32, (3, 3), strides = (2, 2), scope = "conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_1")

                net = conv_layer(net, 64, (3, 3), scope = "conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_2")

                residual = conv_layer(net, 128, (1,1), strides = (2, 2), padding='SAME', scope='res_conv', weights_param_list = weights_param_list)
                residual = batch_normalization(residual, is_training=is_training, scope="res_bn", weights_param_list = weights_param_list)

                print(residual.name, residual.shape)

        #Block 2
        with tf.variable_scope("block_2"):
                net = separable_conv_layer(net, 128, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_1")

                net = separable_conv_layer(net, 128, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)

                net = max_pool(net, (3,3), strides = (2, 2), padding = "SAME", scope = "pool")

                net = tf.add(net, residual, name='add')

                residual = conv_layer(net, 256, (1,1), strides = (2, 2), padding='SAME', scope='res_conv', weights_param_list = weights_param_list)
                residual = batch_normalization(residual, is_training=is_training, scope="res_bn", weights_param_list = weights_param_list)

                print(residual.name, residual.shape)


        #Block 3
        with tf.variable_scope("block_3"):
                net = tf.nn.relu(net, name= "relu_1")

                net = separable_conv_layer(net, 256, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_2")

                net = separable_conv_layer(net, 256, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)

                net = max_pool(net, (3,3), strides = (2, 2), padding = "SAME", scope = "pool")

                net = tf.add(net, residual, name='add')

                residual = conv_layer(net, 728, (1,1), strides = (2, 2), padding='SAME', scope='res_conv', weights_param_list = weights_param_list)
                residual = batch_normalization(residual, is_training=is_training, scope="res_bn", weights_param_list = weights_param_list)

                print(residual.name, residual.shape)

        #Block 4
        with tf.variable_scope("block_4"):

                net = tf.nn.relu(net, name= "relu_1")

                net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_2")

                net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)
                
                net = max_pool(net, (3,3), strides = (2, 2), padding = "SAME", scope = "pool")                

                net = tf.add(net, residual, name='add')

                print(net.name, net.shape)

        #===========MIDDLE FLOW===============
        #Block 5 ~ 12
        for i in range(8):
                block_prefix = 'block_%s' % (str(i + 5))

                with tf.variable_scope(block_prefix):
                        residual = net

                        net = tf.nn.relu(net, name= "relu_1")

                        net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                        net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                        net = tf.nn.relu(net, name= "relu_2")

                        net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                        net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)
                        net = tf.nn.relu(net, name= "relu_3")

                        net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_3", weights_param_list = weights_param_list)
                        net = batch_normalization(net, is_training = is_training, scope = "bn_3", weights_param_list = weights_param_list)
                
                        net = tf.add(net, residual, name='add')

                        print(net.name, net.shape)

        #========EXIT FLOW============
        #Block 13
        with tf.variable_scope("block_13"):
                residual = conv_layer(net, 1024, (1,1), strides = (2, 2), padding='SAME', scope='res_conv', weights_param_list = weights_param_list)
                residual = batch_normalization(residual, is_training=is_training, scope="res_bn", weights_param_list = weights_param_list)

                net = tf.nn.relu(net, name= "relu_1")

                print(net.name, net.shape)

        #Block 14
        with tf.variable_scope("block_14"):
                net = separable_conv_layer(net, 728, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_1")

                net = separable_conv_layer(net, 1024, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)

                net = max_pool(net, (3,3), strides = (2, 2), padding = "SAME", scope = "pool")                

                net = tf.add(net, residual, name='add')

                print(net.name, net.shape)

        #Block 15
        with tf.variable_scope("block_15"):
                net = separable_conv_layer(net, 1536, [3,3], padding = "SAME", scope = "dws_conv_1", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_1", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_1")

                net = separable_conv_layer(net, 2048, [3,3], padding = "SAME", scope = "dws_conv_2", weights_param_list = weights_param_list)
                net = batch_normalization(net, is_training = is_training, scope = "bn_2", weights_param_list = weights_param_list)
                net = tf.nn.relu(net, name= "relu_2")

                print(net.name, net.shape)

        #Block 16
        with tf.variable_scope("block_16"):
                net = avg_pool(net, (10, 10), scope = "avg_pool")

                net = conv_layer(net, 2048, (1, 1), scope = "conv_1", weights_param_list = weights_param_list)
                logits = conv_layer(net, num_classes, [1,1], scope='conv_2', weights_param_list = weights_param_list)
                logits = tf.squeeze(logits, [1,2], name='logits') #Squeeze height and width only

                print(logits.name, logits.shape)

        return logits, weights_param_list