from builtins import range
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from net.xception import xception
import os
import time
slim = tf.contrib.slim

from data.data_pipeline import data_pipeline, stats_sample_num

from weights_parse.parse_pretrain_weights import load_weights_param_from_npz, check_weights_param

# use multi tfrecord and multiple read obtain higher performance

# single tfrecord file
# tfrecord_file_list = ["/home/nvme/cell_data_test_0.tfrecords"]
# multi tfrecord file
tfrecord_file_list = ["/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords",
                               "/home/nvme/cell_data_test_0.tfrecords"]

# remote train data
# tfrecord_file_list = ["/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords",
#                                "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=data_samba/cell_data_test_0.tfrecords"]
#tfrecord_file_list = ["/dev/shm/cell_data_test_0.tfrecords"]
#tfrecord_file_list = ["/dev/shm/cell_data_test_0.tfrecords", "/dev/shm/cell_data_test_0.tfrecords", "/dev/shm/cell_data_test_0.tfrecords"]

num_gpus = 4

batch_size = 64
epochs = 10
num_classes = 5

weights_file_imagenet = "/home/hdd0/Develop/algo/xception-tf/tensorflow-xception-multigpu-base/weights_file/xception_weights_tf_imagenet_ordering_dict_notop.npz"

def tower_loss(scope, images, one_hot_labels, istraining):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

#    images = tf.cast(images, tf.float16)
#    one_hot_labels = tf.cast(one_hot_labels, tf.float16)

    # Build inference Graph.
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    logits, weights_param_list = xception(images, num_classes = num_classes, is_training = True)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)

    predicted = tf.nn.softmax(logits, name = "predicted")
    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

    return loss, accuracy, weights_param_list

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:            
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def build_train_op(images_batch, labels_batch):

    global_step = tf.Variable(0, trainable=False)
    opt = tf.train.AdamOptimizer()
    tower_grads = []
    tower_losses = []
    tower_accuracies = []
    weights_param_lists = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    istraining = True
                    loss, accuracy, weights_param_list = tower_loss(scope, images_batch[i], labels_batch[i], istraining)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)
                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    tower_losses.append(loss)
                    tower_accuracies.append(accuracy)
                    weights_param_lists.append(weights_param_list)

    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
    
    #with tf.control_dependencies(update_op):
    #    apply_gradient_op = opt.minimize(loss, 
    #                                     global_step = global_step, 
    #                                     var_list = vars_det)

    # Group all updates to into a single train op.
    with tf.control_dependencies(update_op):
        train_op = apply_gradient_op

    return train_op, tower_losses, tower_accuracies, weights_param_lists

def train(train_op, tower_losses, tower_accuracies, weights_param_lists):
    # cal image num
    images_num = stats_sample_num(file_tfrecords = tfrecord_file_list)
    num_batches_per_epoch = images_num // batch_size
    num_epochs = epochs

    print(images_num)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    gs = 0

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(init)

#    load_weights_param_from_npz(sess, npz_file = weights_file_imagenet, weights_param_list = weights_param_lists[0])
#    check_weights_param(sess, weights_param_lists = weights_param_lists)

    avg_losses_r = [0, 0, 0, 0]
    avg_accuracies_r = [0, 0, 0, 0]

    for i in range(gs, 10000):
        
        start_time = time.time()
        
        _ = sess.run(train_op)

        duration = time.time() - start_time
        print(i, "step time : ", duration)

        if i / 20 != 0 and i % 20 == 0:
            r = sess.run(tower_losses + tower_accuracies)

            avg_losses_r =  [avg_losses_r[i] + r[i] for i in range(num_gpus)]
            avg_accuracies_r = [avg_accuracies_r[i] + r[i + num_gpus] for i in range(num_gpus)]

            if i / 100 != 0 and i % 100 == 0:
                print("##################################")
                print("###### step", i, ",", i * batch_size, "images ######")
                for gpu_id, loss_r in enumerate(avg_losses_r):
                    print('tower loss ', gpu_id, ": ", loss_r / 5.0)
                for gpu_id, accuracy_r in enumerate(avg_accuracies_r):
                    print('tower accuracy ', gpu_id, ": ", accuracy_r / 5.0)
                print("##################################")

                avg_losses_r =  [0, 0, 0, 0]
                avg_accuracies_r = [0, 0, 0, 0]

def make_train_data_batch():
    images, labels = data_pipeline(file_tfrecords = tfrecord_file_list, batch_size = batch_size)
    labels = tf.one_hot(labels, num_classes)

    images = tf.reshape(images, [batch_size, 299, 299, 3])
    labels = tf.reshape(labels, [batch_size, 5])

    sub_batch_size = int(batch_size // num_gpus)

    images_batch = []
    labels_batch = []
    for i in range(num_gpus):
        images_batch.append(tf.slice(images, [sub_batch_size * i, 0, 0, 0],     [sub_batch_size, 299, 299, 3]))
        labels_batch.append(tf.slice(labels, [sub_batch_size * i, 0],     [sub_batch_size, 5]))

        print(images_batch[i].shape)
        print(labels_batch[i].shape)

    return images_batch, labels_batch

def run():

    #======================= TRAINING PROCESS =========================
    # start to construct the graph and build our model
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #create the dataset and load one batch
        images_batch, labels_batch = make_train_data_batch()

        # build train op
        train_op, tower_losses, tower_accuracies, weights_param_lists = build_train_op(images_batch, labels_batch)

        # train
        train(train_op, tower_losses, tower_accuracies, weights_param_lists)

        print("Complete!!")

if __name__ == '__main__':
    run()

