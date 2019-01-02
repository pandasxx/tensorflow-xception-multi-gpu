import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def parser(example):

    features = {
                        'img': tf.FixedLenFeature((), tf.string, default_value=''),
                        'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

    feats = tf.parse_single_example(example, features)

    label = feats['label']

    img = tf.decode_raw(feats['img'], tf.float32)
    img = tf.reshape(img, [299,299,3])

    return img, label

def data_pipeline(file_tfrecords, batch_size):
    # buffer_size = 3GB, num_parallel_reads 8 = 4 > 2 > 1
    # num_parallel_reads = 4, buffer_size 3GB = 1GB > NONE

#    dt = tf.data.TFRecordDataset(file_tfrecords, buffer_size = 3 * 1024 * 1024 * 1024, num_parallel_reads = 4)
    dt = tf.data.TFRecordDataset(file_tfrecords, buffer_size = 1 * 1024 * 1024 * 1024, num_parallel_reads = 4)
    
    # 12 is faster then 1    
    dt = dt.map(parser, num_parallel_calls = 8)
    dt = dt.prefetch(buffer_size = batch_size * 3)
    dt = dt.shuffle(3000)
    dt = dt.repeat(1000)
    dt = dt.batch(batch_size)

    iterator = dt.make_one_shot_iterator()
    imgs, labels = iterator.get_next()

    # dt = tf.data.TFRecordDataset(file_tfrecords)
    # dt = dt.map(parser)
    # dt = dt.shuffle(buffer_size = batch_size)
    # dt = dt.repeat()
    # dt = dt.batch(batch_size)

    # iterator = dt.make_one_shot_iterator()
    # imgs, labels = iterator.get_next()

    return imgs, labels

def stats_sample_num(file_tfrecords):
    num_samples = 0

    for file_tfrecord in file_tfrecords:
        for record in tf.python_io.tf_record_iterator(file_tfrecord):
            num_samples += 1

    return num_samples