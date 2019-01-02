import numpy as np
import os
import tensorflow as tf
from PIL import Image
import random



def convert_label(image_path, class_label_dict):
        folder_name = image_path.split("/")[-2]
        label = class_label_dict[folder_name]

        return label

def convert_img(image_path):
        image = Image.open(image_path)
        resized_image = image.resize((299, 299), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32') / 255
        img_raw = image_data.tobytes()

        return img_raw

def make_one_tfrecord_file(images_path, filename, class_label_dict):
        writer = tf.python_io.TFRecordWriter(filename)
    
        count = 0
        for image_path in images_path:
                if (image_path != ''):
                        label = convert_label(image_path, class_label_dict)
                        img_raw = convert_img(image_path)

                        example = tf.train.Example(features=tf.train.Features(feature={
                                'label':
                                        tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                                'img':
                                        tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                        }))
                writer.write(example.SerializeToString())
                count += 1
                if (count % 1000 == 0):
                        print(count)
        writer.close()

def scan_all_files(filePath):
        
        fileList = []
        
        for top, dirs, nondirs in os.walk(filePath):
                for item in nondirs:
                        fileList.append(os.path.join(top, item))

        random.shuffle(fileList)

        return fileList

def make_tfrecord_files(src_path, dst_path, name, class_label_dict, num_split = 10000):

        images_path = scan_all_files(src_path)

        tfrecord_file_nums = int(len(images_path) / num_split) + 1

        for i in range(tfrecord_file_nums):
                filename = os.path.join(dst_path, name + '_%d' % (i) +'.tfrecords')
    
                print('make %d file' % (i))
                make_one_tfrecord_file(images_path[i*num_split : (i+1) * num_split], filename, class_label_dict)