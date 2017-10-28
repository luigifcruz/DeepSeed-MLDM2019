import tensorflow as tf
from PIL import Image
from glob import glob
import numpy as np
import random
import sys

train_size = 0.90
test_size = 0.2
width = 352
height = 560
train_tfrecords_filename = 'train-dataset.tfrecords'
valid_tfrecords_filename = 'valid-dataset.tfrecords'
test_tfrecords_filename = 'test-dataset.tfrecords'
dataset_path = '../../Banco_Imagens/Sementes_Soja/Secundario'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def pack_image(path, label, writer):
    img = Image.open(path)
    img = img.resize((width, height), Image.BILINEAR)
    img_raw = np.array(img).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(img_raw),
        'label': _int64_feature(label)
    }))
        
    writer.write(example.SerializeToString())

def create_dataset():
    train_writer = tf.python_io.TFRecordWriter(train_tfrecords_filename)
    valid_writer = tf.python_io.TFRecordWriter(valid_tfrecords_filename)
    test_writer = tf.python_io.TFRecordWriter(test_tfrecords_filename)
    dataset_list = sorted(glob('{}/*/'.format(dataset_path)))

    with open('./logs/labels.tsv', 'w') as f:
        for i, path in enumerate(dataset_list):
            print('Packing folder: {}/{}'.format(i+1, len(dataset_list)))
            ft_list = sorted(glob('{}*'.format(path)), key=lambda k: random.random())
            train_data = ft_list[:int((len(ft_list)+1)*train_size)]
            valid_data = ft_list[int(len(ft_list)*train_size+1):]
            test_data = ft_list[:int((len(ft_list)+1)*test_size)]

            for j, image in enumerate(train_data):
                sys.stdout.write('\r    Train images: {}/{}'.format(j+1, len(train_data)))
                pack_image(image, i, train_writer)
                sys.stdout.flush()

            sys.stdout.write('\n')

            for j, image in enumerate(valid_data):
                sys.stdout.write('\r    Validation images: {}/{}'.format(j+1, len(valid_data)))
                pack_image(image, i, valid_writer)
                sys.stdout.flush()

            sys.stdout.write('\n')

            for j, image in enumerate(test_data):
                sys.stdout.write('\r    Test images: {}/{}'.format(j+1, len(test_data)))
                pack_image(image, i, test_writer)
                f.write('%d\n' % i)
                sys.stdout.flush()

            sys.stdout.write('\n')
        
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    sys.stdout.flush()

if __name__ == "__main__":
    create_dataset()