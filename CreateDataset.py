import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from glob import glob
import numpy as np
import random
import sys
import os

# Dataset Config
train_size = 0.90
test_size = 0.1
width = 352
height = 560

# Files Config
datasets_dir = './datasets/'
train_tfrecords_filename = 'train-dataset.tfrecords'
valid_tfrecords_filename = 'valid-dataset.tfrecords'
test_tfrecords_filename = 'test-dataset.tfrecords'

test_labels = 'labels-test.tsv'
dataset_labels = 'labels-dataset.txt'
np_test_labels = 'labels-test.dat'

plot_filename = 'dataset-plot.png'

dataset_path = '../../Datasets/soy_beans/secundario'


def generate_plot(n_dir, n_sample, dir_n):
    fig, ax = plt.subplots()
    index = np.arange(n_dir)
    bar_width = 0.8
    opacity = 0.4

    rects1 = plt.bar(index, n_sample, bar_width, alpha=opacity, color='b', label='Samples')

    plt.xlabel('Groups')
    plt.ylabel('Samples')
    plt.title('Dataset Analytics')
    plt.xticks(index, dir_n)
    plt.legend()

    plt.tight_layout()
    plt.savefig(datasets_dir + plot_filename, dpi=250)

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
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    train_writer = tf.python_io.TFRecordWriter(datasets_dir + train_tfrecords_filename)
    valid_writer = tf.python_io.TFRecordWriter(datasets_dir + valid_tfrecords_filename)
    test_writer = tf.python_io.TFRecordWriter(datasets_dir + test_tfrecords_filename)
    dataset_list = sorted(glob('{}/*/'.format(dataset_path)))

    n_dir = 0
    n_sample = []
    dir_n = []

    with open(datasets_dir + test_labels, 'w') as test_l, open(datasets_dir + dataset_labels, 'w') as dataset_l:
        for i, path in enumerate(dataset_list):
            print('Packing folder: {}/{}'.format(i+1, len(dataset_list)))
            ft_list = sorted(glob('{}*'.format(path)), key=lambda k: random.random())
            dir_name = os.path.basename(os.path.normpath(path))

            train_data = ft_list[:int((len(ft_list)+1)*train_size)]
            valid_data = ft_list[int(len(ft_list)*train_size+1):]
            test_data = ft_list[:int((len(ft_list)+1)*test_size)]

            dataset_l.write("{}:{}\n".format(i, dir_name))

            dir_n.append(dir_name)
            n_sample.append(0);
            n_dir += 1

            for j, image in enumerate(train_data):
                sys.stdout.write('\r    Train images: {}/{}'.format(j+1, len(train_data)))
                pack_image(image, i, train_writer)
                sys.stdout.flush()
                n_sample[-1] += 1

            sys.stdout.write('\n')

            for j, image in enumerate(valid_data):
                sys.stdout.write('\r    Validation images: {}/{}'.format(j+1, len(valid_data)))
                pack_image(image, i, valid_writer)
                sys.stdout.flush()
                n_sample[-1] += 1

            sys.stdout.write('\n')

            for j, image in enumerate(test_data):
                sys.stdout.write('\r    Test images: {}/{}'.format(j+1, len(test_data)))
                pack_image(image, i, test_writer)
                test_l.write("{}\n".format(dir_name))
                sys.stdout.flush()

            sys.stdout.write('\n')
    
    generate_plot(n_dir, n_sample, dir_n)
    np.array(dir_n).tofile(datasets_dir + np_test_labels)

    train_writer.close()
    valid_writer.close()
    test_writer.close()
    sys.stdout.flush()

if __name__ == "__main__":
    create_dataset()