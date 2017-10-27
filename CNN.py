import tensorflow as tf
import ReadDataset
from datetime import datetime;
from tensorflow.contrib.tensorboard.plugins import projector
import math
import sys

n_classes = 26
batch_size = 10
image_width = 352
image_height = 560
number_channels = 3

x = tf.placeholder('float', [None, (352*560)])
y = tf.placeholder('float', [batch_size, n_classes])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    input_layer = tf.reshape(x, shape=[-1, image_width, image_height, number_channels])

    # Convolution 1
    conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="SAME",
                activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolution 2
    conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="SAME",
                activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolution 3
    conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[5, 5],
                padding="SAME",
                activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Convolution 4
    conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=256,
                kernel_size=[5, 5],
                padding="SAME",
                activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool4_flat = tf.reshape(pool4,[-1, (image_width/4)*(image_height/4)*256])
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=26)

    return logits

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    train_images, train_labels = ReadDataset.load_labels_and_images('train-dataset.tfrecords')
    valid_images, valid_labels = ReadDataset.load_labels_and_images('valid-dataset.tfrecords')
    train_num_examples = ReadDataset.example_number('train-dataset.tfrecords')
    valid_num_examples = ReadDataset.example_number('valid-dataset.tfrecords')
    
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_iteration = 3000
        total_iterations = 0

        msg = '\n##### Convolutional Neural Network #####\nTrain Dataset Size: {} examples\nValidation Dataset Size: {} examples\nNumber of Interations: {}\n\nStarting at '
        sys.stdout.write(msg.format(train_num_examples, valid_num_examples, num_iteration))
        sys.stdout.write(datetime.today().strftime("%Y-%m-%d %H:%M"))
        sys.stdout.write('...\n')

        for i in range(total_iterations, total_iterations + num_iteration):
            batch_xs, batch_ys = sess.run([train_images, train_labels])
            vbatch_xs, vbatch_ys = sess.run([valid_images, valid_labels])

            feed_dict_train = {x: batch_xs, y: batch_ys}
            feed_dict_validate = {x: vbatch_xs, y: vbatch_ys}

            sess.run(optimizer, feed_dict=feed_dict_train)

            if i % 50 == 0: 
                val_loss = sess.run(cost, feed_dict=feed_dict_validate)
                acc = sess.run(accuracy, feed_dict=feed_dict_train)
                val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
                epoch = int(i / int(train_num_examples/batch_size))  

                msg = "Epoch {0}/{1} -- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
                print(msg.format(epoch + 1, int(math.ceil(num_iteration/(train_num_examples/batch_size))), acc, val_acc, val_loss))
                saver.save(sess, 'soy-beans-model')

        total_iterations += num_iteration

        coord.request_stop()
        coord.join(threads)

train_neural_network(x)