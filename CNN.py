import tensorflow as tf
import ReadDataset
from datetime import datetime;
import math
import sys

n_classes = 26
batch_size = 10

x = tf.placeholder('float', [None, (352*560)])
y = tf.placeholder('float', [batch_size, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([5,5,128,256])),
               'W_fc':tf.Variable(tf.random_normal([22*35*256,512])),
               'out':tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_conv4':tf.Variable(tf.random_normal([256])),
              'b_fc':tf.Variable(tf.random_normal([512])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 352, 560, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool2d(conv4)

    fc = tf.reshape(conv4,[-1, 22*35*256])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    drop_out = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(drop_out, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.99).minimize(cost)

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

            feed_dict_train = {x: batch_xs, y: batch_ys, keep_prob: 0.8}
            feed_dict_validate = {x: vbatch_xs, y: vbatch_ys, keep_prob: 1.0}

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