import tensorflow as tf
import ReadDataset

n_classes = 26
batch_size = 10
keep_rate = 0.7

x = tf.placeholder('float', [None, (375*250)])
y = tf.placeholder('float', [batch_size, n_classes])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
               'W_fc':tf.Variable(tf.random_normal([47*32*128,512])),
               'out':tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_fc':tf.Variable(tf.random_normal([512])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 375, 250, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv3,[-1, 47*32*128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    train_images, train_labels = ReadDataset.load_labels_and_images('train-dataset.tfrecords')
    valid_images, valid_labels = ReadDataset.load_labels_and_images('valid-dataset.tfrecords')
    
    hm_epochs = 100
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            batch_xs, batch_ys = sess.run([train_images, train_labels])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            epoch_loss += c

            print(prediction)
            print(y)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            vbatch_xs, vbatch_ys = sess.run([valid_images, valid_labels])
            valid_accuracy = accuracy.eval({x:vbatch_xs, y:vbatch_ys})
            train_accuracy = accuracy.eval({x:batch_xs, y:batch_ys})

            print('Epoch {}/{} -- Loss: {}, Validation Accuracy: {:.2%}, Training Accuracy: {:.2%}'
                  .format(epoch+1, hm_epochs, epoch_loss, valid_accuracy, train_accuracy))

        coord.request_stop()
        coord.join(threads)

train_neural_network(x)