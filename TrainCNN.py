from datetime import datetime
import tensorflow as tf
import ReadDataset
import math
import sys

n_classes = 12
batch_size = 50
image_width = 352
image_height = 560
n_channels = 3
keep_rate = 0.8
n_iteration = 5000

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder('float', [None, (image_width*image_height*n_channels)], name="Input_Data")
y = tf.placeholder('float', [batch_size, n_classes], name="Label_Data")

def convolutional_layer(input, size_in, size_out, name):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("Weights", w)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("Activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def convolutional_neural_network(x):
    # Reshape Image
    input_layer = tf.reshape(x, shape=[-1, image_height, image_width, n_channels])
    image_summary = tf.summary.image("Example_Images", input_layer)

    # Convolution Layer
    convolution = convolutional_layer(input_layer, n_channels, 32, "Conv_1")
    convolution = convolutional_layer(convolution, 32, 64, "Conv_2")
    convolution = convolutional_layer(convolution, 64, 128, "Conv_3")
    convolution = convolutional_layer(convolution, 128, 256, "Conv_4")

    # Dense Layer
    pool4_flat = tf.reshape(convolution, [-1, int(image_width/16)*int(image_height/16)*256])
    dense = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu, name="Dense_Layer")

    # Dropout Layer
    dropout = tf.layers.dropout(inputs=dense, rate=keep_prob, name="Dropout_Layer")

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=n_classes, name="Logits_Layer")

    return logits

def train_neural_network(x):
    prediction = convolutional_neural_network(x)

    with tf.name_scope('Train'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
        tf.summary.scalar('Cross_Entropy', cross_entropy)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

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
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log/train".format(log_dir))
        valid_writer = tf.summary.FileWriter("./log/valid".format(log_dir))
        train_writer.add_graph(sess.graph)

        msg = '\n##### Convolutional Neural Network #####\nTrain Dataset Size: {} examples\nValidation Dataset Size: {} examples\nNumber of Interations: {}\n\nStarting at '
        sys.stdout.write(msg.format(train_num_examples, valid_num_examples, num_iteration))
        sys.stdout.write(datetime.today().strftime("%Y-%m-%d %H:%M"))
        sys.stdout.write('...\n')

        for i in range(n_iteration):
            batch_xs, batch_ys = sess.run([train_images, train_labels])
            vbatch_xs, vbatch_ys = sess.run([valid_images, valid_labels])

            feed_dict_train = {x: batch_xs, y: batch_ys, keep_prob: keep_rate}
            feed_dict_validate = {x: vbatch_xs, y: vbatch_ys, keep_prob: 1.0}

            summary, _, acc = sess.run([merged, optimizer, accuracy], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)
            train_writer.flush()
            
            if i % 20 == 0: 
                val_acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict_validate)
                epoch = int(i / int(train_num_examples/batch_size))  
                valid_writer.add_summary(summary, i)
                valid_writer.flush()

                msg = "Epoch {0}/{1} -- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%}, Step: {4}"
                print(msg.format(epoch + 1, int(math.ceil(num_iteration/(train_num_examples/batch_size))), acc, val_acc, i))
                saver.save(sess, 'model/soy-beans-model')

        coord.request_stop()
        coord.join(threads)

train_neural_network(x)