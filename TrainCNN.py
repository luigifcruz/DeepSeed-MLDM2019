from datetime import datetime
import tensorflow as tf
import sys

# Training Features
batch_size = 10
test_size = 40
keep_rate = 0.8
n_iteration = 1
dense_size = 1024
learning_rate = 1e-4 #AdamOptimizer

# Dataset Features
n_classes = 12
n_channels = 3
image_width = 352
image_height = 560

LOGDIR = "/Users/luigifreitas/CNPq/Comum/tensorflow_cnn/logs/"

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder('float', [None, (image_width*image_height*n_channels)], name="Input_Data")
y = tf.placeholder('float', [None, n_classes], name="Label_Data")

def get_size(tfrecords_filename):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))

def get_single_features(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image_shape = tf.stack([image_width * image_height * n_channels])
    label = tf.stack(tf.one_hot(label, n_classes, on_value=1, off_value=0))
    image = tf.reshape(image, image_shape)

    return image, label

def get_shuffle_batch(tfrecords_filename, batch_size, name):
    with tf.name_scope(name):
        filename_queue = tf.train.string_input_producer([tfrecords_filename])
        image, label = get_single_features(filename_queue)

        return tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2, capacity=600, min_after_dequeue=400)

def get_batch(tfrecords_filename, batch_size, name):
    with tf.name_scope(name):
        filename_queue = tf.train.string_input_producer([tfrecords_filename])
        image, label = get_single_features(filename_queue)

        return tf.train.batch([image, label], batch_size=batch_size, allow_smaller_final_batch=True)

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
    with tf.name_scope("CNN"):
        # Reshape Image
        input_layer = tf.reshape(x, shape=[-1, image_height, image_width, n_channels])
        image_summary = tf.summary.image("Example_Images", input_layer)

        # Convolution Layer
        convolution = convolutional_layer(input_layer, n_channels, 32, "Conv_1")
        convolution = convolutional_layer(convolution, 32, 64, "Conv_2")
        convolution = convolutional_layer(convolution, 64, 128, "Conv_3")
        convolution = convolutional_layer(convolution, 128, 256, "Conv_4")

        # Flatten Layer
        input_size = int(image_width/16)*int(image_height/16)*256
        flattened = tf.reshape(convolution, [-1, input_size])

        # Dense Layer
        dense = tf.layers.dense(inputs=flattened, units=dense_size, activation=tf.nn.relu, name="Dense_Layer")

        # Dropout Layer
        dropout = tf.layers.dropout(inputs=dense, rate=keep_prob, name="Dropout_Layer")

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=n_classes, name="Logits_Layer")

        return logits, dense

def train_neural_network(x):
    prediction, dense_out  = convolutional_neural_network(x)

    with tf.name_scope('Train'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        tf.summary.scalar('Cross_Entropy', cross_entropy)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    train_images, train_labels = get_shuffle_batch('train-dataset.tfrecords', batch_size, 'Train_Dataset')
    valid_images, valid_labels = get_shuffle_batch('valid-dataset.tfrecords', batch_size, 'Valid_Dataset')
    test_images, test_labels = get_batch('test-dataset.tfrecords', test_size, 'Test_Dataset')
    train_num_examples = get_size('train-dataset.tfrecords')

    embedding = tf.Variable(tf.zeros([test_size, dense_size]), name="test_embedding")
    assignment = embedding.assign(dense_out)
    
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOGDIR + "train")
        valid_writer = tf.summary.FileWriter(LOGDIR + "validation")
        train_writer.add_graph(sess.graph)

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.metadata_path = LOGDIR + 'labels.tsv'
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_writer, config)

        msg = '\n##### Convolutional Neural Network #####\nTrain Dataset Size: {} examples\nNumber of Interations: {}\n\nStarting at '
        sys.stdout.write(msg.format(train_num_examples, n_iteration))
        sys.stdout.write(datetime.today().strftime("%Y-%m-%d %H:%M"))
        sys.stdout.write('...\n')

        for i in range(n_iteration):
            epoch = int(i / int(train_num_examples/batch_size))
            batch_xs, batch_ys = sess.run([train_images, train_labels])
            feed_dict_train = {x: batch_xs, y: batch_ys, keep_prob: keep_rate}

            summary, acc, _ = sess.run([merged, accuracy, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)
            train_writer.flush()

            if i % 100 == 0:
                print("Calculating and saving assignments...")
                tbatch_xs, tbatch_ys = sess.run([test_images, test_labels])
                sess.run(assignment, feed_dict={x: tbatch_xs, y: tbatch_ys, keep_prob: 1.0})

            if i % 50 == 0: 
                vbatch_xs, vbatch_ys = sess.run([valid_images, valid_labels])
                feed_dict_validate = {x: vbatch_xs, y: vbatch_ys, keep_prob: 1.0}

                val_acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict_validate)
                valid_writer.add_summary(summary, i)
                valid_writer.flush() 

                msg = "Epoch {}/{} -- Training Accuracy: {:>6.1%}, Validation Accuracy: {:>6.1%}, Step: {}"
                print(msg.format(epoch + 1, int(n_iteration/(train_num_examples/batch_size)), acc, val_acc, i))
                saver.save(sess, LOGDIR + 'model.ckpt', i)

        coord.request_stop()
        coord.join(threads)

train_neural_network(x)