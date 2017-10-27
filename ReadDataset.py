import tensorflow as tf

def read_and_decode(filename_queue, real=0):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    if real:
        image_shape = tf.stack([560, 352, 3])
    else: 
        image_shape = tf.stack([352 * 560])
        label = tf.stack(tf.one_hot(label, 26, on_value=1, off_value=0))

    image = tf.reshape(image, image_shape)

    return tf.train.shuffle_batch(
        [image, label], 
        batch_size=10,
        num_threads=2,
        capacity=6000,
        min_after_dequeue=4000)


def load_labels_and_images(tfrecords_filename, real=0):
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=20)
    return read_and_decode(filename_queue, real)
    
def example_number(tfrecords_filename):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))

def show_labels_and_images(tfrecords_filename):
    import matplotlib.pyplot as plt
    images, labels = load_labels_and_images(tfrecords_filename, 1)

    with tf.Session()  as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in xrange(10):
            image, label = sess.run([images, labels])
            #print(label)
            plt.title(label)
            plt.imshow(image[0, :, :, :])
            plt.show()

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    show_labels_and_images('train-dataset.tfrecords')


