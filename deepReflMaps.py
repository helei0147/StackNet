import tensorflow as tf
import numpy as np
import glob
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_dir', 'data',
                            """Number of images to process in a batch.""")

NUM_EPOCHS_PER_DECAY = 50
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.01
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1200
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 300
MOVING_AVERAGE_DECAY = 0.9999
TOWER_NAME = 'tower'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=tf.float32)
    return var



def inputs(eval_data):
    if not eval_data:
        filenames = glob.glob("../train/*.tfrecord")
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = glob.glob("../tfr/test/*.tfrecord")
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST

    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [64, 64, 16])

    min_queue_examples = int(num_examples_per_epoch * 0.4)

    images_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)
    #tf.image_summary('images', images_batch)

    return images_batch, label_batch


def inference(images, train=True, AngFlag=True):
    if AngFlag:
        with tf.variable_scope('conv0') as scope:
            initializer = tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            kernel = _variable_on_cpu('weigths',
                                      shape=[4, 4, 1, 64],
                                      initializer=initializer)

            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv0 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv0)
    else:
        with tf.variable_scope('conv0') as scope:
            initializer = tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            kernel = _variable_on_cpu('weigths',
                                      shape=[4, 4, 16, 64],
                                      initializer=initializer)

            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv0 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv0)

    with tf.variable_scope('conv1') as scope:
        initializer = tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
        kernel = _variable_on_cpu('weigths',
                                  shape=[5, 5, 64, 64],
                                  initializer=initializer)


        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv0, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                      beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:

        kernel = _variable_on_cpu('weigths',
                                  shape=[5, 5, 64, 128],
                                  initializer=initializer)

        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # pool1
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm2
    norm2 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                      beta=0.75, name='norm2')

    with tf.variable_scope('conv3') as scope:

        kernel = _variable_on_cpu('weigths',
                                  shape=[3, 3, 128, 128],
                                  initializer=initializer)

        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # norm3
    norm3 = tf.nn.lrn(conv3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                      beta=0.75, name='norm3')
    # pool2
    pool2 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')


    # fc1
    with tf.variable_scope('fc1') as scope:

        initializer = tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32)
        conv5_reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = conv5_reshape.get_shape()[1].value
        weigths = _variable_on_cpu('weigths',
                                   shape=[dim, 384],
                                   initializer=initializer)

        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(tf.matmul(conv5_reshape, weigths), biases)
        fc1 = tf.nn.relu(bias, name=scope.name)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.6)
        _activation_summary(fc1)


    with tf.variable_scope('fc2') as scope:

        weigths = _variable_on_cpu('weigths',
                                   shape=[384, 192],
                                   initializer=initializer)

        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(tf.matmul(fc1, weigths), biases)
        fc2 = tf.nn.relu(bias, name=scope.name)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.6)
        _activation_summary(fc2)

    with tf.variable_scope('softmax_linear') as scope:
        initializer = tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32)
        weigths = _variable_on_cpu('weigths',
                                   shape=[192, 5],
                                   initializer=initializer)
        biases = _variable_on_cpu('biases', [5], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weigths), biases, name=scope.name)
        _activation_summary(softmax_linear)


    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def error_rate(logits, labels):
    logits = tf.argmax(logits, axis=1)
    logits = tf.cast(logits, tf.int32)
    error_count = tf.count_nonzero(tf.subtract(labels, logits))
    return tf.cast(error_count, tf.float32) / FLAGS.batch_size



def training(loss, global_step):
    #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                global_step,
    #                                decay_steps,
    #                                LEARNING_RATE_DECAY_FACTOR,
    #                                staircase=True)

    #train_op = tf.train.GradientDescentOptimizer(
    #    lr).minimize(loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return train_op
