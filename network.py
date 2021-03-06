import tensorflow as tf
import tensorflow.contrib.slim as slim


receptive_field_size = 288
step_in_pixels = 32 # 2^(num of max pools)


def common_representation(inputs, lcr):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], reuse=tf.AUTO_REUSE, weights_initializer=tf.initializers.he_normal()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', padding='same')  # 288
            print('conv1: ' + str(net))
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            print('pool1: ' + str(net))
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', padding='same')  # 144
            print('conv2: ' + str(net))
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            print('pool2: ' + str(net))
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', padding='same')  # 72
            print('conv3: ' + str(net))
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            print('pool3: ' + str(net))
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', padding='same')  # 36
            print('conv4: ' + str(net))
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            print('pool4: ' + str(net))
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', padding='same')  # 18
            print('conv5: ' + str(net))
            net = slim.max_pool2d(net, [2, 2], stride=1, scope='pool5', padding='same')
            print('pool5: ' + str(net))

            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6_sub')
            net = slim.conv2d(net, 1024, [1, 1], scope='fc7_sub')

            with tf.variable_scope('conv6_2'):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                print('conv6/conv1x1: ' + str(net))
                net = tf.pad(net, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                print('conv6/pad: ' + str(net))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')  # 9
                print('conv6/conv3x3: ' + str(net))

            with tf.variable_scope('conv7_2'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                print('conv7/conv1x1: ' + str(net))
                net = tf.pad(net, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                print('conv7/pad: ' + str(net))
                net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3x3', padding='VALID')
                print('conv7/conv3x3: ' + str(net))

            # net = slim.max_pool2d(net, [2, 2], stride=1, scope='pool6', padding='same')

            with tf.variable_scope('conv8_2'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')  # 7

            # with tf.variable_scope('conv9_2'):
            #     net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            #     net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')

            with tf.variable_scope('new_layers'):
                net = slim.conv2d(net, 256, [3, 1], rate=2, scope='conv1', padding='valid')
                net = slim.conv2d(net, 256, [1, 3], rate=2, scope='conv2', padding='valid')
                net = slim.conv2d(net, 256, [3, 1], scope='conv3', padding='valid')
                net = slim.conv2d(net, 256, [1, 3], scope='conv4', padding='valid')
                net = slim.conv2d(net, lcr, [1, 1], scope='conv_last')

    return net


def prediction_path(net, opts, n_channles_last):
    with tf.variable_scope('vgg_16'):
        with tf.variable_scope('prediction'):
            with slim.arg_scope([slim.conv2d], reuse=tf.AUTO_REUSE, weights_initializer=tf.initializers.he_normal()):
                net = slim.conv2d(net, opts.lcr, [1, 1], scope='layer1')
                net = slim.conv2d(net, n_channles_last, [1, 1], activation_fn=None, scope='layer2')
    return net




