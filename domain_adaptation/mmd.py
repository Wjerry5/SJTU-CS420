import utils
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from functools import partial
import numpy as np
import random


np.random.seed(27)
random.seed(27)


# s data, usps
xs, ys, xs_test, ys_test = utils.load_s_usps()
# t data, mnist
xt, yt, xt_test, yt_test = utils.load_mnist()

# config

l2_param = 1e-4
lr = 1e-4
batch_size = 64
num_steps = 40000
mmd_param = 0.3
max_target_test_acc=0


with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 2025])
    y_ = tf.placeholder("float", shape=[None, 10])
    x_image = tf.reshape(x, [-1, 45, 45, 1])
    train_flag = tf.placeholder(tf.bool)

with tf.name_scope('feature_generator'):
    W_conv1 = utils.weight_variable([5, 5, 1, 32], 'conv1_weight')
    b_conv1 = utils.bias_variable([32], 'conv1_bias')
    h_conv1 = tf.nn.relu(utils.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = utils.max_pool_2x2(h_conv1)

    W_conv2 = utils.weight_variable([5, 5, 32, 64], 'conv2_weight')
    b_conv2 = utils.weight_variable([64], 'conv2_bias')
    h_conv2 = tf.nn.relu(utils.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = utils.max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
    W_fc1 = utils.weight_variable([12 * 12 * 64, 1024], 'fc1_weight')
    b_fc1 = utils.bias_variable([1024], 'fc1_bias')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('slice_data'):
    h_s = tf.cond(train_flag, lambda: tf.slice(h_fc1, [0, 0], [batch_size / 2, -1]), lambda: h_fc1)
    h_t = tf.cond(train_flag, lambda: tf.slice(h_fc1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h_fc1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_, [0, 0], [batch_size / 2, -1]), lambda: y_)

with tf.name_scope('mmd'):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = utils.maximum_mean_discrepancy(h_s, h_t, kernel=gaussian_kernel)
    mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)

with tf.name_scope('classifier'):
    W_fc2 = utils.weight_variable([1024, 10], 'fc2_weight')
    b_fc2 = utils.bias_variable([10], 'fc2_bias')
    pred_logit = tf.matmul(h_s, W_fc2) + b_fc2
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_logit, 1)), tf.float32))

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + mmd_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2)

    for i in range(num_steps):
        xs_batch, ys_batch = S_batches.next()
        xt_batch, yt_batch = T_batches.next()

        xb = np.vstack([xs_batch, xt_batch])
        yb = np.vstack([ys_batch, yt_batch])
		# train
        sess.run(train_op, feed_dict={x: xb, y_: yb, train_flag: True})

        if i % 100 == 0:
			# test
            acc, clf_ls = sess.run([clf_acc, clf_loss], feed_dict={x: xs_test, y_: ys_test, train_flag: False})
            acc_m, clf_ls_m = 0, 0
            for test_epoch in range(100):
                acc_m_, clf_ls_m_ = sess.run([clf_acc, clf_loss], feed_dict={x: xt_test[test_epoch:test_epoch + 100],
                                                                             y_: yt_test[test_epoch:test_epoch + 100],
                                                                             train_flag: False})
                acc_m += acc_m_
                clf_ls_m += clf_ls_m_
            acc_m /= 100.
            clf_ls_m /= 100.
            print('step', i, max_target_test_acc)
            print('source classifier loss: %f, source accuracy: %f' % (clf_ls, acc))
            print('target classifier loss: %f, target accuracy: %f ' % (clf_ls_m, acc_m))
            if acc_m > max_target_test_acc:
                max_target_test_acc = acc_m
	
    print('best t_acc', max_target_test_acc)