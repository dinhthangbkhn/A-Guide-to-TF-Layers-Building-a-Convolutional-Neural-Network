from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def random_dataset(train, test):
    lengh = len(train)
    shuffle_id = np.arange(lengh)
    shuffle_id = np.random.shuffle(shuffle_id)
    train_shuffled = train[shuffle_id]
    test_shuffled = test[shuffle_id]
    return train_shuffled, test_shuffled


def create_placeholder():
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.int32, [None])
    return X,Y

def build_model(X):
    conv1 = tf.layers.conv2d(X,32,kernel_size=[5,5],strides=(1,1),padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=10)
    # logits = tf.argmax(logits, axis=1)

    return logits

def compute_cost(logits, labels):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits= logits))
    return cost

def model(input_train, labels_train, input_test, labels_test,
          learning_rate = 0.001, num_epochs = 100, minibatch_size = 64 ):
    X, Y = create_placeholder()

    logits = build_model(X)

    cost = compute_cost(logits, Y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    predict_op = tf.cast(tf.argmax(logits, 1), tf.int32)
    correct_prediction = tf.cast(tf.equal(predict_op, Y), tf.float32)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # input_train, labels_train = random_dataset(input_train, labels_train)
        for epoch in range(1,num_epochs+1):
            minibatch_cost = 0
            minibatch_acc = 0
            num_minibatches = int(len(input_train)/minibatch_size)
            # print(len(input_train))
            for m_index in range(num_minibatches):
                batch_X = input_train[m_index*minibatch_size : (m_index+1)*minibatch_size]
                batch_Y = labels_train[m_index*minibatch_size : (m_index+1)*minibatch_size]
                _, cost_val, acc_val = sess.run([optimizer, cost, acc], feed_dict={X:batch_X, Y: batch_Y})
                minibatch_cost += cost_val/num_minibatches
                minibatch_acc += acc_val/num_minibatches
            print('Epoch: ',epoch,', Loss: ',minibatch_cost,', Acc: ',minibatch_acc)

        #evaluate:

        print('Test acc: ',acc.eval({X:input_test[0:minibatch_size], Y: labels_test[0:minibatch_size]}))

def main():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.float32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = np.reshape(train_data, [-1,28,28,1])
    eval_data = np.reshape(eval_data, [-1,28,28,1])
    # print(train_data.shape)
    model(train_data, train_labels, eval_data, eval_labels, num_epochs=2)
if __name__ == '__main__':
    main()