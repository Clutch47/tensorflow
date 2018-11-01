#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time

time_start = time.time()
input_train = pd.read_csv( 'data/train_64-32.csv', delimiter=",", header = None, dtype = np.int32).values
time_end = time.time()
print ('time cost read:', time_end - time_start, 's', end = " ")

time_start = time.time()
input_valid = pd.read_csv( 'data/valid_64-32.csv', delimiter=",", header = None, dtype = np.int32).values

# train dataset and corresponding labels
train_dataset = np.array(input_train).astype(np.float32) #read train_dataset by numpy
valid_dataset = np.array(input_valid).astype(np.float32) #read test_dataset by numpy

# Parameters
learning_rate = 0.01
seed = 1540616162.7054856
# tf Graph input 
X = tf.placeholder(tf.float32, shape=[2000, 64])
X_valid= tf.placeholder(tf.float32, shape=[1669, 64])

#Network parameters
n_input = 64 #input data: 64-bit 
print ('seed:', seed)
# hidden layer settings
n_hidden_1 = 32 # 1st layer: num of neurons

encoder_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], seed = seed)) #64*32
decoder_h1 = tf.transpose(encoder_h1) # get from transposed encoder_h2 4*4
encoder_b1 = tf.Variable(tf.random_normal([n_hidden_1], seed = seed)) #16
decoder_b1 = tf.Variable(tf.random_normal([n_input], seed = seed)) #4

# Building the encoder function
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),
                                   encoder_b1))
    return l1  

# Building the decoder function
def decoder(x):
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1),
                                   decoder_b1))
    return output

# Building the test function
def pred(x):
    # Encoder Hidden layer with sigmoid activation #1
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),
                                   encoder_b1))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, decoder_h1),
                                   decoder_b1))
    return l2       
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_train = decoder_op
y_pred = pred(X_valid)

# Targets (Labels) are the input data.
x_train = X
x_valid = X_valid

# Define cost and optimizer, minimize the squared error
diff_train = abs (y_train - x_train)
error_train = tf.reduce_sum(diff_train) #error train

diff_pred = abs (y_pred - x_valid)
error_pred = tf.reduce_sum(diff_pred)

cost = tf.reduce_mean(tf.square(diff_train))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #cost and optimizer 

# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer() #initialization
    sess.run(init) 
    saver_h1 = tf.train.Saver({'h1':encoder_h1})
    saver_b1 = tf.train.Saver({'b1':encoder_b1})
    saver_b4 = tf.train.Saver({'b4':decoder_b1})
    # Training cycle
    for epoch in range(6001):
        # Run optimization op (backprop) and cost op (to get cost value)
        _, c = sess.run([optimizer, cost], feed_dict={X: train_dataset})
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), sess.run(error_train, feed_dict={X: train_dataset}), sess.run(error_pred, feed_dict={X_valid: valid_dataset}))
    save_path = saver_h1.save(sess, 'save/save_h1/encoder_h1.ckpt', global_step = 6001)
    save_path = saver_b1.save(sess, 'save/save_b1/encoder_b1.ckpt', global_step = 6001)
    save_path = saver_b4.save(sess, 'save/save_b4/decoder_b1.ckpt', global_step = 6001)
    time_end = time.time()
    print ('time cost training:', time_end - time_start, 's')
    np.set_printoptions(threshold = np.NaN)
    print(sess.run(error_train, feed_dict={X: train_dataset}))
    print(sess.run(diff_train, feed_dict={X: train_dataset}))
    print(sess.run(error_pred, feed_dict={X_valid: valid_dataset}))
    print(sess.run(diff_pred, feed_dict={X_valid: valid_dataset}))
    print("central hidden layer:\n", sess.run(encoder_op, feed_dict={X: train_dataset}))
    print("output:\n", sess.run(decoder_op, feed_dict={X: train_dataset}))
    print("weight1:\n", sess.run(encoder_h1, feed_dict={X: train_dataset}))
    print("weight2:\n", sess.run(decoder_h1, feed_dict={X: train_dataset}))
    print("bias1:\n", sess.run(encoder_b1, feed_dict={X: train_dataset}))
    print("bias2:\n", sess.run(decoder_b1, feed_dict={X: train_dataset}))
