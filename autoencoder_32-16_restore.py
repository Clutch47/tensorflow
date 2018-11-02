#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time

# train dataset
time_start = time.time()
input_train = pd.read_csv('data/train_64-32.csv', delimiter=",", header = None, dtype = np.int32).values #load input of 64-32
input_valid = pd.read_csv('data/valid_64-32.csv', delimiter=",", header = None, dtype = np.int32).values #load input of 64-32
time_end = time.time()
print ('time cost read:', time_end - time_start, 's', end = ' ')

time_start = time.time()
# valid dataset
train_dataset = np.array(input_train).astype(np.float32) #read train_dataset by numpy
valid_dataset = np.array(input_valid).astype(np.float32) #read valid_dataset by numpy

# Parameters
learning_rate = 0.01
seed = time.time()
print ('seed:', seed)

# tf Graph input 
X_64 = tf.placeholder(tf.float32, shape=[2000, 64])
X_valid = tf.placeholder(tf.float32, shape=[1669, 64])
#Network parameters
n_input = 64 #input data: 64-bit 
n_hidden_1 = 32
n_hidden_2 = 16

encoder_h1 = tf.get_variable("encoder_h1", shape = [n_input, n_hidden_1]) #64*32
encoder_b1 = tf.get_variable("encoder_b1", shape = [n_hidden_1]) #32
decoder_h2 = tf.transpose(encoder_h1)
decoder_b2 = tf.get_variable("decoder_b2", shape = [n_input]) #64

encoder_h2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed = seed)) #32*16
decoder_h1 = tf.transpose(encoder_h2) # get from transposed encoder_h2 16*32
encoder_b2 = tf.Variable(tf.random_normal([n_hidden_2], seed = seed)) #16
decoder_b1 = tf.Variable(tf.random_normal([n_hidden_1], seed = seed)) #32

saver_h1 = tf.train.Saver({"h1": encoder_h1}) # declare tf.train.Saver save model
saver_b1 = tf.train.Saver({"b1": encoder_b1})
saver_b4 = tf.train.Saver({"b4": decoder_b2})

# Building the encoder function
def encoder1(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),encoder_b1))
    return l1
def encoder2(x):
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h2),encoder_b2))
    return hidden_layer

# Building the decoder function
def decoder1(x):
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1),decoder_b1))
    return l3
def decoder2(x):
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h2),decoder_b2))
    return output

# Construct model
encoder1_op = encoder1(X_64)
encoder2_op = encoder2(encoder1_op)
decoder1_op = decoder1(encoder2_op)
y_64 = decoder2(decoder1_op)
x_64 = X_64

# Building the valid function
def pred(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),encoder_b1))
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(l1, encoder_h2),encoder_b2))
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, decoder_h1),decoder_b1))
    output = tf.nn.sigmoid(tf.add(tf.matmul(l3, decoder_h2),decoder_b2))
    return output

# Compare initial input and current ouput
diff_train = abs(y_64 - x_64)
error_train = tf.reduce_sum(diff_train)

# Calculate prediction error
y_pred = pred (X_valid)
diff_pred = abs(y_pred - X_valid)
error_pred = tf.reduce_sum(diff_pred)

# Define cost and optimizer, minimize the squared error
cost = tf.reduce_sum(tf.square(diff_train)) #error train
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #cost and optimizer 

# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer() #initialization
    sess.run(init)
    saver_h1.restore(sess, "save/save_h1/encoder_h1.ckpt-6001") # Read session from saved path in hard disk
    saver_b1.restore(sess, "save/save_b1/encoder_b1.ckpt-6001")
    saver_b4.restore(sess, "save/save_b4/decoder_b1.ckpt-6001")
#    print ("bias4:\n", sess.run(decoder_b2))
    # Training cycle
    for epoch in range(6001):
        # Run optimization op (backprop) and cost op (to get cost value)
        _, c = sess.run([optimizer, cost], feed_dict={X_64: train_dataset})
        # Display logs per epoch step
        if epoch % 1 == 0:
            print('%04d' % (epoch + 1), sess.run(error_train, feed_dict={X_64: train_dataset}), sess.run(error_pred, feed_dict={X_valid: valid_dataset}))
    time_end = time.time()
    print ('time cost training:', time_end - time_start, 's')    
    np.set_printoptions(threshold = np.NaN)
    print ("diff_train:\n", sess.run(diff_train, feed_dict={X_64: train_dataset}))
    print ("error_train:\n", sess.run(error_train, feed_dict={X_64: train_dataset}))
    print ("diff_pred:\n", sess.run(diff_pred, feed_dict={X_valid: valid_dataset}))
    print ("error_pred:\n", sess.run(error_pred, feed_dict={X_valid: valid_dataset}))
    print ("h1:\n", sess.run(encoder1_op, feed_dict={X_64: train_dataset}))
    print ("central hidden layer:\n", sess.run(encoder2_op, feed_dict={X_64: train_dataset}))
    print ("h3:\n", sess.run(decoder1_op, feed_dict={X_64: train_dataset}))
    print ("output:\n", sess.run(y_64, feed_dict={X_64: train_dataset}))
    print ("weight1:\n", sess.run(encoder_h1, feed_dict={X_64: train_dataset}))
    print ("weight2:\n", sess.run(encoder_h2, feed_dict={X_64: train_dataset}))
    print ("weight3:\n", sess.run(decoder_h1, feed_dict={X_64: train_dataset}))
    print ("weight4:\n", sess.run(decoder_h2, feed_dict={X_64: train_dataset}))
    print ("bias1:\n", sess.run(encoder_b1, feed_dict={X_64: train_dataset}))
    print ("bias2:\n", sess.run(encoder_b2, feed_dict={X_64: train_dataset}))
    print ("bias3:\n", sess.run(decoder_b1, feed_dict={X_64: train_dataset}))
    print ("bias4:\n", sess.run(decoder_b2, feed_dict={X_64: train_dataset}))
    
