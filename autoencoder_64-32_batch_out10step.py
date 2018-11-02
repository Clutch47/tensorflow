#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

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
epoch_step = 15000
batch_size = 1
total_batch = 2000
display_step = 1
save_step = 10
seed = time.time()
# tf Graph input 
X = tf.placeholder(tf.float32, shape=[1, 64])
X_valid= tf.placeholder(tf.float32, shape=[1669, 64])
X_train = tf.placeholder(tf.float32, shape=[2000, 64])

#Network parameters
n_input = 64 #input data: 64-bit 
print ('seed:', seed)
# hidden layer settings
n_hidden_1 = 32 # 1st layer: num of neurons

encoder_h1 = tf.get_variable("encoder_h1", shape = [n_input, n_hidden_1]) #64*32
encoder_b1 = tf.get_variable("encoder_b1", shape = [n_hidden_1]) #32
decoder_h1 = tf.transpose(encoder_h1)
decoder_b1 = tf.get_variable("decoder_b2", shape = [n_input]) #64

saver_h1 = tf.train.Saver({"h1": encoder_h1}) # declare tf.train.Saver save model
saver_b1 = tf.train.Saver({"b1": encoder_b1})
saver_b2 = tf.train.Saver({"b2": decoder_b1})
# Building the encoder function
def encoder(x):
    # Encoder Hidden layer with sigmoid activation
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1), encoder_b1))
    return l1  

# Building the decoder function
def decoder(x):
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1), decoder_b1))
    return output

# Building the test function
def pred_1(x):
    # Encoder Hidden layer with sigmoid activation #1
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1), encoder_b1))
    return l1
def pred_2(x):
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1), decoder_b1))
    return l2       
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_train_batch = decoder_op
x_train_batch = X

y_pred = pred_2 (pred_1 (X_valid))
x_valid = X_valid

l1 = pred_1(X_train)
y_train = pred_2(l1)
x_train = X_train

# Define cost and optimizer, minimize the squared error
diff_train_batch = abs (y_train_batch - x_train_batch)
error_train_batch = tf.reduce_sum(diff_train_batch) #error train

diff_pred = abs (y_pred - x_valid)
error_pred = tf.reduce_sum(diff_pred)

diff_train = abs (y_train - x_train)
error_train = tf.reduce_sum(diff_train)

cost = tf.reduce_mean(tf.square(diff_train_batch))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #cost and optimizer 

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initialization
    saver_h1.restore(sess, "save/each_1000/m1/h1/h1.ckpt-1000") # Read session from saved path in hard disk
    saver_b1.restore(sess, "save/each_1000/m1/b1/b1.ckpt-1000")
    saver_b2.restore(sess, "save/each_1000/m1/b2/b2.ckpt-1000")
    saver_h1 = tf.train.Saver({'h1':encoder_h1})
    saver_b1 = tf.train.Saver({'b1':encoder_b1})
    saver_b2 = tf.train.Saver({'b2':decoder_b1})
    # Training cycle
    for epoch in range (epoch_step):
        j = 0
        for i in range (total_batch):
            # Run optimization op (backprop) and cost op (to get cost value)
            j = j + 1
            _, c = sess.run([optimizer, cost], feed_dict={X: np.reshape(train_dataset[i], (1, 64))})
            # Display logs per epoch step
            if epoch % display_step == 0 and j == total_batch:
                print('%04d' % (epoch + 1), sess.run(error_train, feed_dict = {X_train: train_dataset}), sess.run(error_pred, feed_dict = {X_valid: valid_dataset}))
                
            if (epoch + 1) % save_step == 0 and j == total_batch:
                s = epoch + 1
                base = '/home/yu/tensorflow/yu/64-32_w/save/each_1000-15000/m1'
                os.chdir(base)
                s_str = str(s)
                os.mkdir(s_str)
                os.mkdir(s_str + "/h1")
                os.mkdir(s_str + "/b1")
                os.mkdir(s_str + "/b2")

                os.chdir(s_str)
                save_path = saver_h1.save(sess, 'h1/h1.ckpt', global_step = s)
                save_path = saver_b1.save(sess, 'b1/b1.ckpt', global_step = s)
                save_path = saver_b2.save(sess, 'b2/b2.ckpt', global_step = s)
                
                f = open('output.txt', 'a')
                np.set_printoptions (threshold = np.NaN)
                
                print ("training error:", file = f)
                train_error = sess.run(error_train, feed_dict = {X_train: train_dataset})
                print (train_error, file = f)

                print ("train difference:", file = f)
                train_diff = sess.run(diff_train, feed_dict = {X_train: train_dataset})
                print (train_diff, file = f)
                
                print ("prediction error:", file = f)
                pred_error = sess.run(error_pred, feed_dict = {X_valid: valid_dataset})
                print (pred_error, file = f)

                print ("pred difference:", file = f)
                pred_diff = sess.run(diff_pred, feed_dict = {X_valid: valid_dataset})
                print (pred_diff, file = f)
                
                print ("center:", file = f)
                center = sess.run(l1, feed_dict = {X_train: train_dataset})
                print (center, file = f)
                
                print ("weight1:", file = f)
                w1 = sess.run(encoder_h1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w1, file = f)

                print ("weight2:", file = f)
                w2 = sess.run(decoder_h1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w2, file = f)

                print ("bias1:", file = f)
                b1 = sess.run(encoder_b1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b1, file = f)

                print ("bias2:", file = f)
                b2 = sess.run(decoder_b1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b2, file = f)
                '''
                print ("output:", file = f)
                output = sess.run(y_train, feed_dict = {X_train: train_dataset})
                print (output, file = f)
                
                print ("pred_out:", file = f)
                pred_out = sess.run(y_pred, feed_dict = {X_valid: valid_dataset})
                print (pred_out, file = f)
                '''
                f.close()
    time_end = time.time()
    print ('time cost training:', time_end - time_start, 's')

