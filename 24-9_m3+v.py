#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

# train dataset
time_start = time.time()
input_train = pd.read_csv('data/train_64-32.csv', delimiter=",", header = None, dtype = np.int32).values #load input of 64-32
input_valid = pd.read_csv('data/valid_64-32.csv', delimiter=",", header = None, dtype = np.int32).values #load input of 64-32
#input_center = pd.read_csv('data/cen_64-24.csv', delimiter=",", header = None, dtype = np.float32).values 
time_end = time.time()
print ('time cost read:', time_end - time_start, 's', end = ' ')

time_start = time.time()
# valid dataset
train_dataset = np.array(input_train).astype(np.float32) #read train_dataset by numpy
valid_dataset = np.array(input_valid).astype(np.float32) #read valid_dataset by numpy
#center_dataset = np.array(input_center).astype(np.float32)

# Parameters
learning_rate = 0.01
epoch_step = 100000
batch_size = 1
total_batch = 2000
display_step = 1
save_step = 10
seed = time.time()
print ('seed:', seed)

# tf Graph input 
X_train = tf.placeholder(tf.float32, shape=[2000, 64])
X_valid = tf.placeholder(tf.float32, shape=[1669, 64])
#X_center = tf.placeholder(tf.float32, shape=[2000, 24])
X = tf.placeholder(tf.float32, shape=[1, 64])
#Network parameters
n_input = 64 
n_hidden_1 = 24
n_hidden_2 = 9

encoder_h1 = tf.get_variable("encoder_h1", shape = [n_input, n_hidden_1]) #64*24
encoder_b1 = tf.get_variable("encoder_b1", shape = [n_hidden_1]) #24
decoder_h2 = tf.transpose(encoder_h1) #24*64
decoder_b2 = tf.get_variable("decoder_b2", shape = [n_input]) #64

encoder_h2 = tf.get_variable("encoder_h2", shape = [n_hidden_1, n_hidden_2]) #24*9
decoder_h1 = tf.transpose(encoder_h2) # get from transposed encoder_h2 9*24
encoder_b2 = tf.get_variable("encoder_b2", shape = [n_hidden_2]) #9
decoder_b1 = tf.get_variable("decoder_b1", shape = [n_hidden_1]) #24

saver_h1 = tf.train.Saver({"h1": encoder_h1}) # declare tf.train.Saver save model
saver_b1 = tf.train.Saver({"b1": encoder_b1}) 
saver_b4 = tf.train.Saver({"b2": decoder_b2})

saver_h2 = tf.train.Saver({"h2": encoder_h2}) # declare tf.train.Saver save model
saver_b2 = tf.train.Saver({"b2": encoder_b2})
saver_b3 = tf.train.Saver({"b3": decoder_b1})

# Building the encoder function
def encoder1(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),encoder_b1))
    return l1
def encoder2(x):
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h2),encoder_b2))
    return hidden_layer
def decoder1(x):
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1),decoder_b1))
    return l3
def decoder2(x):
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h2),decoder_b2))
    return output

# Building the valid function
def pred1(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1), encoder_b1))
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(l1, encoder_h2),encoder_b2))
    return hidden_layer
def pred2(x):
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_h1), decoder_b1))
    output = tf.nn.sigmoid(tf.add(tf.matmul(l3, decoder_h2), decoder_b2))
    return output

# Construct model
encoder1_op = encoder1(X)
encoder2_op = encoder2(encoder1_op)
decoder1_op = decoder1(encoder2_op)
decoder2_op = decoder2(decoder1_op)
y_64 = decoder2_op
x_64 = X

# Compare train_64 and current ouput of train_64
diff_train_64 = abs(y_64 - x_64)
error_train_64 = tf.reduce_sum(diff_train_64)

# Calculate training error
encoder2_op = pred1 (X_train)
y_train = pred2 (encoder2_op)
diff_train = abs(y_train - X_train)
error_train = tf.reduce_sum(diff_train)

# Calculate prediction error
y_pred = pred2 (pred1 (X_valid))
diff_pred = abs(y_pred - X_valid)
error_pred = tf.reduce_sum(diff_pred)

# Define cost and optimizer, minimize the squared error
cost = tf.reduce_sum(tf.square(diff_train_64)) 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #cost and optimizer 

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initialization
    
    saver_h1.restore(sess, "save/64-24_m9/11300/h1/h1.ckpt-11300") # Read session from saved path in hard disk
    saver_b1.restore(sess, "save/64-24_m9/11300/b1/b1.ckpt-11300")
    saver_b4.restore(sess, "save/64-24_m9/11300/b2/b2.ckpt-11300")

    saver_h2.restore(sess, "save/m3/4660/h2/h2.ckpt-4660") 
    saver_b2.restore(sess, "save/m3/4660/b2/b2.ckpt-4660")
    saver_b3.restore(sess, "save/m3/4660/b3/b3.ckpt-4660")

    saver_h1 = tf.train.Saver({'h1':encoder_h1})
    saver_b1 = tf.train.Saver({'b1':encoder_b1})
    saver_b4 = tf.train.Saver({'b4':decoder_b2})
    
    saver_h2 = tf.train.Saver({'h2':encoder_h2})
    saver_b2 = tf.train.Saver({'b2':encoder_b2})
    saver_b3 = tf.train.Saver({'b3':decoder_b1})
    
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
                base = '/home/yu/tensorflow/autoEncoder/24-9/save/m3+v' #save
                os.chdir(base)
                s_str = str(s)
                os.mkdir(s_str)
                os.mkdir(s_str + "/h1")
                os.mkdir(s_str + "/b1")
                os.mkdir(s_str + "/b4")
                os.mkdir(s_str + "/h2")
                os.mkdir(s_str + "/b2")
                os.mkdir(s_str + "/b3")

                os.chdir(s_str)
                save_path = saver_h1.save(sess, 'h2/h2.ckpt', global_step = s)
                save_path = saver_b2.save(sess, 'b2/b2.ckpt', global_step = s)
                save_path = saver_b3.save(sess, 'b3/b3.ckpt', global_step = s)
                save_path = saver_h2.save(sess, 'h1/h1.ckpt', global_step = s)
                save_path = saver_b1.save(sess, 'b1/b1.ckpt', global_step = s)
                save_path = saver_b4.save(sess, 'b4/b4.ckpt', global_step = s)

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
                center = sess.run(encoder2_op, feed_dict = {X_train: train_dataset})
                print (center, file = f)
                
                print ("train output:", file = f)
                train_output = sess.run(y_train, feed_dict = {X_train: train_dataset})
                print (train_output, file = f)

                print ("pred output:", file = f)
                pred_out = sess.run(y_pred, feed_dict = {X_valid: valid_dataset})
                print (pred_out, file = f)

                print ("weight1:", file = f)
                w1 = sess.run(encoder_h1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w1, file = f)

                print ("weight2:", file = f)
                w2 = sess.run(encoder_h2, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w2, file = f)
                
                print ("weight3:", file = f)
                w3 = sess.run(decoder_h1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w3, file = f)

                print ("weight4:", file = f)
                w4 = sess.run(decoder_h2, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (w4, file = f)

                print ("bias1:", file = f)
                b1 = sess.run(encoder_b1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b1, file = f)

                print ("bias2:", file = f)
                b2 = sess.run(encoder_b2, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b2, file = f)

                print ("bias3:", file = f)
                b3 = sess.run(decoder_b1, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b3, file = f)

                print ("bias4:", file = f)
                b4 = sess.run(decoder_b2, feed_dict = {X: np.reshape(train_dataset[i], (1, 64))})
                print (b4, file = f)
                f.close()         
    time_end = time.time()
    print ('time cost training:', time_end - time_start, 's')
