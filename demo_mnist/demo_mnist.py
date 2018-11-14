
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-


# In[ ]:


#import necessary libraries
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
zimport matplotlib.pyplot as plt
print("Tensorflow version " + tf.__version__)


# In[ ]:


#load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[ ]:


#load from mnist dataset
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


# In[ ]:


#reshape train_data, eval_data
train_data = np.reshape(train_data,[-1,28,28,1])
eval_data=np.reshape(eval_data,[-1,28,28,1])
print('Training set shape: ', np.shape(train_data))
print('Test set shape: ',np.shape(eval_data))


# In[ ]:


#copy and rename data
X_train = train_data
Y_train = train_labels
X_test = eval_data
Y_test = eval_labels


# In[ ]:


#Create placeholders for X and y
X = tf.placeholder(name='X', shape=(None,28,28,1), dtype=tf.float32)
y = tf.placeholder(name='y', shape=(None,10), dtype=tf.float32)


# In[ ]:


#initialize weights and bias
#using xavier initialization for weights and zero initialization for bias
W1 = tf.get_variable(name='W1', shape=(5,5,1,6), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name='W2', shape=(5,5,6,16), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable(name='W3', shape=(400,120), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable(name='W4', shape=(120,84), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable(name='W5', shape=(84,10), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.get_variable(name='B1', shape=(6), dtype=tf.float32, initializer=tf.zeros_initializer())
B2 = tf.get_variable(name='B2', shape=(16), dtype=tf.float32, initializer=tf.zeros_initializer())
B3 = tf.get_variable(name='B3', shape=(120), dtype=tf.float32, initializer=tf.zeros_initializer())
B4 = tf.get_variable(name='B4', shape=(84), dtype=tf.float32, initializer=tf.zeros_initializer())
B5 = tf.get_variable(name='B5', shape=(10), dtype=tf.float32, initializer=tf.zeros_initializer())


# In[ ]:


#CNN model - A 'modern' LeNet-5
#
#conv -> relu -> max pooling -> conv -> relu -> FC -> relu -> FC -> relu -> FC -> softmax
#
#input - m * 28 * 28 *1  
#
#convolutional layer 1 (relu) - 28 * 28 * 1 -> 28 * 28 * 6     stride=1, filter_size=5
#max pooling layer 1 - 28 * 28 * 6 -> 14 * 14 * 6              stride=2, filter_size=2
#
#convolutional layer 2 (relu) - 14 * 14 * 6 -> 10 * 10 * 16    stride=1, filter_size=5
#max pooling layer 2 - 10 * 10 * 16 -> 5 * 5 * 16              stride=2, filter_size=2
#
#flatten - 5 * 5 * 16 -> 400
#
#full-connected layer 1 (relu) - 400 -> 120
#
#full-connected layer 2 (relu) - 120 ->84
#
#full-connected layer 3 (softmax) - 84 -> 10

A1 = tf.nn.relu(tf.nn.conv2d(input=X, filter=W1, strides=(1,1,1,1), padding='SAME')+B1)
print(A1)
P1 = tf.nn.max_pool(value=A1, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
print(P1)
A2 = tf.nn.relu(tf.nn.conv2d(input=P1, filter=W2, strides=(1,1,1,1), padding='VALID')+B2)
print(A2)
P2 = tf.nn.max_pool(value=A2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
print(P2)
P2_flatten = tf.contrib.layers.flatten(P2)
print(P2_flatten)
A3 = tf.nn.relu(tf.matmul(P2_flatten, W3) + B3)
print(A3)
A4 = tf.nn.relu(tf.matmul(A3, W4) + B4)
print(A4)
Z5 = tf.matmul(A4, W5) + B5
print(Z5)
y_pred = tf.nn.softmax(Z5)
print(y_pred)

#compute cost
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=y))
#define a step
step = tf.train.AdamOptimizer(learning_rate=0.006).minimize(loss)
#prediction
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
#compute accuracy    #use this to compute train or test accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[ ]:


#initialize global variables in tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[ ]:


#Let's create the list storing cost first
costs=[]


# In[ ]:


#train the CNN with 250 iterations and learning_rate=0.006


for i in range(1,251):
    _, tmp_cost = sess.run([step,loss], feed_dict={X:X_train, y:Y_train})
    costs.append(tmp_cost)
    if i%5 == 0:
        print('cost after iteration {0} : {1}'.format(i, tmp_cost))

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.006))
plt.show()  


train_accuracy = accuracy.eval(session=sess, feed_dict={X: X_train, y: Y_train})
test_accuracy = accuracy.eval(session=sess, feed_dict={X: X_test, y: Y_test})
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[ ]:


#save the model
saver = tf.train.Saver()
path = saver.save(sess, './my_model.ckpt')
print(path)


# In[ ]:


#close the session
sess.close()


# In[ ]:


import tensorflow as tf
from tensorflow.python.framework import ops
#load the model 
#We will see the train accuracy is consistent with the result shown before
ops.reset_default_graph()
with tf.Session() as sess:
    #state = tf.train.get_checkpoint_state('./')
    #print(state)
    saver = tf.train.import_meta_graph('./my_model.ckpt.meta')
    saver.restore(sess,'./my_model.ckpt/')

    print(sess.graph)
    print(accuracy.graph)

    train_accuracy = accuracy.eval(session=sess, feed_dict={X: X_train, y: Y_train})
    test_accuracy = accuracy.eval(session=sess, feed_dict={X: X_test, y: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


# In[ ]:


def display_one_image(index = None):
    """
    Display a designated or random image and predicted label from the test set.

    Arguments:
    index -- an integer (0~9999) , index of the image

    Return:
    None
    """

    if index==None:
        index = np.random.randint(1,10000) - 1
        
    raw_image = np.reshape(X_test[index],[1,28,28,1])
    raw_label = Y_test[index]
    label = np.argmax(raw_label)
    image = np.reshape(raw_image,[28,28])
    plt.imshow(image)
    print('Image {} of 10000 from the test set.'.format(index+1))
    prediction = tf.argmax(y_pred, 1)
    tmp_predict = sess.run(prediction, feed_dict={X:raw_image})
    print('Label: {}'.format(label))
    print('Prediction: {}'.format(np.asscalar(tmp_predict)))


# In[ ]:


display_one_image(999)


# In[ ]:


def display_conv_relu_1(index=None):
    """
    Visualize the convolutional layer 1 (relu), using a designated or random image from test set

    Arguments:
    index -- an integer (0~9999) , index of the image

    Return:
    None
    """
  
    #running A1
    if index==None:
        index=np.random.randint(0,10000)
    conv_relu_1 = sess.run(A1, feed_dict={X:X_test})
    conv_relu_1_test = conv_relu_1[index]
    print('Showing image {0} of 10000 from the test set'.format(index+1))
    print('Label: {0}'.format(np.asscalar((np.argmax(Y_test[index])))))
    #create a dict storing the image we need
    conv_relu_1_dict={}
    for i in range(0,6):
        conv_relu_1_dict['{0}'.format(i)]=conv_relu_1_test[:,:,i]
    #displaying the images
    fig = plt.figure(figsize=[20,20])
    for i in range(0,6):
        fig.add_subplot(6,1,i+1)
        plt.imshow(np.reshape(conv_relu_1_dict['{0}'.format(i)],[28,28]))
    plt.show()
    fig.clear()
  
  
    


# In[ ]:


display_conv_relu_1(999)


# In[ ]:


def display_pool_1(index=None):
    """
    Visualize the max pooling layer 1, using a designated or random image from test set

    Arguments:
    index -- an integer (0~9999) , index of the image

    Return:
    None
    """
    #running P1
    if index==None:
        index=np.random.randint(0,10000)
    pool_1 = sess.run(P1, feed_dict={X:X_test})
    pool_1_test = pool_1[index]
    print('Showing image {0} of 10000 from the test set'.format(index+1))
    print('Label: {0}'.format(np.asscalar((np.argmax(Y_test[index])))))
    #create a dict storing the image we need
    pool_1_dict={}
    for i in range(0,6):
        pool_1_dict['{0}'.format(i)]=pool_1_test[:,:,i]
    #displaying the images
    fig = plt.figure(figsize=[20,20])
    for i in range(0,6):
        fig.add_subplot(6,1,i+1)
        plt.imshow(np.reshape(pool_1_dict['{0}'.format(i)],[14,14]))
    plt.show()
    fig.clear()


# In[ ]:


display_pool_1(999)


# In[ ]:


def display_conv_relu_2(index=None):
    """
    Visualize the convolutional layer 2 (relu), using a designated or random image from test set

    Arguments:
    index -- an integer (0~9999) , index of the image

    Return:
    None
    """
  
    #running P1
    if index==None:
        index=np.random.randint(0,10000)
    conv_relu_2 = sess.run(A2, feed_dict={X:X_test})
    conv_relu_2_test = conv_relu_2[index]
    print('Showing image {0} of 10000 from the test set'.format(index+1))
    print('Label: {0}'.format(np.asscalar((np.argmax(Y_test[index])))))
    #create a dict storing the image we need
    conv_relu_2_dict={}
    for i in range(0,16):
        conv_relu_2_dict['{0}'.format(i)]=conv_relu_2_test[:,:,i]
    #displaying the images
    fig = plt.figure(figsize=[50,50])
    for i in range(0,16):
        fig.add_subplot(16,1,i+1)
        plt.imshow(np.reshape(conv_relu_2_dict['{0}'.format(i)],[10,10]))
    plt.show()
    fig.clear()


# In[ ]:


display_conv_relu_2(999)


# In[ ]:


#get gpu info

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

