
# coding: utf-8

# In[1]:


import tensorflow as tf
print("Tensorflow version " + tf.__version__)


# In[2]:


#Create placeholders for X and y
X = tf.placeholder(name='X', shape=(None,28,28,1), dtype=tf.float32)
y = tf.placeholder(name='y', shape=(None,10), dtype=tf.float32)


# In[3]:


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


# In[4]:


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

#use tf.name_scope to make you visualization more organized.
with tf.name_scope('conv_layers') as scope:
        A1 = tf.nn.relu(tf.nn.conv2d(input=X, filter=W1, strides=(1,1,1,1), padding='SAME')+B1)
        print(A1)
        P1 = tf.nn.max_pool(value=A1, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
        print(P1)
        A2 = tf.nn.relu(tf.nn.conv2d(input=P1, filter=W2, strides=(1,1,1,1), padding='VALID')+B2)
        print(A2)
        P2 = tf.nn.max_pool(value=A2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
        print(P2)
with tf.name_scope('fc_layers') as scope:
        P2_flatten = tf.contrib.layers.flatten(P2)
        print(P2_flatten)
        A3 = tf.nn.relu(tf.matmul(P2_flatten, W3) + B3)
        print(A3)
        A4 = tf.nn.relu(tf.matmul(A3, W4) + B4)
        print(A4)
        Z5 = tf.matmul(A4, W5) + B5
        print(Z5)
with tf.name_scope('output'):
        y_pred = tf.nn.softmax(Z5)
        print(y_pred)

#compute cost
with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=y))
    
    #Since we won't actually run this graph , we don't need to use the method 'summary', but if you want to keep track 
    #something while training, it is useful to use 'summary'
    tf.summary.scalar("cost_function", loss)
#define a step
with tf.name_scope("train") as scope:
    step = tf.train.AdamOptimizer(learning_rate=0.006).minimize(loss)
with tf.name_scope('accuracy') as scope:    
    #prediction
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    #compute accuracy    #use this to compute train or test accuracy.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
#merge all summaries
merged_summary = tf.summary.merge_all()


# In[6]:


with tf.Session() as sess:

    #file will be written to a folder named 'the_graph'
    writer = tf.summary.FileWriter('the_graph', sess.graph)
    writer.add_graph(graph=sess.graph)
    
    #For summary, we will use something like
    
    #summary,... = sess.run([merged_summary,...],...)
    #writer.add_summary(summary,...)
    
    print(writer.get_logdir())
    
    #just make sure everything is working
    writer.flush()
    writer.close()


# In[6]:


#if you want to rerun the code above, don't forget to clear up the previous graph first
tf.reset_default_graph()


# 在命令行模式中键入以下代码：
# 
# tensorboard --logdir=你储存的文件夹路径名
# 
# 注意：使用绝对路径可以保证读取到，不要引号，不要引号，不要引号！！！文件夹名字即上面print出来的结果。之后根据提示打开网页即可。
