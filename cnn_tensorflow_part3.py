import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

names_breed=[]
names = pd.read_csv('labels.csv')
names1=names.id
names=names.values


for i in range(names.shape[0]):
  names_breed.append(names[i][1])
#names = names.id
no_of_images=1000   #taking only 1000 images though any number of images can be taken  


#names_no_taken=names[0:no_of_images]

names_1=names_breed[0:no_of_images]

    
    
a=[]
for i in range(no_of_images):
 img=cv2.imread(names1[i]+'.jpg')
 img = cv2.resize(img,(60,60),    interpolation =   cv2.INTER_CUBIC )
 a.append(img)
 
a=np.array(a)
    
no_of_images=len(a)

######################defining the layers

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name='w')


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name='b')


def deepnn(x):
  
  x_image = tf.reshape(x, [-1, 60, 60, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 3, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([15 * 15* 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 120])
  b_fc2 = bias_variable([120])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

##W####################################3


x = tf.placeholder(tf.float32, [None, 60,60,3])

  # Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 120])

y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  saver = tf.train.Saver()
names_1=np.array(names_1)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
en = LabelEncoder()
names_1 = en.fit_transform(names_1)
names_1 =names_1.reshape(1000,1)
enc=OneHotEncoder()

names_1=enc.fit_transform(names_1).toarray()
names_1_test=names_1[0:int(no_of_images*0.8)]
names_1_train=names_1[int(no_of_images*0.8):no_of_images]

writer=tf.summary.FileWriter('/home/aditya/Downloads/Downloads/check/')

a = np.array(a) 
a_test=a[0:int(no_of_images*0.8)]

a_train=a[int(no_of_images*0.8):no_of_images]

 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    writer.add_graph(sess.graph)
    writer.close()
    image_batch,label_batch = tf.train.batch([a,names_1], batch_size=32,enqueue_many=True,)

    image_batch = sess.run( image_batch)
    label_batch = sess.run( label_batch)

    for i in range(1000):
#        if i % 100 == 0:
       # sess.run(train_step, feed_dict={x:a, y_:names_1,keep_prob:0.5})
        _, loss_val = sess.run([train_step, cross_entropy],
                           feed_dict={x: a_test, y: names_1_test,keep_prob:0.5})
       
        train_accuracy = accuracy.eval(feed_dict={
            x: a_test, y: names_1_test, keep_prob: 1.0})
        
    print('step %d, training accuracy %g' % (i, train_accuracy))
        #train_step.run(feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
        
       # print loss_val
#        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
#        print('step %d, training accuracy %g' % (i, train_accuracy))
#        print('cost',train_step.run(feed_dict={x: image_batch[0], y_: label_batch[1], keep_prob: 0.5}))
#      if i%5==0:
#        saver.save(sess, '/home/aditya/Downloads/Downloads/check/',global_step=i)
#                                                                                                        

#      print cross_entropy
