import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
#import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

from sklearn import svm
clf = svm.SVC()

names = pd.read_csv('labels.csv')
names1=names.id #id of the pictures
names_breed=names.breed[0:10200] #names of the breed

#importing the version 2 pretrained inception network
module = hub.Module("https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1")
height, width = hub.get_expected_image_size(module)


images=[]
for i in range(names.shape[0]):
  img=cv2.imread(names1[i]+'.jpg')
  img = cv2.resize(img,(height,width),    interpolation =   cv2.INTER_CUBIC )
  images.append(img)
 
images=np.array(images) #an array of images  



features_all=np.zeros((10200,1024))

with tf.Session() as sess:   
    sess.run(tf.global_variables_initializer())   

    no_of_batches=255 #255 batches of 40 datapoints
    for i in range(no_of_batches):
         images_batch=images[i*40:i*40+40]

         print i
 

         features = module(images_batch)
         array = features.eval()
         features_all[i*40:i*40+40,:] = array
         
    X_train, X_test, y_train, y_test = train_test_split(features_all, names_breed, test_size=0.2, random_state=42)
  
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
 



pd.DataFrame(features_all).to_csv("feature.csv")