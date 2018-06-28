import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

names = pd.read_csv('labels.csv')
names = names.id
no_of_images=names.shape[0]
names_no_taken=names[0:no_of_images] #names of number of images taken
a=[] #image array
for i in range(no_of_images):
 img=cv2.imread(names[i]+'.jpg')
 img = cv2.resize(img,(60,60),    interpolation =   cv2.INTER_CUBIC )
 a.append(img)
 
#cv2.imshow('image',a[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
 
def augment_brightness_camera_images(image):   #a function to randomly set brightness of a picture
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


#plt.imshow(a[2])

plt.axis('off')
#plt.imshow(a[2])
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
plt.figure(figsize=(12,12))

extra_names=[] #array to store names of extra generated pictures

for j in range(no_of_images):
  for i in range(5):   #generating extra 5 augmented imgaes for every single image
    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    img = transform_image(a[j],20,10,5,brightness=1)

    plt.subplot(10,10,i+1)
    #plt.imshow(img)
#    plt.axis('off')
#    plt.imshow(img)
    a.append(img)
    extra_names.append(names_no_taken[j])

names_no_taken=names_no_taken.tolist()
names_no_taken=names_no_taken+extra_names

   
    


