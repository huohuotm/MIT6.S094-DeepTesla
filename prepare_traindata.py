import sys
import os
import time
import subprocess as sp
import itertools
import cv2
import numpy as np
import utils
import params 
import imageio
from utils import without_ext,ext,mkv_to_mp4
import h5py


def crop_resize_image(img):
# Crop
shape = img.shape
img_crop = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]

# Resize the image
img_resize = cv2.resize(img_crop, (200, 66), interpolation=cv2.INTER_AREA)

assert img_resize.shape == (66,200,3)

return img_resize

def YUV_normal_image(img,YUV=True):
if YUV:
    # RGB to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Normalize, need int???
    return img_yuv/255.0
else:
    return img/255.0



def add_shadow(img,orient='vertical'):
h, w = img.shape[:2]

if orient=='vertical':
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1) # slope
    b = - k * x1      # y intercept 
    for i in range(h):
        c = int((i - b) / k)
        img[i, :c, :] = (img[i, :c, :] * .5) #.astype(np.int32)

elif orient=='horizontal':
    [y1, y2] = np.random.choice(h, 2, replace=False)
    k = w / (y2 - y1) # slope
    b = - k * y1      # x intercept 
    for i in range(w):
        c = int((i - b) / k)
        img[:c, i, :] = (img[:c, i, :] * .5) #.astype(np.int32)

return img


def add_shadow_images(idx, images, steerings, add_new=True):

from math import ceil 
rand_idx = np.random.choice(len(idx), ceil(len(idx)*0.2), replace=False)
shadow_idx = idx[rand_idx]

if add_new:
    
    idx_v = shadow_idx[:len(shadow_idx)//2]  
    idx_h = shadow_idx[len(shadow_idx)//2:]

    # add vertical shadow
    images_new_v = [add_shadow(img,'vertical') for img in images[idx_v]] # list
    images_new_v = np.stack(images_new_v, axis=0)   #numpy array
    steerings_new_v = steerings[idx_v]
    assert images_new_v.shape[1:]==(66,200,3)
    assert steerings_new_v.shape[0] == images_new_v.shape[0]
    
    # add horizontal shadow
    images_new_h = [add_shadow(img,'horizontal') for img in images[idx_h]] # list
    images_new_h = np.stack(images_new_h, axis=0)   #numpy array
    steerings_new_h = steerings[idx_h]
    assert images_new_h.shape[1:]==(66,200,3)
    assert steerings_new_h.shape[0] == images_new_h.shape[0]
    
    # combine
    images = np.concatenate((images, images_new_v,images_new_h), axis=0)
    steerings = np.concatenate((steerings,steerings_new_v,steerings_new_h), axis=0)
    assert images.shape[1:]==(66,200,3)
    assert steerings.shape[0] == images.shape[0]
    
    return images, steerings
else:
    cnt = 0
    for i in shadow_idx:
        if cnt< (len(shadow_idx)//2):
            images[i] = add_shadow(images[i],'vertical')
            
        else:
            images[i] = add_shadow(images[i],'horizontal')
        cnt+=1
    assert images.shape[1:]==(66,200,3)
    assert steerings.shape[0] == images.shape[0]
    
    return images, steerings



def process_epoch_data(epoch_id, YUV=True, flip=True, shadow=True):
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_mkv_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    
    # convert mkv to mp4
    vid_mp4_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mp4'.format(epoch_id))
    if not os.path.exists(vid_mp4_path):
        mkv_to_mp4(vid_mkv_path, remove_mkv=False)    
    
    # slice mp4 to images
    vid = imageio.get_reader(vid_mp4_path,'ffmpeg')
    images = []
    for num in range(len(vid)):
#     for num in range(200):        
        images.append(vid.get_data(num))
    print('epoch{:0>2}_front.mkv has {} images'.format(epoch_id,len(images)))
    
    ## Crop and Resize before data agumentation
    images = [crop_resize_image(img) for img in images]
    images = np.stack(images, axis=0)
    
    
    # get steerings from csv file
    steerings = process_epoch_csv(epoch_id)
    
    print("images: ",len(images))
    print("steerings: ",steerings.shape)
    print("\t")

    # do flip agumentation
    if flip:
        print('--------------------start flip images--------')
        t0 = time.time()
        # steering not equal 0, flip
        flip_idx = np.where(steerings!=0)[0]  
        images_flip = images[flip_idx, :, ::-1, :]
        steering_flip = -steerings[flip_idx]
        
        print("images_flip num: ", images_flip.shape)
        print("steering_flip num: ", steering_flip.shape)
        
        # combine images_flip and images
        images = np.concatenate((images, images_flip), axis=0)
        steerings = np.concatenate((steerings,steering_flip), axis=0)
        
        print("combine images num: ", images.shape)
        print("combine steerings num: ", steerings.shape)
        
        print("---------------------flip images using time: ",time.time()-t0)
    
    
    # shadow agumentation
    if shadow:
        print("-------------------start adding shadow-------")
        t0 = time.time()
        # abs(steering) > 2, copy images, add shadowon new iamges
        idx_new = np.where(np.abs(steerings)>2)[0] 
        # add shadow if num >3
        if len(idx_new)>3:
            images, steerings = add_shadow_images(idx_new, images, steerings, add_new=True)
        # abs(steering) <= 2, add shadow on original images
        idx_old = np.where(np.abs(steerings)<=2)[0]  
        if len(idx_old)>3:
            images, steerings = add_shadow_images(idx_old, images, steerings, add_new=False)
        
        print("images num after shadow : ", images.shape)
        print("steerings num after shadow: ", steerings.shape)
        print("--------------------adding shadow using time: ",time.time()-t0)

    
    #yuv, normalization        
    images = [YUV_normal_image(img, YUV) for img in images]
    images = np.stack(images, axis=0)
    print("epoch {} has {} images after process: \n \n ".format(epoch_id,images.shape[0]))
                       
    assert np.max(images) <=1
    assert np.min(images) >=0
    return images,steerings


# -----------------------call function: process_epoch_data --------------------
epoch_ids=range(1,10)

images_train=[]
steerings_train =[]
for epoch_id in epoch_ids:
    images, steerings = process_epoch_data(epoch_id)
    images_train.append(images)
    steerings_train.append(steerings)
    
images_train = np.concatenate(images_train, axis=0)
steerings_train = np.concatenate(steerings_train, axis=0)

del images
del steerings   


print("after process images_train", images_train.shape)
print("after process steerings_train", steerings_train.shape)

# ---------------------- sample when abs(steering)=0.5 -------------------------
idx_sample = np.where(np.abs(steerings_train)==0.5)[0]
mask = np.random.choice(len(idx_sample), len(idx_sample)-2400*2, replace=False)
images_train = np.delete(images_train, mask, axis=0)
steerings_train = np.delete(steerings_train, mask, axis=0)

# --------------------------shuffle --------------------------------------------
np.random.seed(0)
permutation = list(np.random.permutation(steerings_train.shape[0]))
images_train = images_train[permutation,:,:,:]
steerings_train = steerings_train[permutation].reshape(-1,1)

print("final images_train", images_train.shape)
print("final steerings_train", steerings_train.shape)






