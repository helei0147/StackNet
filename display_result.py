import numpy as np
import cv2
import sys,os

images = []
for i in range(8):
    images.append(np.load('image_test/%d.npy'%(i)))
images = np.concatenate(images)
print(images.shape)
result = np.load('result.npy')
count = result.shape[0]
gt = []
for i in range(8):
    gt.append(np.load('label_test/%d.npy'%(i)))
gt = np.concatenate(gt)
print(gt)
print(gt.shape)
for i in range(5):
    folder = 'result_im/%d/'%i
    if not os.path.exists(folder):
        os.makedirs(folder)    
# save images
for i in range(count):
    cv2.imwrite('result_im/%d/%d_gt%d.png'%(result[i],i, gt[i]), images[i])

gt = gt[:count]
acc_count = np.sum(gt==result)
acc = acc_count/count
print('acc: %.2f, accurate count: %d'%(acc, acc_count))
