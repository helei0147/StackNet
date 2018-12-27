import tensorflow as tf
import numpy as np
import glob
import os
from datetime import datetime
from scipy import misc

import deepReflMaps

# test data do not need to convert to tfrecord
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'data',
                           """Directory where to read model checkpoints.""")

def to_stack(image, side_len):
    result = []
    for i in range(image.shape[0]):
        canvas = image[i]
        buffer = []
        for m in range(side_len):
            for n in range(side_len):
                pic = canvas[m::side_len, n::side_len]
                buffer.append(pic)
        buffer = np.array(buffer)
        buffer = np.transpose(buffer, [1,2,0])
        result.append(buffer)
    return np.array(result)

if __name__ == '__main__':

    file_names = glob.glob('image_test')
    ph_size=128
    
    with tf.Graph().as_default():
        image_num = 7650
        image_buffer = []
        gt_buffer = []
        file_number = 8
        for i in range(file_number):
            image_buffer.append(np.load('image_test/%d.npy'%(i)))
            gt_buffer.append(np.load('label_test/%d.npy'%i))
        print()
        print('readin finished')
        print()
        gt = np.concatenate(gt_buffer)
        image = np.concatenate(image_buffer)
        ex_image_count = int(image_num/ph_size+1)*ph_size
        ex_image = np.zeros([ex_image_count, 256, 256], np.float32)
        ex_image[:image_num, :,:] = image
        image = to_stack(ex_image, 4)
        print(image.shape)
        image_number = image.shape[0]
        image_placeholder = tf.placeholder(tf.float32, shape=(ph_size,64, 64, 16))
        predict_op = deepReflMaps.inference(image_placeholder, train=False, AngFlag=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
                print('Model restored.')
            else:
                print('No checkpoint file found')

            image_outputs = []
            #for i in range(int(image_number/ph_size)):
            #    print(i)
            #    
            #    
            while True:
                index = input('please input image index:')
                index = int(index)
                im = np.zeros([ph_size, 64,64,16], np.float32)
                im[0,:,:,:] = image[index, :,:,:]
                feed_dict = {image_placeholder: im}
                vec = sess.run(predict_op, feed_dict=feed_dict)
                result = np.argmax(vec, axis=1)
                result = result[0]
                print('predicted:', result, ' ground truth: ', gt[index])
    predicted = np.array(image_outputs)
    predicted = np.concatenate(predicted)
    result = np.argmax(predicted,axis=1)
    
    result = result[:image_num]
    np.save('result.npy', result)
#    predict_outputs = np.concatenate(tuple(image_outputs), axis=0)
#    predict_outputs = predict_outputs[:file_number-1]



