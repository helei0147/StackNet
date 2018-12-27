import tensorflow as tf
import numpy as np
import glob
import time
import os

def get_tfrecords(image):

    image_uint8 = tf.cast(image, tf.uint8)

    return image_uint8

def lf_to_stack(image, side_len):
    height = int(image.shape[0]/side_len)
    width = int(image.shape[1]/side_len)
    image_stack = np.zeros([height,width, side_len*side_len])
    for i in range(side_len):
        for j in range(side_len):
            image_stack[:,:,i*side_len+j] = image[i::side_len, j::side_len]
    return image_stack

def write_records_file(record_location, data_type):
    if not os.path.exists(record_location):
        os.makedirs(record_location)
    if data_type == 'train':
        image_filenames = glob.glob("../source/image_train/*.npy")
        label_filenames = glob.glob('../source/label_train/*.npy')
    elif data_type == 'test':
        image_filenames = glob.glob("../source/image_test/*.npy")
        label_filenames = glob.glob('../source/label_test/*.npy')
    else:
        print('train or test?')
        return 0
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(tf.uint8, shape = (64, 64, 16))
        tfrecord = get_tfrecords(image_placeholder)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        writer = None
        current_index = 0
        counter = 0


        for npy_index in range(len(image_filenames)):
            start_time = time.time()
            image_filename = image_filenames[npy_index]
            label_filename = label_filenames[npy_index]

            image_buffer = np.load(image_filename)
            print(image_buffer.shape, 'image_buffer size')
            label_buffer = np.load(label_filename)
            print('label_buffer.shape', label_buffer.shape)

            record_filename = '%s/%d.tfrecord'%(record_location, npy_index)
            writer = tf.python_io.TFRecordWriter(record_filename)



            for i in range(image_buffer.shape[0]):
                img = image_buffer[i,:,:]
                img_stack = lf_to_stack(img, 4)
                label = label_buffer[i]
                feed_dict = {image_placeholder: img_stack}

                record_value = sess.run([tfrecord], feed_dict=feed_dict)

                image_bytes = record_value[0].tobytes()
                label_int64 = label

                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_int64])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))
                writer.write(example.SerializeToString())

            counter += 1
            print('file %d complete, used %.2fs '%(npy_index, float(time.time())-start_time))
            writer.close()
        sess.close()

if __name__=='__main__':
    write_records_file("../StackNet/train", 'train')
    #write_records_file("../tfr/test/testing-images", 'test')
