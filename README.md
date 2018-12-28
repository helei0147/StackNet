# StackNet

light field image patch dataset is available at https://drive.google.com/open?id=1zdeWF2pUkYc5CIkMYV1tLP1BLw1NgpEa
for each image patch, the BRDF classification index is at https://drive.google.com/open?id=1UD68yuO3ulj9YGJcSYFkxWPao8Rz4w00

This dataset is proposed at "Identifying Surface BRDF From a Single 4-D Light Field Image via Deep Neural Network"
In this folder, we have 48 npy files containing 47650 light field image patches.
0.npy to 46.npy, each file have 1000 lightfield image patches with shape 256*256*3. Image patch have three channels, RGB.
47.npy contains the last 650 image patches.
In folder "patches" display the image patches in 0.npy.

For training, we use tfrecord to contain labels and images. tfrecords are available at
https://pan.baidu.com/s/1KStWD4FzU8dEATdLZSQr3g
these tfrecords have grayscale image sets with shape 64*64*16, representing height/width/channel_count.
and matching material index label of each grayscale light field image.
